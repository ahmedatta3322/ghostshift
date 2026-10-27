"""Shared pieces of the GhostShift pipeline (used by track.py; diff.py still carries its own copies)."""
import glob, pathlib, re, warnings
import numpy as np, requests
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
import astropy.units as u
from reproject import reproject_interp
from scipy import ndimage

warnings.filterwarnings("ignore")

PIX_ARCSEC = 6.2
# FLAGS bits that mean "do not trust this pixel" (MP_* keys in the FLAGS header). Not 12/14/21,
# and NOT 19 (OUTLIER): SPHEREx flags a pixel as outlier when it disagrees with other visits of the same
# sky, which is exactly what a moving asteroid does. Our every-exposure test handles cosmic rays instead.
BAD_MASK = sum(1 << b for b in [0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 15, 17])


def make_grid(ra, dec, size_deg, pix_arcsec=PIX_ARCSEC):
    npix = int(size_deg * 3600 / pix_arcsec)
    grid = WCS(naxis=2)
    grid.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    grid.wcs.crval = [ra, dec]
    grid.wcs.crpix = [npix / 2 + 0.5, npix / 2 + 0.5]
    grid.wcs.cdelt = [-pix_arcsec / 3600, pix_arcsec / 3600]
    return grid, (npix, npix)


def header_time(path):
    with fits.open(path) as h:
        return Time(h["IMAGE"].header["DATE-OBS"])


def list_files(cutout_dir, week=None):
    files = sorted(glob.glob(str(pathlib.Path(cutout_dir) / "*.fits")))
    if week:
        files = [f for f in files if pathlib.Path(f).name.startswith(week)]
    return files


def cluster_visits(files, times, max_gap_days=0.1):
    """Split files into visits whose exposures are within max_gap_days of each other."""
    fs = sorted(files, key=lambda f: times[f].mjd)
    visits, cur = [], [fs[0]]
    for f in fs[1:]:
        if times[f].mjd - times[cur[-1]].mjd <= max_gap_days:
            cur.append(f)
        else:
            visits.append(cur); cur = [f]
    visits.append(cur)
    return visits


def load_clean(path):
    with fits.open(path) as h:
        img = h["IMAGE"].data.astype(float)
        flags = h["FLAGS"].data
        zodi = h["ZODI"].data.astype(float) if "ZODI" in h else 0.0
        wcs = WCS(h["IMAGE"].header)
    img = img - zodi
    img[(flags & BAD_MASK) != 0] = np.nan
    img[~np.isfinite(img)] = np.nan
    _, med, _ = sigma_clipped_stats(img, sigma=3.0)
    return img - med, wcs


def reproject_stack(files, grid, shape):
    stack = []
    for f in files:
        img, wcs = load_clean(f)
        arr, _ = reproject_interp((img, wcs), grid, shape_out=shape, order="bilinear")
        stack.append(arr)
    return np.array(stack)


def combine(stack, min_hits=2):
    """Median (or mean for <3 frames), the per-pixel minimum, and the hit count."""
    nhits = np.isfinite(stack).sum(axis=0)
    comb = np.nanmedian(stack, axis=0) if len(stack) >= 3 else np.nanmean(stack, axis=0)
    low = np.nanmin(stack, axis=0)
    bad = nhits < min(min_hits, len(stack))
    comb[bad] = np.nan; low[bad] = np.nan
    return comb, low, nhits


def mpchecker(t: Time, center: SkyCoord, radius_arcmin, vmag_lim=20.0, grid=None):
    """Known asteroids near `center` at time `t` from the Minor Planet Center's MPChecker."""
    ra = center.ra.to_string(unit=u.hour, sep=" ", precision=1, pad=True)
    dec = center.dec.to_string(unit=u.deg, sep=" ", precision=0, alwayssign=True, pad=True)
    d = t.datetime
    day = d.day + (d.hour + d.minute / 60 + d.second / 3600) / 24
    r = requests.post("https://minorplanetcenter.net/cgi-bin/mpcheck.cgi", timeout=180, data=dict(
        year=d.year, month=f"{d.month:02d}", day=f"{day:.3f}", which="pos", ra=ra, decl=dec, TextArea="",
        radius=radius_arcmin, limit=vmag_lim, oc="500", sort="d", mot="h", tmot="s", pdes="u",
        needed="f", ps="n", type="p"))
    rows = []
    for line in re.sub(r"<[^>]+>", "", r.text).splitlines():
        m = re.match(r"^\s*(.{24}?)\s+(\d{2} \d{2} \d{2}\.\d)\s+([+-]\d{2} \d{2} \d{2})\s+(\d{1,2}\.\d)", line)
        if m:
            sc = SkyCoord(m.group(2), m.group(3), unit=(u.hourangle, u.deg))
            row = dict(name=m.group(1).strip(), ra=sc.ra.deg, dec=sc.dec.deg, V=float(m.group(4)))
            if grid is not None:
                px = grid.world_to_pixel(sc); row["x"], row["y"] = float(px[0]), float(px[1])
            rows.append(row)
    return rows


def local_noise(arr, block=29):
    """Robust per-block std (1.4826*MAD), smoothed back to full resolution; returns (map, global median)."""
    ny, nx = arr.shape
    by, bx = -(-ny // block), -(-nx // block)
    out = np.full((by, bx), np.nan)
    for j in range(by):
        for i in range(bx):
            blk = arr[j * block:(j + 1) * block, i * block:(i + 1) * block]
            blk = blk[np.isfinite(blk)]
            if blk.size > 20:
                out[j, i] = 1.4826 * np.median(np.abs(blk - np.median(blk)))
    g = np.nanmedian(out)
    out = np.where(np.isfinite(out), out, g)
    full = ndimage.zoom(out, (ny / by, nx / bx), order=1)[:ny, :nx]
    return np.maximum(full, 0.5 * g), g


def fit_scale(ref, img):
    """Linear brightness match img ~= a*ref + b on the brighter half of the overlapping pixels."""
    both = np.isfinite(ref) & np.isfinite(img)
    x, y = ref[both], img[both]
    sel = (x > np.nanpercentile(x, 50)) & (x < np.nanpercentile(x, 99.5))
    a, b = np.polyfit(x[sel], y[sel], 1)
    return a, b, both


def find_appeared(img, low, template, both, detect_sigma=5.0, grow_sigma=2.5, min_pix=2,
                  in_all_sigma=3.0, other_sigma=4.0, other_ratio=0.5, dilate=1, edge=1, nexp=1):
    """Lone positive sources in img that are absent from template.
    Returns (candidates, snr_map). Each candidate carries the test results."""
    diff = img - template; diff[~both] = np.nan
    sigmap, _ = local_noise(diff)
    snr = np.nan_to_num(diff / sigmap)
    _, _, sig_t = sigma_clipped_stats(template[both], sigma=3.0)
    _, _, sig_low = sigma_clipped_stats(low[both], sigma=3.0) if low is not None else (0, 0, 1.0)
    interior = ndimage.binary_erosion(both, iterations=edge)
    lab, n = ndimage.label((snr > grow_sigma) & interior)
    out = []
    for i in range(1, n + 1):
        sel = lab == i
        peak = float(np.max(snr[sel]))
        if sel.sum() < min_pix or peak < detect_sigma:
            continue
        cy, cx = ndimage.center_of_mass(np.clip(snr, 0, None) * sel)
        # star residual = the template already has a comparable source ON the blob (a star 2 px away is not a veto)
        region = ndimage.binary_dilation(sel, iterations=dilate)
        t_max = float(np.nanmax(template[region])); own_max = float(np.nanmax(img[sel]))
        star = (t_max > other_sigma * sig_t) and (t_max > other_ratio * own_max)
        if low is not None and nexp >= 2:
            low_max = float(np.nanmax(low[sel])); single = low_max < in_all_sigma * sig_low
        else:
            low_max, single = float("nan"), False
        out.append(dict(x=float(cx), y=float(cy), npix=int(sel.sum()), snr=peak,
                        star_residual=star, single_exposure=single, accepted=not star and not single,
                        template_sigma=t_max / sig_t, in_all_sigma=(low_max / sig_low) if nexp >= 2 else None))
    return out, snr
