"""
GhostShift stage 2: align two SPHEREx epochs onto one grid, subtract, find movers,
and check each mover against the Minor Planet Center's known-object list.

Input : cutouts/*.fits produced by main.py (filenames start with the survey week)
Output: out/epoch1.npy, out/epoch2.npy, out/diff.npy      aligned arrays
        out/epoch1.png, out/epoch2.png, out/diff.png       quick-look images
        out/summary.png, out/blink.gif                     side-by-side + blink
        out/candidates.json                                movers with known/unknown label
        out/grid.fits                                      WCS of the common grid

pip install astropy reproject numpy scipy matplotlib requests
"""
import glob, json, pathlib, re, warnings
import numpy as np, requests
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
import astropy.units as u
from reproject import reproject_interp
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

warnings.filterwarnings("ignore")

# ---------------- settings ----------------
import argparse
_ap = argparse.ArgumentParser(description="GhostShift: align, subtract, find and identify movers")
_ap.add_argument("--ra", type=float, default=27.911, help="field centre RA (deg), same as main.py")
_ap.add_argument("--dec", type=float, default=11.472, help="field centre Dec (deg), same as main.py")
_ap.add_argument("--size", type=float, default=1.0, help="field size (deg), same as main.py")
_ap.add_argument("--cutouts", default="cutouts", help="folder with the FITS cutouts from main.py")
_ap.add_argument("--out", default="out", help="output folder")
_args = _ap.parse_args()
CENTER_RA, CENTER_DEC = _args.ra, _args.dec
CUTOUT_DEG = _args.size
PIX_ARCSEC = 6.2                         # SPHEREx native pixel scale
NPIX = int(CUTOUT_DEG * 3600 / PIX_ARCSEC)
DETECT_SIGMA = 5.0                       # threshold on the difference image (local noise)
MIN_BLOB_PIX = 2                         # smallest blob we believe (connected pixels above GROW_SIGMA)
GROW_SIGMA = 2.5                         # blob = connected pixels above this, containing a peak above DETECT_SIGMA
MIN_HITS = 2                             # a pixel needs this many exposures in a visit to be trusted (2 is safe: the every-exposure test below rejects cosmic rays)
IN_ALL_SIGMA = 3.0                       # a real source is present in EVERY exposure: min over the stack > this x noise
OTHER_EPOCH_SIGMA = 4.0                  # a real mover leaves NOTHING (< this many sigma) in the other epoch
OTHER_EPOCH_DILATE = 2                   # ... checked within this many pixels of the blob
OTHER_EPOCH_RATIO = 0.2                  # ... and at least this fraction of the blob's own flux
EDGE_PIX = 1                             # ignore blobs this close to the footprint edge
CLUSTER_DAYS = 0.1                       # exposures closer than this are "the same visit"
MATCH_ARCSEC = 30.0                      # candidate within this of a predicted asteroid -> "known"
VMAG_LIM = 20.0                          # MPChecker magnitude limit
NOISE_BLOCK = 29                         # block size (pixels) for the local noise map
OUT = pathlib.Path(_args.out); OUT.mkdir(exist_ok=True)
CUTOUT_DIR = pathlib.Path(_args.cutouts)

# FLAGS bits that mean "do not trust this pixel" (MP_* keys in the FLAGS header). Not 12/14/21, and not 19
# (OUTLIER): SPHEREx flags moving objects as outliers against other visits, so masking it erases asteroids.
BAD_MASK = sum(1 << b for b in [0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 15, 17])

# ---------------- 1. common grid ----------------
grid = WCS(naxis=2)
grid.wcs.ctype = ["RA---TAN", "DEC--TAN"]
grid.wcs.crval = [CENTER_RA, CENTER_DEC]
grid.wcs.crpix = [NPIX / 2 + 0.5, NPIX / 2 + 0.5]
grid.wcs.cdelt = [-PIX_ARCSEC / 3600, PIX_ARCSEC / 3600]
shape = (NPIX, NPIX)
fits.PrimaryHDU(header=grid.to_header()).writeto(OUT / "grid.fits", overwrite=True)
center = SkyCoord(CENTER_RA, CENTER_DEC, unit="deg")
SEARCH_RADIUS_ARCMIN = CUTOUT_DEG * 60 * 0.75

# ---------------- known-object lookup (Minor Planet Center MPChecker) ----------------
def mpchecker(t: Time, c: SkyCoord, radius_arcmin):
    ra = c.ra.to_string(unit=u.hour, sep=" ", precision=1, pad=True)
    dec = c.dec.to_string(unit=u.deg, sep=" ", precision=0, alwayssign=True, pad=True)
    d = t.datetime
    day = d.day + (d.hour + d.minute / 60 + d.second / 3600) / 24
    r = requests.post("https://minorplanetcenter.net/cgi-bin/mpcheck.cgi", timeout=180, data=dict(
        year=d.year, month=f"{d.month:02d}", day=f"{day:.3f}", which="pos", ra=ra, decl=dec, TextArea="",
        radius=radius_arcmin, limit=VMAG_LIM, oc="500", sort="d", mot="h", tmot="s", pdes="u",
        needed="f", ps="n", type="p"))
    rows = []
    for line in re.sub(r"<[^>]+>", "", r.text).splitlines():
        m = re.match(r"^\s*(.{24}?)\s+(\d{2} \d{2} \d{2}\.\d)\s+([+-]\d{2} \d{2} \d{2})\s+(\d{1,2}\.\d)", line)
        if m:
            sc = SkyCoord(m.group(2), m.group(3), unit=(u.hourangle, u.deg))
            px = grid.world_to_pixel(sc)
            rows.append(dict(name=m.group(1).strip(), ra=sc.ra.deg, dec=sc.dec.deg, V=float(m.group(4)),
                             x=float(px[0]), y=float(px[1])))
    return rows

def inside(valid, x, y):
    xi, yi = int(round(x)), int(round(y))
    return 0 <= xi < NPIX and 0 <= yi < NPIX and bool(valid[yi, xi])

# ---------------- 2. inventory: group files into visits ----------------
def header_time(path):
    with fits.open(path) as h:
        return Time(h["IMAGE"].header["DATE-OBS"])

files = sorted(glob.glob(str(CUTOUT_DIR / "*.fits")))
if not files:
    raise SystemExit("no cutouts found; run main.py first")
times = {f: header_time(f) for f in files}

def cluster_visits(fs):
    """Split a week's files into visits whose exposures are within CLUSTER_DAYS of each other."""
    fs = sorted(fs, key=lambda f: times[f].mjd)
    visits, cur = [], [fs[0]]
    for f in fs[1:]:
        if times[f].mjd - times[cur[-1]].mjd <= CLUSTER_DAYS:
            cur.append(f)
        else:
            visits.append(cur); cur = [f]
    visits.append(cur)
    return visits

weeks = sorted({pathlib.Path(f).name.split("_")[0] for f in files})
if len(weeks) < 2:
    raise SystemExit("need cutouts from at least two survey weeks")
w1, w2 = weeks[0], weeks[-1]

# ---------------- 3. load, clean, reproject, combine every visit ----------------
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

def build_visit(fs):
    stack = []
    for f in fs:
        img, wcs = load_clean(f)
        arr, _ = reproject_interp((img, wcs), grid, shape_out=shape, order="bilinear")
        stack.append(arr)
    stack = np.array(stack)
    nhits = np.isfinite(stack).sum(axis=0)
    comb = np.nanmedian(stack, axis=0) if len(fs) >= 3 else np.nanmean(stack, axis=0)
    low = np.nanmin(stack, axis=0)                      # faintest the spot ever was in this visit
    bad = nhits < min(MIN_HITS, len(fs))                # drop pixels seen by too few exposures
    comb[bad] = np.nan; low[bad] = np.nan
    t = Time(np.mean([times[f].mjd for f in fs]), format="mjd")
    return dict(files=fs, img=comb, low=low, nhits=nhits, time=t, valid=np.isfinite(comb))

visits = {}
for w in (w1, w2):
    visits[w] = []
    for fs in cluster_visits([f for f in files if pathlib.Path(f).name.startswith(w)]):
        v = build_visit(fs)
        try:
            v["asteroids"] = mpchecker(v["time"], center, SEARCH_RADIUS_ARCMIN)
        except Exception as e:
            print(f"MPChecker failed for {v['time'].iso[:16]}: {e}"); v["asteroids"] = []
        print(f"{w}  visit {v['time'].iso[:16]}  {len(fs)} exposures  valid={v['valid'].mean():.0%}  "
              f"{len(v['asteroids'])} known asteroids within {SEARCH_RADIUS_ARCMIN:.0f} arcmin")
        visits[w].append(v)

# ---------------- 4. choose the visit pair whose shared footprint holds the brightest asteroid ----------------
def pair_score(v1, v2):
    both_ = v1["valid"] & v2["valid"]
    ast = [a for a in v1["asteroids"] + v2["asteroids"] if inside(both_, a["x"], a["y"])]
    brightest = min([a["V"] for a in ast], default=99.0)
    return (brightest, -len([a for a in ast if a["V"] <= 19.0]), -both_.mean()), ast

def good_visits(vs):
    multi = [v for v in vs if len(v["files"]) >= 3]
    return multi if multi else vs

best = None
for v1 in good_visits(visits[w1]):
    for v2 in good_visits(visits[w2]):
        score, ast = pair_score(v1, v2)
        if best is None or score < best[0]:
            best = (score, v1, v2, ast)
_, V1, V2, shared_ast = best
e1, e2, t1, t2 = V1["img"], V2["img"], V1["time"], V2["time"]
low1, low2 = V1["low"], V2["low"]
print(f"chosen: epoch 1 = {t1.iso[:16]} ({len(V1['files'])} exp), epoch 2 = {t2.iso[:16]} ({len(V2['files'])} exp); "
      f"{len(shared_ast)} known asteroids in the shared footprint, brightest V={best[0][0]}")
known = {"epoch1": V1["asteroids"], "epoch2": V2["asteroids"]}

# ---------------- 5. scale-match and subtract ----------------
both = np.isfinite(e1) & np.isfinite(e2)
x, y = e1[both], e2[both]
sel = (x > np.nanpercentile(x, 50)) & (x < np.nanpercentile(x, 99.5))
a, b = np.polyfit(x[sel], y[sel], 1)
e1s = a * e1 + b                         # epoch 1 on epoch 2's brightness scale
low1s = a * low1 + b
diff = e2 - e1s; diff[~both] = np.nan
_, _, sig_e1s = sigma_clipped_stats(e1s[both], sigma=3.0)
_, _, sig_e2 = sigma_clipped_stats(e2[both], sigma=3.0)
_, _, sig_low1s = sigma_clipped_stats(low1s[both], sigma=3.0)
_, _, sig_low2 = sigma_clipped_stats(low2[both], sigma=3.0)

def local_noise(arr, block):
    """Robust per-block std (1.4826*MAD), smoothed back to full resolution."""
    ny, nx = arr.shape
    by, bx = -(-ny // block), -(-nx // block)
    out = np.full((by, bx), np.nan)
    for j in range(by):
        for i in range(bx):
            blk = arr[j * block:(j + 1) * block, i * block:(i + 1) * block]
            blk = blk[np.isfinite(blk)]
            if blk.size > 20:
                out[j, i] = 1.4826 * np.median(np.abs(blk - np.median(blk)))
    glob_ = np.nanmedian(out)
    out = np.where(np.isfinite(out), out, glob_)
    full = ndimage.zoom(out, (ny / by, nx / bx), order=1)[:ny, :nx]
    return np.maximum(full, 0.5 * glob_), glob_

sigmap, sig = local_noise(diff, NOISE_BLOCK)
print(f"scale epoch1->epoch2 gain={a:.3f} offset={b:.4f}; diff noise sigma={sig:.4f} MJy/sr (global median), "
      f"local range {np.nanmin(sigmap):.4f}-{np.nanmax(sigmap):.4f}; overlap={both.mean():.0%}")
np.save(OUT / "epoch1.npy", e1); np.save(OUT / "epoch2.npy", e2); np.save(OUT / "diff.npy", diff)

# ---------------- 6. candidates ----------------
# A mover between epochs months apart has travelled far outside the field, so it shows as
# a lone blob: "appeared" (positive, only in epoch 2) or "vanished" (negative, only in epoch 1).
# A star residual also makes a blob, but the star is present in BOTH epochs, so we reject blobs
# where the other epoch has anything detectable (> OTHER_EPOCH_SIGMA x its noise) nearby.
snr_map = np.nan_to_num(diff / sigmap)
interior = ndimage.binary_erosion(both, iterations=EDGE_PIX)

def blobs(mask, sign):
    lab, n = ndimage.label(mask & interior)
    other, sig_other = (e2, sig_e2) if sign < 0 else (e1s, sig_e1s)   # epoch where object should be ABSENT
    own = e1s if sign < 0 else e2                                       # epoch where the object IS
    own_low, sig_low = (low1s, sig_low1s) if sign < 0 else (low2, sig_low2)
    out = []
    for i in range(1, n + 1):
        sel_i = lab == i
        snr = float(np.max(sign * snr_map[sel_i]))
        if sel_i.sum() < MIN_BLOB_PIX or snr < DETECT_SIGMA:
            continue
        cy, cx = ndimage.center_of_mass(np.clip(sign * snr_map, 0, None) * sel_i)
        region = ndimage.binary_dilation(sel_i, iterations=OTHER_EPOCH_DILATE)
        other_max = float(np.nanmax(other[region]))
        own_max = float(np.nanmax(own[region]))
        low_max = float(np.nanmax(own_low[sel_i]))        # present in the faintest exposure too?
        out.append(dict(x=float(cx), y=float(cy), npix=int(sel_i.sum()), snr=snr,
                        other_epoch_sigma=other_max / sig_other, flux_ratio=other_max / max(own_max, 1e-9),
                        in_all_sigma=low_max / sig_low,
                        single_exposure=low_max < IN_ALL_SIGMA * sig_low,
                        star_residual=(other_max > OTHER_EPOCH_SIGMA * sig_other) and (other_max > OTHER_EPOCH_RATIO * own_max)))
    return out

pos = blobs(snr_map > +GROW_SIGMA, +1)
neg = blobs(snr_map < -GROW_SIGMA, -1)
def accepted(b):
    return not b["star_residual"] and not b["single_exposure"]
print(f"blobs: {len(pos)} appeared, {len(neg)} vanished; star residuals rejected: "
      f"{sum(p['star_residual'] for p in pos)} + {sum(n['star_residual'] for n in neg)}; "
      f"single-exposure glitches rejected: {sum(p['single_exposure'] and not p['star_residual'] for p in pos)} + "
      f"{sum(n['single_exposure'] and not n['star_residual'] for n in neg)}; "
      f"accepted: {sum(accepted(p) for p in pos)} + {sum(accepted(n) for n in neg)}")

# ---------------- 7. label candidates known / unidentified ----------------
cands = []
for kind, lst, t, label in (("appeared", pos, t2, "epoch2"), ("vanished", neg, t1, "epoch1")):
    for bl in lst:
        if not accepted(bl):
            continue
        sc = grid.pixel_to_world(bl["x"], bl["y"])
        match, bestsep = None, MATCH_ARCSEC
        for k in known[label]:
            sep = sc.separation(SkyCoord(k["ra"], k["dec"], unit="deg")).arcsec
            if sep < bestsep:
                match, bestsep = k, sep
        cands.append(dict(kind=kind, epoch=label, date=t.iso[:16], ra=round(sc.ra.deg, 5), dec=round(sc.dec.deg, 5),
                          x=round(bl["x"], 1), y=round(bl["y"], 1), snr=round(bl["snr"], 1), npix=bl["npix"],
                          other_epoch_sigma=round(bl["other_epoch_sigma"], 1), in_all_sigma=round(bl["in_all_sigma"], 1),
                          known=(dict(name=match["name"], V=match["V"], sep_arcsec=round(bestsep, 1)) if match else None)))
cands.sort(key=lambda c: -c["snr"])
n_known = sum(c["known"] is not None for c in cands)
print(f"{len(cands)} mover candidates: {n_known} match a known asteroid, {len(cands) - n_known} unidentified")
for c in cands[:20]:
    if c["known"]:
        tag = f"KNOWN  {c['known']['name']} V={c['known']['V']} (sep {c['known']['sep_arcsec']} arcsec)"
    else:
        tag = "unidentified"
    print(f"  {c['kind']:<8} snr={c['snr']:5.1f} npix={c['npix']:3d}  RA={c['ra']:.4f} Dec={c['dec']:+.4f}  {tag}")

expected = []
for label in ("epoch1", "epoch2"):
    for k in known[label]:
        if not inside(both, k["x"], k["y"]):
            continue
        hit = any(c["known"] and c["known"]["name"] == k["name"] and c["epoch"] == label for c in cands)
        expected.append(dict(epoch=label, name=k["name"], V=k["V"], ra=k["ra"], dec=k["dec"],
                             x=round(k["x"], 1), y=round(k["y"], 1), detected=hit))
def why_missed(e):
    """Nearest blob (of the right sign, incl. rejected ones) within MATCH_ARCSEC and its status."""
    lst = neg if e["epoch"] == "epoch1" else pos
    near = [(np.hypot(b["x"] - e["x"], b["y"] - e["y"]) * PIX_ARCSEC, b) for b in lst]
    near = [(d, b) for d, b in near if d <= MATCH_ARCSEC]
    if not near:
        yi, xi = int(round(e["y"])), int(round(e["x"]))
        peak = np.nanmax(np.abs(snr_map[max(yi-3,0):yi+4, max(xi-3,0):xi+4]))
        return f"no blob within {MATCH_ARCSEC:.0f} arcsec (peak |snr| nearby {peak:.1f}; edge={not interior[yi, xi]})"
    d, b = min(near, key=lambda t: t[0])
    if b["star_residual"]:
        return (f"blob at {d:.0f} arcsec rejected as star residual (other epoch {b['other_epoch_sigma']:.1f} sigma, "
                f"flux ratio {b['flux_ratio']:.2f})")
    if b["single_exposure"]:
        return f"blob at {d:.0f} arcsec rejected: not in every exposure (min-stack {b['in_all_sigma']:.1f} sigma)"
    return f"blob at {d:.0f} arcsec accepted"

print("known asteroids inside the shared footprint:")
for e in sorted(expected, key=lambda e: e["V"]):
    xi, yi = int(round(e["x"])), int(round(e["y"]))
    status = "DETECTED" if e["detected"] else "missed: " + why_missed(e)
    print(f"  {e['epoch']}  {e['name']:<24} V={e['V']:4.1f}  diff snr at prediction {snr_map[yi, xi]:+5.1f}  {status}")

json.dump(dict(epoch1=dict(week=w1, time=t1.iso, files=[pathlib.Path(f).name for f in V1["files"]]),
               epoch2=dict(week=w2, time=t2.iso, files=[pathlib.Path(f).name for f in V2["files"]]),
               candidates=cands, expected_known=expected), open(OUT / "candidates.json", "w"), indent=2)

# ---------------- 8. images ----------------
lo, hi = np.nanpercentile(np.concatenate([e1[both], e2[both]]), [1, 99.5])

def save_png(arr, name, cmap="gray", vmin=lo, vmax=hi):
    plt.imsave(OUT / name, np.nan_to_num(arr, nan=vmin), cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")

save_png(e1, "epoch1.png"); save_png(e2, "epoch2.png")
save_png(diff, "diff.png", cmap="RdBu_r", vmin=-8 * sig, vmax=8 * sig)

def mark(ax):
    for c in cands:
        col = "lime" if c["known"] else "red"
        ax.plot(c["x"], c["y"], "o", mfc="none", mec=col, ms=12, mew=1.2)
        if c["known"]:
            ax.annotate(c["known"]["name"].split(")")[-1].strip(), (c["x"], c["y"]), xytext=(8, 8),
                        textcoords="offset points", color=col, fontsize=7)
    for e in expected:
        if not e["detected"]:
            ax.plot(e["x"], e["y"], "s", mfc="none", mec="yellow", ms=12, mew=1.0)

fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
panels = [(axs[0], e1, f"epoch 1  {t1.iso[:10]}", "gray", (lo, hi)),
          (axs[1], e2, f"epoch 2  {t2.iso[:10]}", "gray", (lo, hi)),
          (axs[2], diff, "epoch 2 - epoch 1   (red = appeared, blue = vanished)", "RdBu_r", (-8 * sig, 8 * sig))]
for ax_, arr, t_, cm, v in panels:
    ax_.imshow(np.nan_to_num(arr, nan=v[0]), cmap=cm, vmin=v[0], vmax=v[1], origin="lower")
    ax_.set_title(t_); ax_.axis("off")
mark(axs[2])
fig.suptitle("green = matches a known asteroid, red = unidentified mover, yellow square = predicted asteroid not detected", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "summary.png", dpi=110)

fig, ax = plt.subplots(figsize=(6, 6)); ax.axis("off")
im = ax.imshow(np.nan_to_num(e1, nan=lo), cmap="gray", vmin=lo, vmax=hi, origin="lower")
title = ax.set_title(f"{w1}  {t1.iso[:10]}"); mark(ax)

def frame(i):
    im.set_data(np.nan_to_num(e1 if i == 0 else e2, nan=lo))
    title.set_text(f"{w1}  {t1.iso[:10]}" if i == 0 else f"{w2}  {t2.iso[:10]}")

FuncAnimation(fig, frame, frames=2, interval=600).save(OUT / "blink.gif", writer=PillowWriter(fps=1.6))
print("wrote", sorted(p.name for p in OUT.iterdir()))
