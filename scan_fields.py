"""
Scan ecliptic positions for a SPHEREx field that (a) was imaged in two sweeps and
(b) had a known asteroid brighter than VMAG_LIM inside it at one of those exposure times.
Uses IRSA SIA for coverage and the Minor Planet Center's MPChecker for asteroids.
"""
import warnings, re, pathlib, requests
warnings.filterwarnings("ignore")
from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord, BarycentricMeanEcliptic
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs import WCS
from astropy.time import Time
import astropy.units as u, pandas as pd, io

DET = "SPHEREx-D2"
SIZE_DEG = 1.0          # cutout size used for the test download
RADIUS_ARCMIN = 35      # MPChecker search radius (covers the 1 deg box corners)
VMAG_LIM = 19.0         # ~J 17.8, near a single SPHEREx exposure's 5-sigma depth
LONS = [180, 210, 330, 30, 150, 120, 300]

def mpchecker(t: Time, c: SkyCoord):
    ra = c.ra.to_string(unit=u.hour, sep=" ", precision=1, pad=True)
    dec = c.dec.to_string(unit=u.deg, sep=" ", precision=0, alwayssign=True, pad=True)
    d = t.datetime
    day = d.day + (d.hour + d.minute / 60 + d.second / 3600) / 24
    r = requests.post("https://minorplanetcenter.net/cgi-bin/mpcheck.cgi", timeout=120, data=dict(
        year=d.year, month=f"{d.month:02d}", day=f"{day:.3f}", which="pos", ra=ra, decl=dec, TextArea="",
        radius=RADIUS_ARCMIN, limit=VMAG_LIM, oc="500", sort="d", mot="h", tmot="s", pdes="u",
        needed="f", ps="n", type="p"))
    txt = re.sub(r"<[^>]+>", "", r.text)
    rows = []
    for line in txt.splitlines():
        # e.g. "(12345) Name   14 03 12.3 +54 21 00  17.2  ..." -> capture name, ra, dec, V
        m = re.match(r"^\s*(.{24}?)\s+(\d{2} \d{2} \d{2}\.\d)\s+([+-]\d{2} \d{2} \d{2})\s+(\d{1,2}\.\d)", line)
        if m:
            rows.append(dict(name=m.group(1).strip(), ra=m.group(2), dec=m.group(3), V=float(m.group(4))))
    return rows

def first_cutout_time(access_url, c):
    url = f"{access_url}?center={c.ra.deg},{c.dec.deg}&size={SIZE_DEG}"
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=300); r.raise_for_status(); break
        except Exception as e:
            print(f"    download retry {attempt+1}: {str(e)[:50]}"); import time; time.sleep(5)
    with fits.open(io.BytesIO(r.content)) as h:
        hdr = h["IMAGE"].header
        return Time(hdr["DATE-OBS"]), hdr["NAXIS1"], hdr["NAXIS2"], WCS(hdr)

for lon in LONS:
    c = SkyCoord(lon=lon * u.deg, lat=0 * u.deg, frame=BarycentricMeanEcliptic).icrs
    parts = []
    for coll in ["spherex_qr2", "spherex_qr3"]:
        for attempt in range(4):                       # IRSA occasionally returns 504
            try:
                t = Irsa.query_sia(pos=(c, 0.02 * u.deg), collection=coll); break
            except Exception as e:
                print(f"  retry {attempt+1} {coll}: {str(e)[:60]}"); import time; time.sleep(5)
        else:
            t = []
        if len(t): parts.append(t.to_pandas())
    df = pd.concat(parts); df = df[df.energy_bandpassname.eq(DET)].copy()
    df["week"] = df["obs_id"].str.split("_").str[0]
    counts = df.groupby("week").size()
    weeks25 = [w for w in counts.index if w.startswith("2025") and counts[w] >= 3]
    weeks26 = [w for w in counts.index if w.startswith("2026") and counts[w] >= 3]
    if not weeks25 or not weeks26:
        print(f"lon {lon}: no two-epoch coverage with >=3 exposures"); continue
    w1, w2 = max(weeks25, key=counts.get), max(weeks26, key=counts.get)
    print(f"\n=== lon {lon}  RA {c.ra.deg:.3f} Dec {c.dec.deg:+.3f}   epochs {w1} ({counts[w1]} exp)  {w2} ({counts[w2]} exp)")
    for w in (w1, w2):
        row = df[df.week.eq(w)].iloc[0]
        t, nx, ny, wcs = first_cutout_time(row.access_url, c)
        ast = mpchecker(t, c)
        inside = []
        for a in ast:
            sc = SkyCoord(a["ra"], a["dec"], unit=(u.hourangle, u.deg))
            x, y = wcs.world_to_pixel(sc)
            a["in_cutout"] = bool(0 <= x < nx and 0 <= y < ny)
            if a["in_cutout"]: inside.append(a)
        print(f"  {w}  {t.iso[:16]}  cutout {nx}x{ny}px  asteroids V<{VMAG_LIM}: {len(ast)} in radius, {len(inside)} inside this exposure")
        for a in sorted(inside, key=lambda a: a["V"])[:8]:
            print(f"      {a['name']:<24} V={a['V']:4.1f}  {a['ra']}  {a['dec']}")
