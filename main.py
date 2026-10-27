"""
GhostShift stage-1 starter: find SPHEREx exposures covering a sky patch,
group them by epoch (survey sweep), and download small cutouts.

pip install astroquery astropy pandas
"""
from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import requests, pathlib

# --- 1. Pick a patch. Ecliptic field (lon 30, lat 0): visited 2025W30 and 2026W30 with
#        known asteroids inside: (3028) Zhangguoxi V=16.6 in 2025, (1621) Druzhba V=16.0 in 2026.
#        (Old test field M101 was RA 210.80, Dec 54.35 with a 0.2 deg cutout.)
import argparse
ap = argparse.ArgumentParser(description="GhostShift: download two-epoch SPHEREx cutouts of a field")
ap.add_argument("--ra", type=float, default=27.911, help="field centre RA (deg)")
ap.add_argument("--dec", type=float, default=11.472, help="field centre Dec (deg)")
ap.add_argument("--size", type=float, default=1.0, help="cutout box size (deg)")
ap.add_argument("--weeks", nargs=2, default=["2025W30", "2026W30"], metavar=("W1", "W2"),
                help="the two survey weeks to download (use 'auto auto' to pick automatically)")
ap.add_argument("--cutouts", default="cutouts", help="output folder for the FITS cutouts")
args = ap.parse_args()
TARGET = SkyCoord(args.ra, args.dec, unit="deg")
RADIUS = 0.02 * u.deg
CUTOUT_DEG = args.size                          # cutout box size in degrees (1.0 deg ~ 580 px)

# The two survey weeks (found with scan_fields.py); None = pick automatically.
EPOCH_WEEKS = None if args.weeks[0] == "auto" else tuple(args.weeks)

# --- 2. Ask IRSA which Spectral Images overlap the patch (QR2 + QR3 collections).
parts = []
for coll in ("spherex_qr2", "spherex_qr3"):
    for attempt in range(4):                    # IRSA occasionally returns 504
        try:
            t = Irsa.query_sia(pos=(TARGET, RADIUS), collection=coll); break
        except Exception as e:
            print("  retry", attempt + 1, coll, str(e)[:60])
    else:
        t = []
    if len(t):
        parts.append(t.to_pandas())
res = pd.concat(parts, ignore_index=True)
res = res.drop_duplicates(subset=["obs_id", "energy_bandpassname"], keep="last")  # QR3 reprocessing supersedes QR2
print(f"{len(res)} exposures overlap the patch")

# obs_id looks like 2025W19_2B_0346_4 -> first token is the survey week.
res["week"] = res["obs_id"].str.split("_").str[0]
res["detector"] = res["energy_bandpassname"]    # SPHEREx-D1 ... D6
print(res.groupby("week").size().sort_index())  # distinct weeks = distinct epochs

# --- 3. Pick two epochs and one detector (same wavelength range) for a first diff.
det = "SPHEREx-D2"                               # 1.10-1.65 micron
d2 = res[res["detector"].eq(det)]
counts = d2.groupby("week").size()
def week_to_day(w):                              # "2025W19" -> days since 2000 (approx)
    return (int(w[:4]) - 2000) * 365.25 + int(w[5:7]) * 7
if EPOCH_WEEKS:
    epoch1, epoch2 = EPOCH_WEEKS
else:
    epoch1 = counts.idxmax()                     # best-covered week
    far = [w for w in counts.index if abs(week_to_day(w) - week_to_day(epoch1)) > 150]
    if not far:
        raise SystemExit("no second sweep at least 5 months away from " + epoch1)
    epoch2 = max(far, key=counts.get)            # best-covered week in another sweep
pick = d2[d2["week"].isin([epoch1, epoch2])]
print(f"\nUsing {det}: {epoch1} -> {epoch2}, {len(pick)} files")

# --- 4. Download cutouts via IRSA's cutout service (append center & size).
out = pathlib.Path(args.cutouts); out.mkdir(exist_ok=True)
for _, row in pick.iterrows():
    url = f"{row.access_url}?center={TARGET.ra.deg},{TARGET.dec.deg}&size={CUTOUT_DEG}"
    dest = out / f"{row.week}_{pathlib.Path(row.access_url).name}"
    if dest.exists():
        continue
    for attempt in range(4):                    # IRSA sometimes drops connections
        try:
            r = requests.get(url, timeout=300); r.raise_for_status(); break
        except Exception as e:
            print("  retry", attempt + 1, str(e)[:60])
    else:
        continue
    dest.write_bytes(r.content)
    print("saved", dest.name, len(r.content) // 1024, "KB")

# Next: open each cutout (hdul['IMAGE'], hdul['WCS-WAVE']), reproject both epochs
# onto one grid with `reproject`, then subtract. See spherex_intro / spherex_cutouts
# tutorials at https://caltech-ipac.github.io/irsa-tutorials/
