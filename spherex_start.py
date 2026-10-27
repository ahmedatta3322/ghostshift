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

# --- 1. Pick a patch. M101 is a good test: covered in May 2025 and Dec 2025.
TARGET = SkyCoord(210.80, 54.35, unit="deg")   # RA, Dec in degrees
RADIUS = 0.02 * u.deg
CUTOUT_DEG = 0.2                                # cutout box size in degrees

# --- 2. Ask IRSA which Spectral Images overlap the patch (QR2 collection).
res = Irsa.query_sia(pos=(TARGET, RADIUS), collection="spherex_qr2").to_table().to_pandas()
print(f"{len(res)} exposures overlap the patch")

# obs_id looks like 2025W19_2B_0346_4 -> first token is the survey week.
res["week"] = res["obs_id"].str.split("_").str[0]
res["detector"] = res["energy_bandpassname"]    # SPHEREx-D1 ... D6
print(res.groupby("week").size().sort_index())  # distinct weeks = distinct epochs

# --- 3. Pick two epochs and one detector (same wavelength range) for a first diff.
weeks = sorted(res["week"].unique())
epoch1, epoch2 = weeks[0], weeks[-1]
det = "SPHEREx-D2"                               # 1.10-1.65 micron
pick = res[res["detector"].eq(det) & res["week"].isin([epoch1, epoch2])]
print(f"\nUsing {det}: {epoch1} -> {epoch2}, {len(pick)} files")

# --- 4. Download cutouts via IRSA's cutout service (append center & size).
out = pathlib.Path("cutouts"); out.mkdir(exist_ok=True)
for _, row in pick.iterrows():
    url = f"{row.access_url}?center={TARGET.ra.deg},{TARGET.dec.deg}&size={CUTOUT_DEG}"
    dest = out / f"{row.week}_{pathlib.Path(row.access_url).name}"
    if dest.exists():
        continue
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print("saved", dest.name, len(r.content) // 1024, "KB")

# Next: open each cutout (hdul['IMAGE'], hdul['WCS-WAVE']), reproject both epochs
# onto one grid with `reproject`, then subtract. See spherex_intro / spherex_cutouts
# tutorials at https://caltech-ipac.github.io/irsa-tutorials/
