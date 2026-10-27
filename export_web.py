"""
GhostShift stage 3: turn the aligned arrays + candidate list of one or more fields into
browser-ready layers (PNG) and a manifest (web/data/fields.js) for the viewer in web/index.html.

    python export_web.py out_ast:"Asteroid field (ecliptic)" out:"Ecliptic field" out_m101:"M101 test"

Each argument is  <diff.py output folder>[:<display name>].
Images are exported at native resolution (6.2 arcsec/pixel). A 1-degree field is only ~580 px,
so one PNG per layer is faster than a tile pyramid; cut tiles here if fields ever exceed ~4000 px.
"""
import sys, json, pathlib, re
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

WEB = pathlib.Path("web"); DATA = WEB / "data"; DATA.mkdir(parents=True, exist_ok=True)

def asinh_stretch(arr, lo_pct=5, hi_pct=99.7, soft=0.02):
    """Map a sky image to 0..255 with an asinh stretch: faint things visible, bright stars not clipped."""
    finite = np.isfinite(arr)
    lo, hi = np.nanpercentile(arr[finite], [lo_pct, hi_pct])
    x = (arr - lo) / max(hi - lo, 1e-9)
    y = np.arcsinh(x / soft) / np.arcsinh(1 / soft)
    out = np.clip(np.nan_to_num(y, nan=0.0), 0, 1) * 255
    return out.astype(np.uint8), finite

def to_rgba_gray(arr):
    g, finite = asinh_stretch(arr)
    rgba = np.dstack([g, g, g, np.where(finite, 255, 0).astype(np.uint8)])
    return rgba

def to_rgba_diverging(diff, nsig=6, deadband=1.5):
    """Difference image: blue = brighter before (vanished), red = brighter now (appeared), white = no change.
    Linear colour scale saturating at +-nsig sigma; changes below `deadband` sigma stay white so noise is quiet."""
    finite = np.isfinite(diff)
    med = np.nanmedian(diff); mad = 1.4826 * np.nanmedian(np.abs(diff - med))
    z = np.nan_to_num((diff - med) / mad, nan=0.0)
    z = np.sign(z) * np.maximum(np.abs(z) - deadband, 0)
    x = np.clip(z / (nsig - deadband), -1, 1)
    r = np.where(x > 0, 255, 255 * (1 + x)); b = np.where(x < 0, 255, 255 * (1 - x)); g = 255 * (1 - np.abs(x))
    rgba = np.dstack([r, g, b, np.where(finite, 255, 0)]).astype(np.uint8)
    return rgba

def save(rgba, path):
    # FITS/numpy row 0 is the bottom of the sky image; PNG row 0 is the top -> flip
    Image.fromarray(rgba[::-1]).save(path, optimize=True)

def export(folder, name):
    folder = pathlib.Path(folder)
    key = re.sub(r"[^a-z0-9]+", "_", folder.name.lower())
    out = DATA / key; out.mkdir(exist_ok=True)
    hdr = fits.getheader(folder / "grid.fits")
    layers = {}
    if (folder / "epoch1.npy").exists():
        e1 = np.load(folder / "epoch1.npy"); e2 = np.load(folder / "epoch2.npy"); d = np.load(folder / "diff.npy")
        save(to_rgba_gray(e1), out / "epoch1.png")
        save(to_rgba_gray(e2), out / "epoch2.png")
        save(to_rgba_diverging(d), out / "diff.png")
        ny, nx = e1.shape
        layers = dict(epoch1=f"data/{key}/epoch1.png", epoch2=f"data/{key}/epoch2.png", diff=f"data/{key}/diff.png")
    else:   # track-only folder: take the size from the first frame
        tr0 = json.load(open(folder / "track.json"))["frames"][0]
        nx, ny = Image.open(folder / tr0["image"]).size
    cands = {}
    cj = folder / "candidates.json"
    if cj.exists():
        cands = json.load(open(cj))
    def ep(label):
        v = cands.get(label, {})
        return v if isinstance(v, dict) else {"week": v, "time": ""}
    field = dict(
        key=key, name=name, width=nx, height=ny, pixel_arcsec=abs(hdr["CDELT2"]) * 3600,
        # TAN (gnomonic) projection parameters, 0-based pixel coordinates in FITS orientation (row 0 = bottom)
        wcs=dict(crval=[hdr["CRVAL1"], hdr["CRVAL2"]], crpix=[hdr["CRPIX1"] - 1, hdr["CRPIX2"] - 1],
                 cdelt=[hdr["CDELT1"], hdr["CDELT2"]]),
        epoch1=ep("epoch1"), epoch2=ep("epoch2"),
        candidates=[c for c in cands.get("candidates", []) if "x" in c],
        expected=cands.get("expected_known", []),
        layers=layers,
    )
    # optional motion track from track.py
    tj = folder / "track.json"
    if tj.exists():
        tr = json.load(open(tj))
        for fr in tr["frames"]:
            src = folder / fr["image"]; dst = out / f"track_{fr['index']}.png"
            dst.write_bytes(src.read_bytes()); fr["image"] = f"data/{key}/{dst.name}"
        field["track"] = tr
        det = sum(1 for o in tr["observations"] if o["detected"])
        print(f"  track: {tr['object']} over {len(tr['frames'])} visits, detected in {det}")
    print(f"{name}: {nx}x{ny} px, {len(field['candidates'])} candidates, {len(field['expected'])} predicted asteroids -> web/data/{key}/")
    return field

if __name__ == "__main__":
    args = sys.argv[1:] or ["out_ast:Asteroid field (ecliptic)", "out:Ecliptic field", "out_m101:M101 test"]
    fields = []
    for a in args:
        folder, _, name = a.partition(":")
        if not pathlib.Path(folder, "grid.fits").exists():
            print(f"skip {folder}: no grid.fits (run diff.py or track.py first)"); continue
        fields.append(export(folder, name or folder))
    (DATA / "fields.js").write_text("window.GHOSTSHIFT_FIELDS = " + json.dumps(fields, indent=1) + ";\n")
    print("wrote web/data/fields.js  ->  open web/index.html (or: python -m http.server -d web 8765)")
