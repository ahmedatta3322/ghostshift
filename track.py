"""
GhostShift motion track: one aligned image per visit within a survey week, each compared against a
static template built from the OTHER week, so an asteroid shows up at a new place in every frame.

    python track.py --ra 28.17 --dec 11.44 --cutouts cutouts_ast --out out_ast --week 2026W30 --template-week 2025W30

Output (in --out):
  track_frames/frame_<i>.png    quick-look image of each visit
  track.json                    frames (time, exposures, detections, predicted asteroids), the chosen
                                object's predicted path and where we actually detected it
  track_summary.png             all frames side by side with the track drawn on
"""
import argparse, json, pathlib
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import (make_grid, header_time, list_files, cluster_visits, reproject_stack, combine,
                    mpchecker, fit_scale, find_appeared, PIX_ARCSEC)

ap = argparse.ArgumentParser()
ap.add_argument("--ra", type=float, default=28.17); ap.add_argument("--dec", type=float, default=11.44)
ap.add_argument("--size", type=float, default=1.0)
ap.add_argument("--cutouts", default="cutouts_ast"); ap.add_argument("--out", default="out_ast")
ap.add_argument("--week", default="2026W30", help="week whose visits become the movie frames")
ap.add_argument("--template-week", default="2025W30", help="week combined into the static-sky template")
ap.add_argument("--object", default=None, help="name substring of the asteroid to track (default: brightest detected)")
ap.add_argument("--match-arcsec", type=float, default=30.0)
ap.add_argument("--vmag", type=float, default=20.0)
args = ap.parse_args()

OUT = pathlib.Path(args.out); FR = OUT / "track_frames"; FR.mkdir(parents=True, exist_ok=True)
grid, shape = make_grid(args.ra, args.dec, args.size)
from astropy.io import fits
fits.PrimaryHDU(header=grid.to_header()).writeto(OUT / "grid.fits", overwrite=True)
NPIX = shape[0]
center = SkyCoord(args.ra, args.dec, unit="deg")
radius = args.size * 60 * 0.75

def in_grid(x, y, valid=None):
    xi, yi = int(round(x)), int(round(y))
    ok = 0 <= xi < NPIX and 0 <= yi < NPIX
    return ok and (valid is None or bool(valid[yi, xi]))

# ---------------- template: every exposure of the other week, median-combined ----------------
tfiles = list_files(args.cutouts, args.template_week)
if not tfiles:
    raise SystemExit(f"no cutouts for template week {args.template_week}")
print(f"template: {len(tfiles)} exposures from {args.template_week}")
template, _, t_hits = combine(reproject_stack(tfiles, grid, shape), min_hits=2)
print(f"  template covers {np.isfinite(template).mean():.0%} of the grid")

# ---------------- frames: one per visit of the target week ----------------
files = list_files(args.cutouts, args.week)
if not files:
    raise SystemExit(f"no cutouts for week {args.week}")
times = {f: header_time(f) for f in files}
visits = cluster_visits(files, times)
frames = []
for i, v in enumerate(visits):
    t = Time(np.mean([times[f].mjd for f in v]), format="mjd")
    stack = reproject_stack(v, grid, shape)
    img, low, nhits = combine(stack, min_hits=2)
    a, b, both = fit_scale(template, img)
    cands, snr = find_appeared(img, low, a * template + b, both, nexp=len(v))
    try:
        known = mpchecker(t, center, radius, args.vmag, grid)
    except Exception as e:
        print(f"  MPChecker failed: {e}"); known = []
    known = [k for k in known if in_grid(k["x"], k["y"], both)]
    # match accepted candidates to predictions
    for c in cands:
        c["known"] = None
        if not c["accepted"]:
            continue
        sc = grid.pixel_to_world(c["x"], c["y"]); c["ra"], c["dec"] = float(sc.ra.deg), float(sc.dec.deg)
        best, bsep = None, args.match_arcsec
        for k in known:
            sep = sc.separation(SkyCoord(k["ra"], k["dec"], unit="deg")).arcsec
            if sep < bsep: best, bsep = k, sep
        if best:
            c["known"] = dict(name=best["name"], V=best["V"], sep_arcsec=round(bsep, 1))
    for k in known:
        hit = [c for c in cands if c["known"] and c["known"]["name"] == k["name"]]
        k["detected"] = bool(hit)
        if hit:
            k["det_x"], k["det_y"], k["det_snr"] = hit[0]["x"], hit[0]["y"], hit[0]["snr"]
    acc = [c for c in cands if c["accepted"]]
    print(f"frame {i}: {t.iso[:16]}  {len(v)} exp  valid={np.isfinite(img).mean():.0%}  "
          f"{len(acc)} accepted movers ({sum(c['known'] is not None for c in acc)} known)  "
          f"predicted in field: {', '.join(f'{k['name']} V={k['V']}' + (' DET' if k['detected'] else '') for k in sorted(known, key=lambda k: k['V'])[:6])}")
    # quick-look PNG (same stretch as export_web)
    fin = np.isfinite(img); lo, hi = np.nanpercentile(img[fin], [5, 99.7])
    g = np.arcsinh(((img - lo) / max(hi - lo, 1e-9)) / 0.02) / np.arcsinh(50)
    g = (np.clip(np.nan_to_num(g, nan=0), 0, 1) * 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(np.dstack([g, g, g, np.where(fin, 255, 0).astype(np.uint8)])[::-1]).save(FR / f"frame_{i}.png", optimize=True)
    frames.append(dict(index=i, time=t.iso, mjd=float(t.mjd), n_exposures=len(v),
                       files=[pathlib.Path(f).name for f in v], valid_fraction=float(fin.mean()),
                       image=f"track_frames/frame_{i}.png",
                       candidates=[dict(x=round(c["x"], 1), y=round(c["y"], 1), snr=round(c["snr"], 1), npix=c["npix"],
                                        ra=round(c.get("ra", 0), 5), dec=round(c.get("dec", 0), 5), known=c["known"]) for c in acc],
                       predicted=[dict(name=k["name"], V=k["V"], ra=k["ra"], dec=k["dec"], x=round(k["x"], 1), y=round(k["y"], 1),
                                       detected=k["detected"], det_x=k.get("det_x"), det_y=k.get("det_y"), det_snr=k.get("det_snr")) for k in known]))

# ---------------- choose the object to track ----------------
dets = {}
for fr in frames:
    for k in fr["predicted"]:
        if k["detected"]:
            dets.setdefault(k["name"], []).append((fr["index"], k))
if args.object:
    name = next((n for n in {k["name"] for fr in frames for k in fr["predicted"]} if args.object.lower() in n.lower()), None)
    if name is None:
        raise SystemExit(f"object matching '{args.object}' was never predicted inside the field")
else:
    if not dets:
        raise SystemExit("no known asteroid was detected in any frame")
    name = max(dets, key=lambda n: (len(dets[n]), -min(k["V"] for _, k in dets[n])))
print(f"tracking {name}: detected in {len(dets.get(name, []))} of {len(frames)} frames")

# ---------------- predicted path at 6-hour steps ----------------
path = []
t0, t1 = frames[0]["mjd"], frames[-1]["mjd"]
for mjd in np.arange(t0 - 0.25, t1 + 0.26, 0.25):
    try:
        rows = mpchecker(Time(mjd, format="mjd"), center, radius * 1.6, args.vmag, grid)
    except Exception:
        continue
    for k in rows:
        if k["name"] == name:
            path.append(dict(mjd=float(mjd), time=Time(mjd, format="mjd").iso[:16], ra=k["ra"], dec=k["dec"], x=round(k["x"], 1), y=round(k["y"], 1)))
print(f"predicted path: {len(path)} points over {t1 - t0:.1f} days")

obs = []
for fr in frames:
    k = next((k for k in fr["predicted"] if k["name"] == name), None)
    if k:
        obs.append(dict(frame=fr["index"], time=fr["time"][:16], predicted=dict(x=k["x"], y=k["y"], ra=k["ra"], dec=k["dec"]),
                        detected=(dict(x=k["det_x"], y=k["det_y"], snr=k["det_snr"]) if k["detected"] else None)))
if len([o for o in obs if o["detected"]]) >= 2:
    d = [o for o in obs if o["detected"]]
    p0, p1 = d[0], d[-1]
    sep = grid.pixel_to_world(p0["detected"]["x"], p0["detected"]["y"]).separation(
        grid.pixel_to_world(p1["detected"]["x"], p1["detected"]["y"])).arcmin
    days = frames[p1["frame"]]["mjd"] - frames[p0["frame"]]["mjd"]
    print(f"measured motion: {sep:.1f} arcmin in {days:.2f} days = {sep / days / 60:.3f} deg/day")
    rate = dict(arcmin=round(sep, 1), days=round(days, 2), deg_per_day=round(sep / days / 60, 3))
else:
    rate = None

json.dump(dict(week=args.week, template_week=args.template_week, object=name, frames=frames, path=path,
               observations=obs, measured_motion=rate), open(OUT / "track.json", "w"), indent=1)

# ---------------- summary figure ----------------
n = len(frames)
fig, axs = plt.subplots(1, n, figsize=(3.6 * n, 4.2))
for fr, ax in zip(frames, np.atleast_1d(axs)):
    im = plt.imread(OUT / fr["image"])
    ax.imshow(im, origin="upper"); ax.axis("off")
    ax.set_title(f"{fr['time'][:16]}\n{fr['n_exposures']} exposures", fontsize=9)
    ax.plot([p["x"] for p in path], [NPIX - 1 - p["y"] for p in path], ":", color="#facc15", lw=1)
    o = next((o for o in obs if o["frame"] == fr["index"]), None)
    if o:
        ax.plot(o["predicted"]["x"], NPIX - 1 - o["predicted"]["y"], "s", mfc="none", mec="#facc15", ms=12)
        if o["detected"]:
            ax.plot(o["detected"]["x"], NPIX - 1 - o["detected"]["y"], "o", mfc="none", mec="#4ade80", ms=14, mew=1.5)
fig.suptitle(f"{name}: yellow dotted = predicted path, yellow square = predicted position, green = detected", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "track_summary.png", dpi=110)
print("wrote", OUT / "track.json", OUT / "track_summary.png")
