# GhostShift

Spot-the-difference for the whole sky. GhostShift lines up two SPHEREx sweeps of the same
patch of sky, cancels out everything that stayed put, and shows only what moved. Each mover is
checked against the Minor Planet Center so the viewer can say "known asteroid" or "unidentified".

Built for the NASA Space Apps 2026 challenge *Planet X and SPHEREx*.

## Pipeline: convert, compare, convert

| Stage | Script | What it does |
|---|---|---|
| 1. Download | `main.py` | Finds every SPHEREx exposure covering a field in two survey weeks (IRSA SIA API) and downloads small FITS cutouts. |
| 2. Compare | `diff.py` | Reprojects both epochs onto one grid, subtracts, finds lone "appeared"/"vanished" sources, rejects star residuals and cosmic rays, labels each with MPChecker. |
| 2b. Track | `track.py` | One aligned image per visit within a week, each compared against a static template from the other sweep; detects the asteroid in every visit and overlays the Minor Planet Center's predicted path. |
| 3. Publish | `export_web.py` | Converts the aligned arrays, the candidate list and the track frames into PNG layers + a manifest for the web viewer. |
| Viewer | `web/index.html` | Zoom, pan, blink (Space), difference (D), motion track with time slider (T, arrows, Play), click a marker for details and catalogue links. |
| Helper | `scan_fields.py` | Scans the ecliptic for fields with two-epoch coverage and bright known asteroids. |

## Run it

```bash
pip install astroquery astropy reproject numpy scipy matplotlib pandas requests pillow
```

```bash
python main.py --ra 28.17 --dec 11.44 --size 1.0 --weeks 2025W30 2026W30 --cutouts cutouts_ast
```

```bash
python diff.py --ra 28.17 --dec 11.44 --size 1.0 --cutouts cutouts_ast --out out_ast
```

```bash
python track.py --ra 28.67 --dec 11.91 --size 1.2 --cutouts cutouts_slipher --out out_slipher --week 2026W30 --template-week 2025W30
```

```bash
python export_web.py "out_slipher:Slipher track" "out_ast:Asteroid field"
```

```bash
python web/serve.py 8765
```

(`serve.py` is a plain static server that disables caching, so edits to `index.html` show on reload.)

Then open http://localhost:8765.

## Results so far

- Field centred on RA 28.17, Dec +11.44 (on the ecliptic), SPHEREx detector D2 (1.1 to 1.6 microns),
  visits 2025-07-24 and 2026-07-22.
- Asteroid (1766) Slipher (V = 17.4) detected at 21 sigma and matched to its predicted position within 3 arcsec.
- (2060) Chiron (V = 18.5) is visible at 10 sigma but rejected because a star of similar brightness sits two pixels away.
- Zero false "discoveries" after the cosmic-ray and star-residual filters.

## Things we learned about SPHEREx data

- Each visit is four exposures stepped 0.19 degrees apart. Where only two overlap, a median is a plain average and
  cosmic rays leak through, so a real source must be present in every exposure of the visit.
- Do not mask the OUTLIER flag (bit 19). SPHEREx flags a pixel as an outlier when it disagrees with other
  visits of the same sky, which is exactly what a moving asteroid does. Masking it erased Slipher from three of
  four visits. Mask the hardware flags only and let the every-exposure test handle cosmic rays.
- The same star lands on a different wavelength of the detector in each visit, so stars never cancel exactly.
  Hundreds of star residuals are normal; reject them by checking the other epoch, not by trusting the difference alone.
- The 102 all-sky spectral cubes are not public yet (planned 2026 to 2027). Today's products are per-exposure
  spectral images, which is why we build our own aligned images.
- Asteroid lookups: MPChecker (Minor Planet Center) works from Python; JPL's identification API timed out and
  SkyBoT blocked our requests.

## Honest framing

SPHEREx is not sensitive enough to see Planet Nine. The tool genuinely finds asteroids, comets and other movers.
Planet X is the story; known asteroids are the proof.
