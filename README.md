# spectrogasm

It's a wavelength calibration pipeline for échelle/grating spectroscopy, one night at a time (March 12). The physics: a CCD records intensity vs. pixel, but you need intensity vs. wavelength (λ). The mapping pixel→λ depends on the instrument's optomechanical state at the time of exposure, so it must be re-derived for each observation epoch from a Th-Ar lamp taken close in time to the science exposure.
Concretely, the pipeline does this:

Loads three files: the Th-Ar lamp (thor_marzo12.dat), the star (estrella_marzo12.dat), and a Th-Ar atlas (thar_uves.dat) with known laboratory wavelengths.
Detects emission peaks in the lamp with scipy.signal.find_peaks (prominence + min distance).
Filters multi-peaked / blended structures (pico_unico).
Refines each peak center with a 3-point parabolic fit (vertex = sub-pixel centroid).
Builds a seed polynomial (deg 2) from 5 hardcoded manual identifications (pixel↔λ).
Uses the seed to predict λ for every detected peak, then nearest-neighbor matches each prediction against the atlas within ±0.20 Å.
Picks ~20 strong, well-separated matches (prominence-ranked, ≥8 px apart).
Fits the final polynomial (deg 2 vs deg 3, picks lower RMS), optionally rejects outliers >0.08 Å and refits.
Applies the polynomial to both lamp and star, saves CSVs.

That is — conceptually — exactly what IRAF's identify/reidentify/dispcor does, written from scratch.
Where the current code is weak (objective)
Before generalizing, fix these or you're scaling up problems:

Centroiding by 3-point parabola is the crudest method. It's biased when the line isn't symmetric or when the true peak isn't near the central pixel. Use a Gaussian fit over 5–7 px (astropy.modeling.models.Gaussian1D + LevMarLSQFitter). Sub-pixel error drops by ~3–5×.
np.polyfit is numerically poor for high-degree fits over wide pixel ranges (Vandermonde matrix is ill-conditioned). Use numpy.polynomial.Chebyshev.fit — same expressive power, much better conditioning. This is what IRAF/specutils default to.
Outlier rejection is a hard 0.08 Å cut, applied once. Not robust. Use astropy.stats.sigma_clip iteratively (typically κ=3, refit, repeat until stable).
5 manual seed lines hardcoded inside the notebook. This breaks the moment you process a different night where the instrument has drifted by more than ±0.20 Å — the nearest-atlas match will be wrong. Generalization needs to break this dependency (see below).
pd.read_csv(..., engine="python", sep=r"\s+") is ~30× slower than np.loadtxt for pure numeric whitespace data. Doesn't matter for one file; matters for batch.
No air↔vacuum λ handling. UVES atlas is air wavelengths in the visible. If you ever need radial velocities, you need vacuum (Edlén/Ciddor formula via specutils or PyAstronomy.pyasl.airtovac2).
No barycentric correction. Required for any RV science. astropy.coordinates.SkyCoord.radial_velocity_correction does this in 3 lines if you have RA/Dec, observatory location, and DATE-OBS.
No persistence of the wavelength solution per night. You should be able to audit drift across nights by saving coefficients + RMS + line list per epoch.
No FITS or header-based metadata. .dat files lose DATE-OBS. The date currently lives only in the filename, which is fragile.
Two near-duplicate pipeline blocks (cells 15 and 25 are 90% the same). Refactor into functions before scaling.

Fundamental steps to generalize for multiple .dat files
The core insight: what changes per night is the pixel→λ map; what stays constant is the algorithm and the atlas. So you build a pure pipeline that takes (thar_path, star_path, date_obs) → (calibrated_thar, calibrated_star, solution_metadata), and call it in a loop.
Step 1 — Define a metadata source for date/time. You have two options, ranked by robustness:

(a) A manifest CSV/YAML you maintain: star_file, thar_file, date_obs, ra, dec.
(b) Filename parsing with a strict regex (e.g., estrella_(?P<month>\w+)(?P<day>\d+)\.dat) plus a month-name → date map.

Option (a) is what professional pipelines do. Option (b) is fine for a teaching dataset but breaks the moment naming conventions slip.
Step 2 — Pair each star with its calibration lamp. For one Th-Ar per star, just match by filename. For real observatory data with multiple lamps per night, pair each star with the closest-in-time Th-Ar (or bracket: one before + one after, then interpolate the solution — best practice).
Step 3 — Refactor the notebook into a module of pure functions. Roughly:

load_spectrum(path) -> (pixel, intensity) using np.loadtxt
load_atlas(path, lam_range) -> np.ndarray
detect_peaks(intensity, prominence, distance) -> indices
centroid_gaussian(pixel, intensity, peak_idx, window=5) -> (mu, sigma, amp, fit_rms) — replace the parabola
match_to_atlas(centroids, atlas, seed_coeffs, tol) -> DataFrame
fit_solution(pix, lam, max_deg=4, sigma=3.0) -> (Chebyshev, rms, mask) — sigma-clipped Chebyshev
apply_solution(pixel, cheb) -> wavelength
save_solution(out_dir, date_obs, cheb, rms, lines_used)

Step 4 — Solve the seed problem (this is the key generalization issue). The hardcoded 5 manual lines only work for one night. Three solutions, ordered by quality:

Best: cross-correlation bootstrapping. Calibrate one night manually (the night you already have). Save its full Th-Ar in pixel space as a template. For every new night, cross-correlate the new Th-Ar against the template to get an integer pixel shift Δp. Use λ_new(p) = λ_ref(p − Δp) as the seed. This eliminates manual seed lines entirely. Implementation: scipy.signal.correlate or astropy.modeling cross-correlation on the lamp spectra, since the lamp pattern is essentially identical night to night.
Good: persistent seed library. Keep your manual identifications in a CSV. After each successful calibration, append the (pixel, λ) pairs to a growing seed library tagged by date. New nights get seeded from the most recent successful run after a coarse pixel-shift correction.
Acceptable: re-identify manually each night. Same as now, but loaded from a per-night JSON: {"2024-03-12": [[71.685, 5973.664], ...], "2024-03-15": [...]}.

Step 5 — Per-night artifact directory. For each (star, date) produce a folder containing: calibrated spectra, the final coefficients (as JSON), the line list used, the residual RMS, and a diagnostic plot (residuals vs. pixel — flat residuals = good fit, structured residuals = need higher degree or different model). This is what lets you spot a bad night before it contaminates downstream analysis.
Step 6 — Drift monitoring across epochs. Once you have a per-night solutions table, plot the polynomial coefficients (and RMS) vs. date. Sudden jumps reveal real instrument events (focus changes, grating moves); slow drifts validate the assumption that you need per-night calibration. This costs 10 lines of code and is the single highest-value diagnostic for a multi-night campaign.
Step 7 — CLI / batch driver. Wrap it as either a Typer CLI (python calibrate.py --manifest manifest.csv --out-dir results/) or a Snakemake/Prefect workflow if you want reproducibility and parallelism. For <50 nights the CLI is fine. Beyond that, Snakemake gives you free re-run-only-what-changed semantics.
