# spectrogasm

Wavelength-calibration pipeline for échelle/grating spectroscopy, one night
at a time. From raw pixel-vs-intensity `.dat` files for a Th-Ar lamp and a
science star, it produces:

- a per-night wavelength solution `λ(pixel)` (polynomial coefficients + RMS),
- the calibrated stellar spectrum (with optional telluric rest-frame correction),
- the stellar **radial velocity** from Doppler shifts of V/Ni/Ti absorption lines,
- instrumental **resolution R = λ/FWHM** measured on the lamp,
- per-line **equivalent widths** of the stellar features,
- a robust **SNR** estimate, and
- an **interactive HTML dashboard** that aggregates every night.

The physics: a CCD records intensity vs. pixel; you need intensity vs.
wavelength. The mapping pixel→λ depends on the spectrograph's
optomechanical state at the time of exposure, so it has to be re-derived
per night from a Th-Ar lamp taken close in time. Once λ is known, the
star's spectrum can be put in the rest frame and stellar lines compared to
laboratory wavelengths to extract physical quantities.

---

## 1 · Repository layout

```
spectrogasm/
├── data/
│   ├── thar_uves.dat                # shared Th-Ar atlas (laboratory λ)
│   ├── estrella_YY-MM-DD.dat        # one stellar spectrum per night
│   └── thar_YY-MM-DD.dat            # one lamp spectrum per night
├── seeds/
│   └── YY-MM-DD.json                # optional manual seed for a night
├── results/
│   └── YYYY-MM-DD/                  # one folder per night (auto-created)
│       ├── thar_calibrado.csv
│       ├── estrella_calibrada.csv
│       ├── lineas_calibracion.csv
│       ├── rv_lines.csv
│       ├── equivalent_widths.csv
│       ├── resolution_per_line.csv
│       ├── solution.json
│       ├── rv.json
│       ├── science.json
│       └── *.png                    # diagnostic figures
├── fingerprint.json                 # lamp template, refreshed by reference night
└── spectrogasm/                     # the package
```

### File-naming convention

| File                         | Pattern                       | Example                  |
|------------------------------|-------------------------------|--------------------------|
| Stellar spectrum (per night) | `data/estrella_YY-MM-DD.dat`  | `estrella_24-03-12.dat`  |
| Lamp spectrum (per night)    | `data/thar_YY-MM-DD.dat`      | `thar_24-03-12.dat`      |
| Atlas (shared)               | `data/thar_uves.dat`          | `thar_uves.dat`          |
| Manual seed (optional)       | `seeds/YY-MM-DD.json`         | `seeds/24-03-12.json`    |

`YY` is mapped to `20YY` for the ISO date used in `results/`. The atlas
shares the `thar_` prefix but is excluded by the strict `YY-MM-DD` regex
in `manifest.discover_nights`.

---

## 2 · Quick start

```bash
# Run every discovered night (data/ + seeds/ scanned automatically)
python -m spectrogasm

# Run only specific dates
python -m spectrogasm 2024-03-12 2024-03-26

# Build the interactive HTML dashboard from existing results/
python -m spectrogasm.dashboard
# → results/dashboard.html  (self-contained, opens in any browser)
```

Programmatic use:

```python
from spectrogasm import calibrate_night, NIGHTS, CalibrationParams

result = calibrate_night(NIGHTS[0], CalibrationParams(make_plots=False))
print(result.solution.rms, result.rv_summary.v_mean_kms)
```

---

## 3 · Pipeline, step by step

`pipeline.calibrate_night(night, params)` runs 12 stages. All paths and
filenames live in `manifest.py`; all tunables in `CalibrationParams`.

### Step 1 — Load spectra and atlas
- `io.load_spectrum(path)` reads a two-column ASCII file (`pixel`,
  `intensidad`) into a DataFrame.
- `io.load_atlas(path, lam_min, lam_max)` parses the multi-column UVES
  Th-Ar atlas, normalises decimal commas → dots, drops non-numeric
  tokens, deduplicates, and returns a sorted `pd.Series` of laboratory
  wavelengths between `lam_min` and `lam_max`.

### Step 2 — Detect and refine lamp peaks
- `peaks.detect_peaks(intensity, prominence, distance)` calls
  `scipy.signal.find_peaks` on the lamp spectrum. Returns the integer
  pixel indices and prominences of every candidate peak.
- `peaks.refine_centroids(pixel, intensity, idx, prom, window=3)`
  refines each integer peak to a sub-pixel centroid by fitting a
  Gaussian `A·exp(−(x−μ)²/2σ²) + offset` over ±3 pixels with
  `scipy.optimize.curve_fit`. Peaks are rejected if the fit fails, the
  centre drifts > 1 px from the integer peak, σ ≤ 0, σ > 5 px, or A ≤ 0
  (catches blends and noise spikes).

### Step 3 — Build the seed polynomial
The seed is a **coarse** pixel→λ map needed to nominate atlas candidates
before the final fit. Two routes:

- **Manual seed** (`seeds/YY-MM-DD.json`): a hand-identified short list
  of `(pixel, λ)` anchors. `seeds.seed_polynomial(seed_df, degree=2)`
  fits a degree-2 polynomial through them. This is the route used for
  the **reference night** (`manifest.REFERENCE_DATE`, default
  `2024-03-12`).
- **Fingerprint bootstrap** (`fingerprint.bootstrap_seed`): for any
  later night without a manual seed, the lamp's intensity profile is
  cross-correlated against the fingerprint saved by the reference
  night. The integer pixel shift is applied to the reference's
  `(pixel, λ)` table and used as the seed.

### Step 4 — Match peaks to atlas
`matching.match_to_atlas(centroids, atlas, seed_coef, tol)`:
1. Predict `λ_seed` for every refined peak using the seed polynomial.
2. Restrict the atlas to `[min λ_seed − pad, max λ_seed + pad]`.
3. For each peak, find the nearest atlas line. Keep only matches with
   `|λ_atlas − λ_seed| < tol` (default 0.20 Å).
4. Deduplicate so each atlas line is claimed by at most one peak (the
   closest one, with prominence as tiebreaker).

A `MatchReport` with the predicted λ window, atlas density and the
nearest absolute difference is also returned, so an empty match is
diagnosable.

`matching.select_strong_lines(matched, n_lines=20, min_separation=8.0)`
then ranks by prominence and skims off up to 20 well-separated lines
(≥ 8 pixels apart) to feed the final fit.

### Step 5 — Final polynomial fit with κ-σ clipping
`fitting.fit_solution(selected, kappa=3.0, max_iter=10)`:
1. `choose_polynomial`: fits both degree 2 and 3, picks degree 3 only if
   it cuts RMS by ≥ 20 % (Occam's razor — avoid overfitting).
2. `sigma_clip_polyfit`: iteratively refits, masking lines whose
   residual exceeds `kappa·σ` of the residual distribution. Stops when
   the mask stabilises or when fewer than `degree + 2` lines would
   survive.

Returns a `Solution` with the coefficients, degree, RMS, the surviving
lines (with `lambda_fit` and `residual_final` columns), and the
iteration count.

### Step 6 — Apply pixel → λ
`solution.apply(pixel)` (= `np.polyval(coef, pixel)`) is run on both the
lamp and the star. Results are stored in a new `lambda_cal` column.

### Step 7 — Telluric rest-frame correction (star only)
`telluric.telluric_velocity(date)` reads the per-night telluric velocity
from `manifest.TELLURIC_KMS` (linear interpolation if the date is between
two logged values). `telluric.to_rest_frame(lambda_cal, v_kms)` divides
the calibrated stellar wavelengths by `(1 + v/c)` so atmospheric features
sit at their lab values, producing the `lambda_rest` column.

### Step 8 — Stellar radial velocity
`radial_velocity.measure_rv_night(star)` fits an inverted Gaussian
`I(λ) = c − a·exp(−(λ − μ)²/2σ²)` over a ±2 Å window around each of the
six laboratory lines (V 6039.7256, V 6058.142, V 6081.4422, V 6090.2084,
Ni 6108.1159, Ti 6126.216). For each line:
`v_kms = c·(λ_obs − λ_lab) / λ_lab`.
`summarize_rv` aggregates: mean, median, σ between lines, SEM = σ/√N.

### Step 9 — Extra science
`science.equivalent_widths(rv_lines)` — analytic
`EW = √(2π)·σ·(depth/continuum)`, in mÅ.
`science.spectral_resolution(thar, lines_used)` — Gaussian fits to the
calibration peaks; `R = λ/FWHM` per peak; median + std summary.
`science.estimate_snr(star)` — robust SNR from a rolling-median
continuum: `median(I) / (1.4826·MAD(residuals))`.
`science.per_element_rv(rv_lines)` — V/Ni/Ti grouped means with SEMs.

### Step 10 — Persist artifacts
Every DataFrame is written as CSV; metadata as JSON (see §4).

### Step 11 — Refresh the fingerprint (reference night only)
If this run used the manual seed, `fingerprint.save_fingerprint` writes
the lamp profile + identified `(pixel, λ)` pairs to `fingerprint.json`
so future nights can bootstrap from it.

### Step 12 — Diagnostic plots
`plotting.plot_spectrum`, `plotting.plot_lines_used`,
`radial_velocity.plot_rv_fits` produce the per-night PNGs (see §5).

After `calibrate_all` finishes, `__main__` calls `drift.plot_drift` and
`radial_velocity.plot_rv_drift` to make multi-night summary figures.

---

## 4 · Output reference — DataFrames and JSON

All CSVs live in `results/<YYYY-MM-DD>/`.

### 4.1 · `thar_calibrado.csv` (Th-Ar lamp, calibrated)

| Column        | Type  | Units | Meaning                                              |
|---------------|-------|-------|------------------------------------------------------|
| `pixel`       | int   | px    | Detector pixel index, copied verbatim from the input |
| `intensidad`  | float | counts| Raw lamp intensity at that pixel                     |
| `lambda_cal`  | float | Å     | Wavelength from the final polynomial `λ(pixel)`      |

### 4.2 · `estrella_calibrada.csv` (stellar spectrum, calibrated)

| Column         | Type  | Units | Meaning                                                |
|----------------|-------|-------|--------------------------------------------------------|
| `pixel`        | int   | px    | Detector pixel index                                   |
| `intensidad`   | float | counts| Raw stellar intensity                                  |
| `lambda_cal`   | float | Å     | Wavelength from the polynomial (Earth/topocentric frame) |
| `lambda_rest`  | float | Å     | `lambda_cal / (1 + v_tel/c)` — telluric rest frame     |

### 4.3 · `lineas_calibracion.csv` (lines that survived the κ-σ clip)

These are the lines the final polynomial was fit on.

| Column            | Type  | Units | Meaning                                                              |
|-------------------|-------|-------|----------------------------------------------------------------------|
| `pixel_centro`    | float | px    | Sub-pixel Gaussian centroid of the lamp peak                         |
| `intensidad`      | float | counts| Fitted peak height (`A + offset`)                                    |
| `prominencia`     | float | counts| `scipy.signal.find_peaks` prominence — local-contrast measure        |
| `lambda_seed`     | float | Å     | `λ` predicted by the **seed** polynomial at `pixel_centro`           |
| `lambda_atlas`    | float | Å     | Nearest UVES atlas line within `match_tol`                           |
| `diff_seed`       | float | Å     | `lambda_atlas − lambda_seed` (signed)                                |
| `absdiff`         | float | Å     | `\|diff_seed\|`                                                      |
| `lambda_fit`      | float | Å     | `λ` predicted by the **final** polynomial at `pixel_centro`          |
| `residual_final`  | float | Å     | `lambda_fit − lambda_atlas` (used to compute the final RMS)          |

### 4.4 · `rv_lines.csv` (per-line Doppler fits)

One row per laboratory line in `STELLAR_LINES`.

| Column        | Type  | Units | Meaning                                                                          |
|---------------|-------|-------|----------------------------------------------------------------------------------|
| `element`     | str   | —     | `"V"`, `"Ni"` or `"Ti"`                                                          |
| `lambda_lab`  | float | Å     | Laboratory rest wavelength of the line                                           |
| `lambda_obs`  | float | Å     | Fitted Gaussian centre on the `lambda_rest` axis (`NaN` if `ok=False`)           |
| `sigma`       | float | Å     | Gaussian width of the absorption                                                 |
| `depth`       | float | counts| Amplitude `a` in `c − a·exp(…)`                                                  |
| `continuum`   | float | counts| Local continuum level `c`                                                        |
| `v_kms`       | float | km/s  | `c·(lambda_obs − lambda_lab)/lambda_lab` — line's contribution to radial velocity |
| `rms`         | float | counts| Residual RMS of the Gaussian fit                                                 |
| `n_points`    | int   | —     | Pixels used inside the ±2 Å window                                               |
| `ok`          | bool  | —     | `True` if the fit converged within bounds                                        |
| `reason`      | str   | —     | Empty when `ok=True`; otherwise why the fit was rejected                         |

### 4.5 · `equivalent_widths.csv`

| Column        | Type  | Units | Meaning                                                                          |
|---------------|-------|-------|----------------------------------------------------------------------------------|
| `element`     | str   | —     | V/Ni/Ti                                                                          |
| `lambda_lab`  | float | Å     | Laboratory wavelength                                                            |
| `sigma`       | float | Å     | From the line's Gaussian fit                                                     |
| `depth`       | float | counts| From the fit                                                                     |
| `continuum`   | float | counts| From the fit                                                                     |
| `ew_mA`       | float | mÅ    | `1000·√(2π)·σ·(depth/continuum)` — analytic EW of the Gaussian                   |
| `ok`          | bool  | —     | Carried over from the underlying fit                                             |

### 4.6 · `resolution_per_line.csv`

One row per Th-Ar line used in the solution; its Gaussian width gives R.

| Column          | Type  | Units | Meaning                                                            |
|-----------------|-------|-------|--------------------------------------------------------------------|
| `lambda_atlas`  | float | Å     | Atlas wavelength of the lamp line                                  |
| `mu_fit`        | float | Å     | Fitted centre on `lambda_cal`                                      |
| `sigma`         | float | Å     | Gaussian σ of the lamp line profile                                |
| `fwhm`          | float | Å     | `2·√(2 ln 2)·σ ≈ 2.355·σ`                                         |
| `R`             | float | —     | `mu_fit / fwhm` — instrumental resolving power at that wavelength  |
| `amplitude`     | float | counts| Fitted peak amplitude `A`                                          |

### 4.7 · `solution.json`

```json
{
  "date":          "2024-03-12",
  "degree":        2,                 // 2 or 3
  "coef":          [c_n, ..., c_0],   // numpy.polyval order: highest first
  "rms":           0.0184,            // Å
  "n_lines":       18,
  "telluric_kms":  -23.913,
  "seed_origin":   "manual"           // or "fingerprint"
}
```

### 4.8 · `rv.json`

```json
{
  "date":          "2024-03-12",
  "n_used":        6,                 // lines with ok=True
  "v_mean_kms":    -14.213,
  "v_median_kms":  -14.198,
  "v_std_kms":     0.082,             // dispersion across the 6 lines
  "v_sem_kms":     0.033              // = v_std_kms / sqrt(n_used)
}
```

### 4.9 · `science.json`

```json
{
  "date": "2024-03-12",
  "snr":  214.7,
  "resolution": {
    "n":         5,
    "R_median":  19780,
    "R_mean":    19735,
    "R_std":     230
  },
  "equivalent_widths": [
    {"element": "V", "lambda_lab": 6039.7256, "ew_mA": 84.0, "ok": true},
    ...
  ],
  "per_element_rv": [
    {"element": "V",  "n": 4, "v_mean_kms": ..., "v_std_kms": ..., "v_sem_kms": ...},
    {"element": "Ni", "n": 1, ...},
    {"element": "Ti", "n": 1, ...}
  ],
  "resolution_lines": [
    {"lambda_atlas": ..., "mu_fit": ..., "sigma": ..., "fwhm": ..., "R": ..., "amplitude": ...}
  ]
}
```

---

## 5 · Output reference — figures

All PNGs are written at 220 dpi with the project palette
(`#870047` primary, `#ff6aa7` secondary, plus derived shades).

### Per-night

| File                                     | What it shows                                                                                                     |
|------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `results/<date>/thar.png`                | Th-Ar lamp spectrum on the calibrated λ axis. Soft pink fill below the curve emphasises the emission profile.     |
| `results/<date>/estrella.png`            | Stellar spectrum on the calibrated λ axis. Line only (no fill) so absorptions stay readable on the bright continuum. |
| `results/<date>/lineas.png`              | Th-Ar with circle markers and labels on the lines that survived the κ-σ clip and entered the final polynomial.    |
| `results/<date>/rv_fits.png`             | 3×2 grid, one panel per V/Ni/Ti line: data points (dark), fitted Gaussian (primary), shaded fill, dashed vertical at `lambda_obs`, dotted vertical at `lambda_lab`. Title carries the per-line v in km/s. |

### Across nights

| File                              | What it shows                                                                                                                     |
|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `results/drift.png`               | One subplot per polynomial coefficient + a final panel with RMS, all vs. date. Sudden jumps = instrument events; smooth = drift. |
| `results/rv_drift.png`            | Mean stellar v_rad per night with **±SEM** error bars (line + filled marker) over a translucent **±1σ** band (line scatter).      |
| `results/dashboard.html`          | Interactive Plotly dashboard, see §6.                                                                                             |

#### What ±1σ vs ±SEM mean on the RV plot

Per night we measure v on **N** independent lines. The reported summary is:

- **σ** (`v_std_kms`) = sample standard deviation **between lines**.
  Tells you how much the individual line measurements scatter around their mean.
- **SEM** (`v_sem_kms`) = `σ/√N` = uncertainty **of the mean itself**.
  Averaging N independent measurements shrinks the error of the average by √N.

The translucent band shows σ (where individual lines fall); the error bars
show SEM (how well the per-night mean is known).

---

## 6 · Dashboard

`python -m spectrogasm.dashboard [results_dir] [out.html]`

A self-contained HTML file (Plotly via CDN) with five sections:

1. **Header + summary cards** — total nights, date range, mean RMS,
   mean v_rad, median R, median SNR.
2. **Resumen por noche** — sortable table with degree, RMS, N lines,
   `v_tel`, `v_rad ± SEM`, R, SNR, seed origin.
3. **Calidad de la calibración** — RMS per night, N lines per night,
   coefficient drift (one subplot per coefficient).
4. **Velocidad radial** — `v_rad` per night with SEM error bars and
   per-element breakdown (V/Ni/Ti each with its own colour).
5. **Diagnóstico y ciencia** — R per night, SNR per night, EW heatmap
   (line × night, sequential pink colormap).
6. **Detalle por noche** — accordion (`<details>` per night) with chips
   for the night's KPIs and three interactive Plotly figures: Th-Ar
   with line overlays, stellar spectrum, and the 6-panel Doppler grid.

---

## 7 · Configuration (`CalibrationParams`)

Defaults in `pipeline.py`:

| Field                  | Default | Meaning                                                              |
|------------------------|---------|----------------------------------------------------------------------|
| `atlas_lam_min`        | 3000.0  | Å — atlas low cut                                                    |
| `atlas_lam_max`        | 10000.0 | Å — atlas high cut                                                   |
| `peak_prominence`      | 2000.0  | Min. prominence for `find_peaks` on the lamp                         |
| `peak_distance`        | 4       | Min. integer-pixel separation between peaks                          |
| `match_tol`            | 0.20    | Å — max. `|λ_atlas − λ_seed|` accepted in the matching step          |
| `n_lines`              | 20      | Cap on lines fed to the final polynomial fit                         |
| `min_line_separation`  | 8.0     | Min. pixel separation between lines kept for the final fit           |
| `sigma_clip_kappa`     | 3.0     | κ for iterative residual clipping                                    |
| `sigma_clip_max_iter`  | 10      | Max. clipping iterations                                             |
| `rv_half_window`       | 2.0     | Å — half-width of the window used to fit each stellar line           |
| `make_plots`           | True    | Skip PNG generation when `False`                                     |

---

## 8 · Module reference

| Module                      | Purpose                                                                    |
|-----------------------------|----------------------------------------------------------------------------|
| `manifest.py`               | Auto-discovers nights from `data/`, exposes `NIGHTS`, `REFERENCE_NIGHT`.   |
| `io.py`                     | Spectrum/atlas loading, CSV saving.                                        |
| `peaks.py`                  | `find_peaks` + Gaussian sub-pixel centroiding with rejection rules.        |
| `seeds.py`                  | Manual seed JSON loader, seed polynomial fit.                              |
| `fingerprint.py`            | Lamp fingerprint persistence + cross-correlation seed bootstrap.           |
| `matching.py`               | Seed-driven nearest-atlas matching + strong-line selection.                |
| `fitting.py`                | Degree choice (2 vs 3) + iterative κ-σ polynomial fit.                     |
| `telluric.py`               | Per-night telluric velocity table + rest-frame shift.                      |
| `radial_velocity.py`        | Doppler-shift line fits, summary, per-night and drift plots.               |
| `science.py`                | EW, spectral resolution R, robust SNR, per-element RV.                     |
| `pipeline.py`               | `calibrate_night`, `calibrate_all`, `CalibrationParams`, `CalibrationResult`. |
| `plotting.py`               | Lamp/star/lines plots.                                                     |
| `drift.py`                  | Multi-night coefficient + RMS plot.                                        |
| `dashboard.py`              | Interactive HTML report.                                                   |
| `style.py`                  | Centralised palette + matplotlib `rcParams` for poster-grade figures.      |
| `__main__.py`               | CLI: `python -m spectrogasm [date ...]`.                                   |

---

## 9 · Conventions and notation

### Symbols and units

| Symbol         | Meaning                                                  | Units    |
|----------------|----------------------------------------------------------|----------|
| `λ`            | Wavelength                                               | Å        |
| `pixel`        | Detector pixel index, 0-based                            | px       |
| `λ_seed`       | Wavelength predicted by the seed polynomial              | Å        |
| `λ_atlas`      | Tabulated UVES Th-Ar wavelength                          | Å        |
| `λ_fit`        | Wavelength predicted by the final polynomial             | Å        |
| `λ_cal`        | Calibrated wavelength applied to the spectrum            | Å        |
| `λ_rest`       | `λ_cal` shifted to the rest frame using `v_tel`          | Å        |
| `λ_lab`        | Laboratory rest wavelength of a stellar line             | Å        |
| `λ_obs`        | Fitted observed centre of a stellar line on `λ_rest`     | Å        |
| `c`            | Speed of light, `299 792.458`                            | km/s     |
| `v_tel`        | Telluric radial velocity (Earth-frame correction)        | km/s     |
| `v_rad`        | Stellar radial velocity from V/Ni/Ti Doppler             | km/s     |
| `R`            | Resolving power = `λ/FWHM`                               | —        |
| `σ`            | Gaussian width                                           | Å or px  |
| `FWHM`         | `2·√(2 ln 2)·σ ≈ 2.355·σ`                               | Å        |
| `EW`           | Equivalent width                                         | mÅ       |
| `RMS`          | Root-mean-square of fit residuals                        | Å        |
| `SEM`          | Standard error of the mean = `σ/√N`                      | km/s     |
| `MAD`          | Median absolute deviation                                | counts   |
| `SNR`          | Signal-to-noise ratio = `median(I) / (1.4826·MAD)`       | —        |

### Doppler conventions

- `v > 0` means the stellar features are red-shifted relative to lab
  (source receding).
- `v_tel` is the velocity at which **telluric** lines were observed.
  The pipeline divides `λ_cal` by `(1 + v_tel/c)` so atmospheric
  features land at their rest values.
- Stellar `v_rad` is then computed from `λ_rest`. This is approximately a
  barycentric-frame velocity, modulo small effects (no full
  barycentric+heliocentric correction is applied; for sub-km/s absolute
  RV one would add `astropy.coordinates`-based corrections).

### Polynomial coefficient order

`solution.coef` follows `numpy.polyval`: highest-order term **first**,
constant term **last**. So a degree-2 solution
`[a, b, c]` evaluates as `λ = a·pixel² + b·pixel + c`. The dashboard's
coefficient panels are labelled `c2, c1, c0` in that order.

### Gaussian models

| Where          | Form                                              | Notes                                |
|----------------|---------------------------------------------------|--------------------------------------|
| Lamp centroids | `A·exp(−(x−μ)²/2σ²) + offset`                     | Emission, fit on ±3 px               |
| Th-Ar R        | `base + A·exp(−(x−μ)²/2σ²)`                       | Emission, fit on ±1.2 Å              |
| Stellar RV/EW  | `c − a·exp(−(λ−μ)²/2σ²)`                          | Absorption, fit on ±2 Å              |

### Stellar line list (`radial_velocity.STELLAR_LINES`)

| Element | λ_lab (Å, air) |
|---------|----------------|
| V       | 6039.7256      |
| V       | 6058.1420      |
| V       | 6081.4422      |
| V       | 6090.2084      |
| Ni      | 6108.1159      |
| Ti      | 6126.2160      |

### Colour palette (`style.py`)

| Token       | Hex       | Used for                                  |
|-------------|-----------|-------------------------------------------|
| `PRIMARY`   | `#870047` | Main signal lines, fitted curves, axes    |
| `SECONDARY` | `#ff6aa7` | Markers, secondary series, fills          |
| `DARK`      | `#3a0020` | Tertiary series, dark accents             |
| `LIGHT`     | `#ffc2da` | Soft fills, error bands, sequential cmap low end |
| `MUTED`     | `#7a5060` | Neutral grouping accent                   |
| `INK`       | `#1a1014` | Text and axis colour                      |

`style.apply_style()` runs on `import spectrogasm` and installs poster-grade
matplotlib defaults (220 dpi saves, bold titles, semibold labels, soft
grid `#dcc6d2`, no top/right spines).

---

## 10 · Adding a new night

1. Drop `data/estrella_YY-MM-DD.dat` and `data/thar_YY-MM-DD.dat` in.
2. (Optional) If the lamp has drifted enough that the fingerprint
   bootstrap fails, write a manual seed at `seeds/YY-MM-DD.json`:
   ```json
   {
     "date": "20YY-MM-DD",
     "lines": [
       {"pixel": 71.685,  "lambda": 5973.664},
       {"pixel": 145.290, "lambda": 5994.128}
     ]
   }
   ```
3. (Optional) Add the night's telluric velocity to
   `manifest.TELLURIC_KMS`. Missing dates inside the logged range are
   linearly interpolated.
4. `python -m spectrogasm` — the new night is auto-discovered.
5. `python -m spectrogasm.dashboard` to refresh the HTML.
