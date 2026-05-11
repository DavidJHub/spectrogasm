"""Extra science measurements computed from the calibrated spectra.

Everything here is derived from artifacts the calibration pipeline
already produces — no extra observations needed.

Measurements:

* ``equivalent_widths``  — line equivalent widths (mÅ) of the V/Ni/Ti
  absorption features, from the same Gaussian fits used for the radial
  velocity. EW is the classical abundance/temperature proxy.

* ``spectral_resolution`` — instrumental R = λ/FWHM derived by fitting
  Gaussian profiles to the Th-Ar lamp peaks the wavelength solution was
  built on. A health metric of the spectrograph for the night.

* ``estimate_snr`` — robust signal-to-noise estimate from the stellar
  continuum (median / 1.4826·MAD over a rolling baseline).

* ``per_element_rv`` — V vs Ni vs Ti mean velocities, an internal
  consistency check on the per-line Doppler measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# --- 1. Equivalent widths --------------------------------------------------

def equivalent_widths(rv_lines: pd.DataFrame) -> pd.DataFrame:
    """Analytic EW (mÅ) for the Gaussian absorption fits.

    For ``I(λ) = c − a·exp(−(λ−μ)²/2σ²)`` the equivalent width is::

        EW = √(2π) · σ · (a / c)
    """
    df = rv_lines.copy()
    df["ew_mA"] = np.where(
        df["ok"] & (df["continuum"] > 0),
        1000.0 * np.sqrt(2.0 * np.pi) * df["sigma"] * (df["depth"] / df["continuum"]),
        np.nan,
    )
    return df[["element", "lambda_lab", "sigma", "depth", "continuum",
               "ew_mA", "ok"]]


# --- 2. Spectral resolution from Th-Ar lamp peaks --------------------------

def _gauss(x, mu, sigma, amp, base):
    return base + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def spectral_resolution(
    thar: pd.DataFrame,
    lines_used: pd.DataFrame,
    half_window: float = 1.2,
    lambda_col: str = "lambda_cal",
) -> pd.DataFrame:
    """Fit a Gaussian to every line used in the solution and return R = λ/FWHM."""
    lam = thar[lambda_col].to_numpy()
    intensity = thar["intensidad"].to_numpy()

    rows: list[dict] = []
    for _, line in lines_used.iterrows():
        lam_atlas = float(line["lambda_atlas"])
        mask = (lam >= lam_atlas - half_window) & (lam <= lam_atlas + half_window)
        x = lam[mask]
        y = intensity[mask]
        if x.size < 5:
            continue

        base0 = float(np.median(y))
        amp0 = float(np.max(y) - base0)
        if amp0 <= 0:
            continue

        p0 = [lam_atlas, 0.1, amp0, base0]
        bounds = (
            [lam_atlas - half_window, 0.02, 0.0, 0.0],
            [lam_atlas + half_window, 1.00, 10.0 * amp0, base0 + amp0],
        )
        try:
            popt, _ = curve_fit(_gauss, x, y, p0=p0, bounds=bounds, maxfev=4000)
        except (RuntimeError, ValueError):
            continue

        mu, sigma, amp, base = popt
        fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
        if fwhm <= 0:
            continue
        rows.append({
            "lambda_atlas": lam_atlas,
            "mu_fit": float(mu),
            "sigma": float(sigma),
            "fwhm": float(fwhm),
            "R": float(mu / fwhm),
            "amplitude": float(amp),
        })

    return pd.DataFrame(rows)


@dataclass
class ResolutionSummary:
    n: int
    R_median: float
    R_mean: float
    R_std: float


def summarize_resolution(per_line: pd.DataFrame) -> ResolutionSummary:
    if per_line.empty:
        return ResolutionSummary(0, float("nan"), float("nan"), float("nan"))
    R = per_line["R"].to_numpy()
    return ResolutionSummary(
        n=int(R.size),
        R_median=float(np.median(R)),
        R_mean=float(np.mean(R)),
        R_std=float(np.std(R, ddof=1)) if R.size > 1 else 0.0,
    )


# --- 3. Signal-to-noise ----------------------------------------------------

def estimate_snr(
    star: pd.DataFrame,
    lambda_col: str = "lambda_rest",
    box: int = 51,
) -> float:
    """Robust per-pixel SNR of the stellar spectrum.

    Subtracts a rolling-median continuum, then SNR = median(I) / σ_robust,
    where σ_robust = 1.4826·MAD over the residual.
    """
    intensity = star["intensidad"].to_numpy(dtype=float)
    if intensity.size < box:
        return float("nan")

    s = pd.Series(intensity)
    baseline = s.rolling(box, center=True, min_periods=1).median().to_numpy()
    resid = intensity - baseline
    mad = np.median(np.abs(resid - np.median(resid)))
    sigma = 1.4826 * mad
    if sigma <= 0:
        return float("nan")
    return float(np.median(intensity) / sigma)


# --- 4. Per-element radial velocity ---------------------------------------

def per_element_rv(rv_lines: pd.DataFrame) -> pd.DataFrame:
    """Group v_kms by element and return mean / std / SEM."""
    good = rv_lines[rv_lines["ok"]]
    rows = []
    for elem, sub in good.groupby("element", sort=False):
        v = sub["v_kms"].to_numpy()
        n = int(v.size)
        std = float(np.std(v, ddof=1)) if n > 1 else 0.0
        rows.append({
            "element": elem,
            "n": n,
            "v_mean_kms": float(np.mean(v)) if n else float("nan"),
            "v_std_kms": std,
            "v_sem_kms": std / np.sqrt(n) if n > 1 else 0.0,
        })
    return pd.DataFrame(rows)


# --- Persistence ---------------------------------------------------------

def save_science_json(
    date: str,
    ew: pd.DataFrame,
    res: pd.DataFrame,
    res_summary: ResolutionSummary,
    snr: float,
    per_elem: pd.DataFrame,
    path: Path,
) -> None:
    import json
    payload = {
        "date": date,
        "snr": snr,
        "resolution": {
            "n": res_summary.n,
            "R_median": res_summary.R_median,
            "R_mean": res_summary.R_mean,
            "R_std": res_summary.R_std,
        },
        "equivalent_widths": ew[["element", "lambda_lab", "ew_mA", "ok"]]
            .to_dict(orient="records"),
        "per_element_rv": per_elem.to_dict(orient="records"),
        "resolution_lines": res.to_dict(orient="records"),
    }
    Path(path).write_text(json.dumps(payload, indent=2))
