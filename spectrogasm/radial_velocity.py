"""Stellar radial velocity from Doppler shifts of V, Ni, Ti absorption lines.

The calibrated stellar spectrum ``estrella_calibrada.csv`` carries a
``lambda_rest`` column whose telluric features have already been brought
to rest. Any residual shift of a stellar absorption line w.r.t. its
laboratory wavelength is then the star's radial velocity::

    v = c * (lambda_obs - lambda_lab) / lambda_lab

Two estimators are provided:

* ``measure_rv_night`` — per-line Gaussian fit with robust continuum,
  bounded sigma, and one pass of residual sigma-clipping. Independent
  estimate per line.
* ``measure_rv_night_joint`` — single non-linear fit of all six lines
  sharing the instrumental ``sigma`` (and the night's ``v_kms``), with
  per-line continuum and depth. Reduces the variance of v_rad on noisy
  nights at the cost of assuming a common instrumental width.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares


C_KMS = 299_792.458


# Laboratory rest wavelengths (air, Å) for the stellar absorption lines
# used to measure the radial velocity.
#
# HD 49331 is an M1 III giant: its visible spectrum is dominated by
# molecular bands (TiO, CN) that wash out weak atomic features. The
# 2024-03-12 and 2024-03-19 rv_diagnostic plots show no absorption at
# lambda_lab for V 6039.7256, V 6081.4422 and V 6090.2084 (the deepest
# features in those windows are unrelated lines ~+1 A redward, which
# previously biased the CCF estimator to ~+70 km/s). Keep only the
# three lines that are reliably visible in this spectral type.
STELLAR_LINES: list[tuple[str, float]] = [
    ("V",  6058.1420),
    ("Ni", 6108.1159),
    ("Ti", 6126.2160),
]

# Full original line list. Re-enable selectively if running on a star of
# a different spectral type where the V lines are stronger.
STELLAR_LINES_EXTENDED: list[tuple[str, float]] = [
    ("V",  6039.7256),  # disabled: no absorption at lam_lab in HD 49331 (M1 III)
    ("V",  6058.1420),
    ("V",  6081.4422),  # disabled: no absorption at lam_lab in HD 49331 (M1 III)
    ("V",  6090.2084),  # disabled: no absorption at lam_lab in HD 49331 (M1 III)
    ("Ni", 6108.1159),
    ("Ti", 6126.2160),
]


# --- Profile and per-line helpers -----------------------------------------

@dataclass
class LineFit:
    element: str
    lambda_lab: float
    lambda_obs: float
    sigma: float
    depth: float
    continuum: float
    v_kms: float
    rms: float
    n_points: int
    ok: bool
    reason: str = ""


def _gaussian_absorption(
    lam: np.ndarray, mu: float, sigma: float, amp: float, cont: float
) -> np.ndarray:
    return cont - amp * np.exp(-0.5 * ((lam - mu) / sigma) ** 2)


def _wing_continuum(
    x: np.ndarray, y: np.ndarray, mu: float, core_half: float, percentile: float = 80.0
) -> float:
    """Estimate the continuum from points OUTSIDE the line core.

    Uses an upper percentile of the wing samples so a few low outliers in
    the wings cannot drag the level down toward the absorption.
    """
    wing = np.abs(x - mu) > core_half
    sample = y[wing] if wing.sum() >= 3 else y
    return float(np.percentile(sample, percentile))


# --- Per-line method ------------------------------------------------------

def fit_absorption_line(
    lam: np.ndarray,
    intensity: np.ndarray,
    lam_lab: float,
    half_window: float = 2.0,
    min_points: int = 7,
    sigma_min: float = 0.08,
    sigma_max: float = 0.40,
    core_half: float = 0.45,
    mu_search_half: float = 0.30,
    clip_kappa: float = 3.0,
) -> LineFit:
    """Fit an inverted Gaussian with robust continuum + bounded sigma + clip."""
    mask = (lam >= lam_lab - half_window) & (lam <= lam_lab + half_window)
    x = lam[mask]
    y = intensity[mask]

    if x.size < min_points:
        return LineFit(
            element="", lambda_lab=lam_lab, lambda_obs=np.nan, sigma=np.nan,
            depth=np.nan, continuum=np.nan, v_kms=np.nan, rms=np.nan,
            n_points=int(x.size), ok=False, reason="too few points in window",
        )

    cont0 = _wing_continuum(x, y, mu=lam_lab, core_half=core_half, percentile=80.0)

    # Seed mu at the deepest pixel near lam_lab, not lam_lab itself.
    near = np.abs(x - lam_lab) <= mu_search_half
    if near.any():
        mu0 = float(x[near][int(np.argmin(y[near]))])
    else:
        mu0 = float(lam_lab)

    depth0 = max(cont0 - float(np.min(y[near] if near.any() else y)), 1.0)
    sigma0 = max(sigma_min, min(0.15, sigma_max))

    p0 = [mu0, sigma0, depth0, cont0]
    bounds = (
        [lam_lab - mu_search_half, sigma_min, 0.0,           0.0],
        [lam_lab + mu_search_half, sigma_max, 5.0 * cont0,   5.0 * cont0 + 1.0],
    )

    def _do_fit(xx: np.ndarray, yy: np.ndarray):
        return curve_fit(_gaussian_absorption, xx, yy, p0=p0, bounds=bounds, maxfev=5000)

    try:
        popt, _ = _do_fit(x, y)
    except (RuntimeError, ValueError) as exc:
        return LineFit(
            element="", lambda_lab=lam_lab, lambda_obs=np.nan, sigma=np.nan,
            depth=np.nan, continuum=np.nan, v_kms=np.nan, rms=np.nan,
            n_points=int(x.size), ok=False, reason=f"fit failed: {exc}",
        )

    # One sigma-clip pass on the residual to kill cosmics / dropouts.
    resid = y - _gaussian_absorption(x, *popt)
    mad = np.median(np.abs(resid - np.median(resid)))
    if mad > 0:
        keep = np.abs(resid - np.median(resid)) <= clip_kappa * 1.4826 * mad
        if keep.sum() >= min_points and keep.sum() < x.size:
            try:
                popt, _ = _do_fit(x[keep], y[keep])
                resid = y[keep] - _gaussian_absorption(x[keep], *popt)
            except (RuntimeError, ValueError):
                pass

    mu, sigma, amp, cont = popt
    rms = float(np.sqrt(np.mean(resid ** 2)))
    v = C_KMS * (mu - lam_lab) / lam_lab

    return LineFit(
        element="", lambda_lab=lam_lab, lambda_obs=float(mu), sigma=float(sigma),
        depth=float(amp), continuum=float(cont), v_kms=float(v), rms=rms,
        n_points=int(x.size), ok=True,
    )


def measure_rv_night(
    star: pd.DataFrame,
    lines: list[tuple[str, float]] = STELLAR_LINES,
    half_window: float = 2.0,
    lambda_col: str = "lambda_rest",
) -> pd.DataFrame:
    """Fit every line independently."""
    if lambda_col not in star.columns:
        raise KeyError(
            f"Star spectrum has no '{lambda_col}' column; "
            f"available: {list(star.columns)}"
        )

    lam = star[lambda_col].to_numpy()
    intensity = star["intensidad"].to_numpy()

    rows: list[dict] = []
    for element, lam_lab in lines:
        fit = fit_absorption_line(lam, intensity, lam_lab, half_window=half_window)
        fit.element = element
        rows.append(fit.__dict__)

    return pd.DataFrame(rows)


# --- Joint method (shared sigma + shared v across all lines) --------------

def measure_rv_night_joint(
    star: pd.DataFrame,
    lines: list[tuple[str, float]] = STELLAR_LINES,
    half_window: float = 2.0,
    lambda_col: str = "lambda_rest",
    sigma_min: float = 0.08,
    sigma_max: float = 0.40,
    core_half: float = 0.45,
    v_init_kms: float = 0.0,
    v_bound_kms: float = 80.0,
) -> tuple[pd.DataFrame, float, float]:
    """Fit all lines simultaneously sharing instrumental sigma and v_kms.

    Free parameters:
      * ``v_kms``         (1, shared by all lines)
      * ``sigma``         (1, shared by all lines — instrumental width)
      * ``cont[i]``       (one per line, from wing percentile init)
      * ``depth[i]``      (one per line)

    Returns ``(per_line, v_kms, sigma)``. The per-line table mirrors the
    schema of :func:`measure_rv_night` so plotting / EW code is reused.
    """
    if lambda_col not in star.columns:
        raise KeyError(
            f"Star spectrum has no '{lambda_col}' column; "
            f"available: {list(star.columns)}"
        )

    lam_all = star[lambda_col].to_numpy()
    int_all = star["intensidad"].to_numpy()

    # Slice the spectrum once per line so we don't recompute masks.
    segments: list[dict] = []
    for element, lam_lab in lines:
        mask = (lam_all >= lam_lab - half_window) & (lam_all <= lam_lab + half_window)
        x = lam_all[mask]
        y = int_all[mask]
        if x.size < 5:
            segments.append({"element": element, "lam_lab": lam_lab,
                             "x": x, "y": y, "ok": False})
            continue
        cont0 = _wing_continuum(x, y, mu=lam_lab, core_half=core_half, percentile=80.0)
        depth0 = max(cont0 - float(np.min(y)), 1.0)
        segments.append({
            "element": element, "lam_lab": lam_lab,
            "x": x, "y": y, "cont0": cont0, "depth0": depth0, "ok": True,
        })

    good = [s for s in segments if s["ok"]]
    if len(good) < 2:
        # Degenerate; fall back to per-line and report sigma = nan.
        per_line = measure_rv_night(star, lines, half_window, lambda_col)
        return per_line, float("nan"), float("nan")

    n_lines = len(good)
    # theta = [v_kms, sigma, cont_0..cont_{n-1}, depth_0..depth_{n-1}]
    theta0 = np.empty(2 + 2 * n_lines)
    theta0[0] = v_init_kms
    theta0[1] = max(sigma_min, min(0.15, sigma_max))
    for i, s in enumerate(good):
        theta0[2 + i] = s["cont0"]
        theta0[2 + n_lines + i] = s["depth0"]

    lower = np.empty_like(theta0)
    upper = np.empty_like(theta0)
    lower[0], upper[0] = -v_bound_kms, v_bound_kms
    lower[1], upper[1] = sigma_min, sigma_max
    for i, s in enumerate(good):
        lower[2 + i] = 0.0
        upper[2 + i] = 5.0 * s["cont0"] + 1.0
        lower[2 + n_lines + i] = 0.0
        upper[2 + n_lines + i] = 5.0 * s["cont0"] + 1.0

    def residuals(theta: np.ndarray) -> np.ndarray:
        v_kms = theta[0]
        sigma = theta[1]
        conts = theta[2:2 + n_lines]
        depths = theta[2 + n_lines:]
        chunks: list[np.ndarray] = []
        for i, s in enumerate(good):
            mu_i = s["lam_lab"] * (1.0 + v_kms / C_KMS)
            model = _gaussian_absorption(s["x"], mu_i, sigma, depths[i], conts[i])
            chunks.append(s["y"] - model)
        return np.concatenate(chunks)

    try:
        res = least_squares(residuals, theta0, bounds=(lower, upper), max_nfev=10_000)
    except ValueError:
        per_line = measure_rv_night(star, lines, half_window, lambda_col)
        return per_line, float("nan"), float("nan")

    v_kms = float(res.x[0])
    sigma = float(res.x[1])
    conts = res.x[2:2 + n_lines]
    depths = res.x[2 + n_lines:]

    # Build a per-line table in the same schema as measure_rv_night.
    good_idx = 0
    rows: list[dict] = []
    for s in segments:
        lam_lab = s["lam_lab"]
        if not s["ok"]:
            rows.append({
                "element": s["element"], "lambda_lab": lam_lab,
                "lambda_obs": np.nan, "sigma": np.nan, "depth": np.nan,
                "continuum": np.nan, "v_kms": np.nan, "rms": np.nan,
                "n_points": int(s["x"].size), "ok": False,
                "reason": "too few points in window",
            })
            continue
        cont_i = float(conts[good_idx])
        depth_i = float(depths[good_idx])
        mu_i = lam_lab * (1.0 + v_kms / C_KMS)
        model = _gaussian_absorption(s["x"], mu_i, sigma, depth_i, cont_i)
        rms_i = float(np.sqrt(np.mean((s["y"] - model) ** 2)))
        rows.append({
            "element": s["element"], "lambda_lab": lam_lab,
            "lambda_obs": float(mu_i), "sigma": sigma, "depth": depth_i,
            "continuum": cont_i, "v_kms": v_kms, "rms": rms_i,
            "n_points": int(s["x"].size), "ok": True, "reason": "",
        })
        good_idx += 1

    return pd.DataFrame(rows), v_kms, sigma


# --- Aggregation -----------------------------------------------------------

@dataclass
class RVSummary:
    n_used: int
    v_mean_kms: float
    v_median_kms: float
    v_std_kms: float
    v_sem_kms: float


def summarize_rv(per_line: pd.DataFrame, mad_kappa: float = 3.0) -> RVSummary:
    """Aggregate per-line velocities with a MAD-based outlier rejection."""
    good = per_line[per_line["ok"]]
    v = good["v_kms"].to_numpy()
    if v.size == 0:
        return RVSummary(0, float("nan"), float("nan"), float("nan"), float("nan"))

    if v.size >= 4:
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        if mad > 0:
            keep = np.abs(v - med) <= mad_kappa * 1.4826 * mad
            if keep.sum() >= 2:
                v = v[keep]

    n = int(v.size)
    std = float(np.std(v, ddof=1)) if n > 1 else 0.0
    return RVSummary(
        n_used=n,
        v_mean_kms=float(np.mean(v)),
        v_median_kms=float(np.median(v)),
        v_std_kms=std,
        v_sem_kms=std / np.sqrt(n) if n > 1 else 0.0,
    )


# --- Cross-correlation method ---------------------------------------------

@dataclass
class CCFResult:
    v_axis: np.ndarray
    ccf: np.ndarray
    v_kms: float
    v_err_kms: float
    fwhm_kms: float
    contrast: float
    n_lines_used: int


def measure_rv_night_ccf(
    star: pd.DataFrame,
    lines: list[tuple[str, float]] = STELLAR_LINES,
    half_window: float = 2.0,
    lambda_col: str = "lambda_rest",
    v_min_kms: float = -80.0,
    v_max_kms: float = 80.0,
    v_step_kms: float = 0.25,
    core_half: float = 0.45,
    percentile: float = 80.0,
) -> CCFResult:
    """Cross-correlate the spectrum against a per-line line-by-line mask.

    For each line we estimate a local continuum from the wings, compute
    the absorption residual ``r(λ) = 1 − I/cont`` (positive in absorption),
    and sum ``r`` evaluated at the Doppler-shifted line positions over a
    grid of trial velocities. The peak of that sum is the night's v_kms;
    a parabolic fit to the three points around the peak gives sub-step
    precision and a curvature-based uncertainty.

    Coadds the signal of all lines coherently, so weak / absent lines
    don't bias the result — they just contribute noise.
    """
    lam_all = star[lambda_col].to_numpy()
    int_all = star["intensidad"].to_numpy()

    segments: list[tuple[float, np.ndarray, np.ndarray]] = []
    for _, lam_lab in lines:
        mask = (lam_all >= lam_lab - half_window) & (lam_all <= lam_lab + half_window)
        x = lam_all[mask]
        y = int_all[mask]
        if x.size < 5:
            continue
        cont = _wing_continuum(x, y, mu=lam_lab, core_half=core_half, percentile=percentile)
        if cont <= 0:
            continue
        r = 1.0 - y / cont
        segments.append((lam_lab, x, r))

    v_axis = np.arange(v_min_kms, v_max_kms + v_step_kms, v_step_kms)

    if not segments:
        return CCFResult(
            v_axis=v_axis, ccf=np.zeros_like(v_axis),
            v_kms=float("nan"), v_err_kms=float("nan"),
            fwhm_kms=float("nan"), contrast=float("nan"),
            n_lines_used=0,
        )

    ccf = np.zeros_like(v_axis)
    for j, v in enumerate(v_axis):
        s = 0.0
        shift = 1.0 + v / C_KMS
        for lam_lab, x, r in segments:
            s += float(np.interp(lam_lab * shift, x, r, left=0.0, right=0.0))
        ccf[j] = s

    i_max = int(np.argmax(ccf))
    v_peak = float(v_axis[i_max])
    v_err = float("nan")
    fwhm_kms = float("nan")

    if 1 <= i_max <= ccf.size - 2:
        y0, y1, y2 = float(ccf[i_max - 1]), float(ccf[i_max]), float(ccf[i_max + 1])
        denom = y0 - 2.0 * y1 + y2
        if denom < 0:  # concave down, true maximum
            offset = 0.5 * (y0 - y2) / denom
            v_peak = float(v_axis[i_max] + offset * v_step_kms)
            # Robust noise estimator from the CCF wings (|v - v_peak| > 30 km/s).
            far = np.abs(v_axis - v_peak) > 30.0
            if far.sum() >= 10:
                noise = 1.4826 * float(np.median(np.abs(ccf[far] - np.median(ccf[far]))))
            else:
                noise = float(np.std(ccf))
            curvature = -denom / (v_step_kms ** 2)  # > 0 at a maximum
            if curvature > 0 and noise > 0:
                v_err = float(noise / np.sqrt(curvature) / max(y1, 1e-12))
                # Approx Gaussian-like FWHM from curvature.
                fwhm_kms = float(2.0 * np.sqrt(2.0 * np.log(2.0)) * np.sqrt(y1 / curvature))

    baseline = float(np.median(ccf))
    contrast = float((ccf[i_max] - baseline) / baseline) if baseline > 0 else float("nan")

    return CCFResult(
        v_axis=v_axis, ccf=ccf,
        v_kms=v_peak, v_err_kms=v_err,
        fwhm_kms=fwhm_kms, contrast=contrast,
        n_lines_used=len(segments),
    )


def plot_rv_ccf(
    ccf: CCFResult, out_path: str | Path,
    title: str = "CCF estelar  ·  V / Ni / Ti",
) -> None:
    """Plot the cross-correlation function with its peak marked."""
    import matplotlib.pyplot as plt

    from . import style as _style  # noqa: F401
    from .style import LIGHT, PRIMARY, SECONDARY, style_axes, with_alpha

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(ccf.v_axis, ccf.ccf, color=PRIMARY, lw=2.0)
    ax.fill_between(ccf.v_axis, ccf.ccf, np.median(ccf.ccf),
                    where=ccf.ccf >= np.median(ccf.ccf),
                    color=with_alpha(SECONDARY, 0.25), linewidth=0)
    if np.isfinite(ccf.v_kms):
        ax.axvline(ccf.v_kms, color=PRIMARY, ls="--", lw=1.4,
                   label=f"v = {ccf.v_kms:+.2f} km/s")
    ax.axvline(0.0, color=LIGHT, ls=":", lw=1.0)
    ax.set_xlabel("v  (km/s)")
    ax.set_ylabel("CCF  (Σ residuos en absorción)")
    err_str = f"±{ccf.v_err_kms:.2f}" if np.isfinite(ccf.v_err_kms) else "—"
    fwhm_str = f"{ccf.fwhm_kms:.1f}" if np.isfinite(ccf.fwhm_kms) else "—"
    ax.set_title(
        f"{title}   ·   v = {ccf.v_kms:+.2f} {err_str} km/s   "
        f"FWHM = {fwhm_str} km/s   ·   {ccf.n_lines_used} líneas"
    )
    ax.legend(loc="best")
    style_axes(ax)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --- Diagnostic plot: spectrum context around each line -------------------

def plot_rv_diagnostic(
    star: pd.DataFrame,
    out_path: str | Path,
    lines: list[tuple[str, float]] = STELLAR_LINES_EXTENDED,
    context_half: float = 5.0,
    fit_half: float = 2.0,
    lambda_col: str = "lambda_rest",
    title: str = "Diagnóstico  ·  contexto de cada línea estelar",
) -> None:
    """One panel per line showing a *wider* spectral context.

    Marks the laboratory wavelength (solid), the fit window (shaded), and
    a robust local continuum (horizontal). Helps decide visually whether
    each line is actually present at the expected position.
    """
    import matplotlib.pyplot as plt

    from . import style as _style  # noqa: F401
    from .style import DARK, LIGHT, PRIMARY, SECONDARY, style_axes, with_alpha

    lam = star[lambda_col].to_numpy()
    intensity = star["intensidad"].to_numpy()

    n = len(lines)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (element, lam_lab) in zip(axes, lines):
        mask = (lam >= lam_lab - context_half) & (lam <= lam_lab + context_half)
        x = lam[mask]
        y = intensity[mask]
        ax.plot(x, y, color=DARK, lw=1.0, alpha=0.65)
        ax.scatter(x, y, s=14, color=DARK, alpha=0.85, edgecolor="none")

        ax.axvspan(lam_lab - fit_half, lam_lab + fit_half,
                   color=with_alpha(SECONDARY, 0.18), linewidth=0,
                   label="ventana de ajuste")
        ax.axvline(lam_lab, color=PRIMARY, ls="--", lw=1.6,
                   label=f"λ_lab = {lam_lab:.3f}")

        if x.size >= 5:
            cont = _wing_continuum(x, y, mu=lam_lab, core_half=0.45, percentile=80.0)
            ax.axhline(cont, color=LIGHT, ls=":", lw=1.4,
                       label=f"continuo ≈ {cont:,.0f}")

        ax.set_title(f"{element}  {lam_lab:.3f} Å", fontsize=11)
        ax.set_xlabel("λ (Å)", fontsize=10)
        ax.set_ylabel("I", fontsize=10)
        ax.legend(loc="best", fontsize=8, framealpha=0.85)
        style_axes(ax)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --- Per-line fit panel plot ----------------------------------------------

def plot_rv_fits(
    star: pd.DataFrame,
    per_line: pd.DataFrame,
    out_path: str | Path,
    half_window: float = 2.0,
    lambda_col: str = "lambda_rest",
    title: str = "Ajustes Doppler  ·  V / Ni / Ti",
) -> None:
    """One small panel per laboratory line with the fitted Gaussian on top."""
    import matplotlib.pyplot as plt

    from . import style as _style  # noqa: F401
    from .style import DARK, LIGHT, PRIMARY, SECONDARY, style_axes, with_alpha

    lam = star[lambda_col].to_numpy()
    intensity = star["intensidad"].to_numpy()

    n = len(per_line)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, per_line.iterrows()):
        lam_lab = row["lambda_lab"]
        mask = (lam >= lam_lab - half_window) & (lam <= lam_lab + half_window)
        ax.scatter(lam[mask], intensity[mask],
                   s=14, color=DARK, alpha=0.75, edgecolor="none")
        if row["ok"]:
            xx = np.linspace(lam_lab - half_window, lam_lab + half_window, 400)
            yy = _gaussian_absorption(xx, row["lambda_obs"], row["sigma"],
                                      row["depth"], row["continuum"])
            ax.plot(xx, yy, color=PRIMARY, lw=2.2)
            ax.fill_between(xx, yy, row["continuum"],
                            color=with_alpha(SECONDARY, 0.25), linewidth=0)
            ax.axvline(row["lambda_obs"], color=PRIMARY, ls="--", lw=1.0)
            ax.set_title(
                f"{row['element']} {lam_lab:.3f}   v = {row['v_kms']:+.2f} km/s",
                fontsize=11,
            )
        else:
            ax.set_title(f"{row['element']} {lam_lab:.3f}   (ajuste falló)",
                         fontsize=11, color=DARK)
        ax.axvline(lam_lab, color=LIGHT, ls=":", lw=1.2)
        ax.set_xlabel("λ (Å)", fontsize=10)
        ax.set_ylabel("I", fontsize=10)
        style_axes(ax)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def collect_rv_summaries(
    results_dir: str | Path, filename: str = "rv.json"
) -> pd.DataFrame:
    """Walk ``results/<date>/<filename>`` and return a per-night summary table."""
    import json

    rows = []
    for path in sorted(Path(results_dir).glob(f"*/{filename}")):
        try:
            rows.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def plot_rv_drift(
    results_dir: str | Path,
    out_path: str | Path,
    filename: str = "rv.json",
    title: str = "Velocidad radial estelar por noche  ·  V / Ni / Ti",
) -> pd.DataFrame:
    """Plot per-night mean radial velocity with SEM error bars."""
    import matplotlib.pyplot as plt

    from . import style as _style  # noqa: F401
    from .style import LIGHT, PRIMARY, SECONDARY, style_axes, with_alpha

    df = collect_rv_summaries(results_dir, filename=filename)
    if df.empty:
        raise RuntimeError(f"No {filename} files under {results_dir}.")

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.fill_between(
        df["date"],
        df["v_mean_kms"] - df["v_std_kms"],
        df["v_mean_kms"] + df["v_std_kms"],
        color=with_alpha(LIGHT, 0.55), linewidth=0,
        label="± 1σ (dispersión entre líneas)",
    )
    ax.errorbar(
        df["date"], df["v_mean_kms"], yerr=df["v_sem_kms"],
        fmt="o-", color=PRIMARY, ecolor=PRIMARY,
        markerfacecolor=SECONDARY, markeredgecolor=PRIMARY,
        markeredgewidth=1.4, markersize=8, linewidth=2.0, capsize=4,
        label="v̄ ± SEM",
    )
    ax.set_ylabel("v_rad  (km/s)")
    ax.set_xlabel("Fecha")
    ax.set_title(title)
    ax.legend(loc="best")
    style_axes(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return df
