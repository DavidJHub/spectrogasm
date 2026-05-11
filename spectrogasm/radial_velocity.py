"""Stellar radial velocity from Doppler shifts of absorption lines.

The calibrated stellar spectrum ``estrella_calibrada.csv`` carries a
``lambda_rest`` column whose telluric features have already been brought
to rest (topocentric frame). Any residual shift of a stellar absorption
line w.r.t. its laboratory wavelength is then the star's topocentric
radial velocity::

    v = c * (lambda_obs - lambda_lab) / lambda_lab

To get heliocentric velocity, ``spectrogasm.barycentric`` adds the
projection of Earth's velocity in the barycentric frame onto the line
of sight to the target.

For each laboratory line we fit an inverted Gaussian to a small window
of the spectrum (robust wing continuum, bounded sigma, one MAD-clip
pass on the residual), take the fitted centre as ``lambda_obs``, and
combine the per-line velocities with MAD outlier rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


C_KMS = 299_792.458


# Laboratory rest wavelengths (air, Å) for the stellar absorption lines
# used to measure the radial velocity.
#
# HD 49331 is an M1 III giant: its visible spectrum is dominated by
# molecular bands (TiO, CN) that wash out weak atomic features. The
# 2024-03-12 and 2024-03-19 rv_diagnostic plots show no absorption at
# lambda_lab for V 6039.7256, V 6081.4422 and V 6090.2084 (the deepest
# features in those windows are unrelated lines ~+1 A redward, which
# previously biased the estimator). Keep only the three lines that are
# reliably visible in this spectral type.
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


# --- Per-line fit ---------------------------------------------------------

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


# --- Across-night drift ---------------------------------------------------

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
    title: str = "Velocidad radial heliocéntrica por noche  ·  V / Ni / Ti",
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
    ax.set_ylabel("v_rad  (km/s, heliocéntrico)")
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
