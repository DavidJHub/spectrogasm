"""Stellar radial velocity from Doppler shifts of V, Ni, Ti absorption lines.

The calibrated stellar spectrum ``estrella_calibrada.csv`` carries a
``lambda_rest`` column whose telluric features have already been brought
to rest. Any residual shift of a stellar absorption line w.r.t. its
laboratory wavelength is then the star's radial velocity::

    v = c * (lambda_obs - lambda_lab) / lambda_lab

For each laboratory line we fit an inverted Gaussian to a small window
of the spectrum, take the fitted centre as ``lambda_obs``, and average
the per-line velocities to get the night's mean v_rad with a standard
error.
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
STELLAR_LINES: list[tuple[str, float]] = [
    ("V",  6039.7256),
    ("V",  6058.1420),
    ("V",  6081.4422),
    ("V",  6090.2084),
    ("Ni", 6108.1159),
    ("Ti", 6126.2160),
]


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


def fit_absorption_line(
    lam: np.ndarray,
    intensity: np.ndarray,
    lam_lab: float,
    half_window: float = 2.0,
    min_points: int = 7,
) -> LineFit:
    """Fit an inverted Gaussian around ``lam_lab`` and return centre + v_kms."""
    mask = (lam >= lam_lab - half_window) & (lam <= lam_lab + half_window)
    x = lam[mask]
    y = intensity[mask]

    if x.size < min_points:
        return LineFit(
            element="", lambda_lab=lam_lab, lambda_obs=np.nan, sigma=np.nan,
            depth=np.nan, continuum=np.nan, v_kms=np.nan, rms=np.nan,
            n_points=int(x.size), ok=False, reason="too few points in window",
        )

    cont0 = float(np.median(y))
    depth0 = max(cont0 - float(np.min(y)), 1.0)
    p0 = [lam_lab, 0.1, depth0, cont0]
    bounds = (
        [lam_lab - half_window, 0.02, 0.0,          0.0],
        [lam_lab + half_window, 1.00, 5.0 * cont0,  5.0 * cont0 + 1.0],
    )

    try:
        popt, _ = curve_fit(_gaussian_absorption, x, y, p0=p0, bounds=bounds, maxfev=5000)
    except (RuntimeError, ValueError) as exc:
        return LineFit(
            element="", lambda_lab=lam_lab, lambda_obs=np.nan, sigma=np.nan,
            depth=np.nan, continuum=np.nan, v_kms=np.nan, rms=np.nan,
            n_points=int(x.size), ok=False, reason=f"fit failed: {exc}",
        )

    mu, sigma, amp, cont = popt
    resid = y - _gaussian_absorption(x, *popt)
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
    """Fit every line in ``lines`` against the stellar spectrum."""
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


@dataclass
class RVSummary:
    n_used: int
    v_mean_kms: float
    v_median_kms: float
    v_std_kms: float
    v_sem_kms: float


def summarize_rv(per_line: pd.DataFrame) -> RVSummary:
    """Aggregate per-line velocities into mean / median / SEM."""
    good = per_line[per_line["ok"]]
    v = good["v_kms"].to_numpy()
    n = int(v.size)
    if n == 0:
        return RVSummary(0, float("nan"), float("nan"), float("nan"), float("nan"))
    std = float(np.std(v, ddof=1)) if n > 1 else 0.0
    return RVSummary(
        n_used=n,
        v_mean_kms=float(np.mean(v)),
        v_median_kms=float(np.median(v)),
        v_std_kms=std,
        v_sem_kms=std / np.sqrt(n) if n > 1 else 0.0,
    )


def plot_rv_fits(
    star: pd.DataFrame,
    per_line: pd.DataFrame,
    out_path: str | Path,
    half_window: float = 2.0,
    lambda_col: str = "lambda_rest",
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

    fig.suptitle("Ajustes Doppler  ·  V / Ni / Ti")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def collect_rv_summaries(results_dir: str | Path) -> pd.DataFrame:
    """Walk ``results/<date>/rv.json`` and return a per-night summary table."""
    import json

    rows = []
    for path in sorted(Path(results_dir).glob("*/rv.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def plot_rv_drift(results_dir: str | Path, out_path: str | Path) -> pd.DataFrame:
    """Plot per-night mean radial velocity with SEM error bars."""
    import matplotlib.pyplot as plt

    from . import style as _style  # noqa: F401
    from .style import LIGHT, PRIMARY, SECONDARY, style_axes, with_alpha

    df = collect_rv_summaries(results_dir)
    if df.empty:
        raise RuntimeError(f"No rv.json files under {results_dir}.")

    fig, ax = plt.subplots(figsize=(12, 4.6))
    # Std band (1-sigma scatter across the 6 lines) behind the SEM bars.
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
    ax.set_title("Velocidad radial estelar por noche  ·  V / Ni / Ti")
    ax.legend(loc="best")
    style_axes(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return df
