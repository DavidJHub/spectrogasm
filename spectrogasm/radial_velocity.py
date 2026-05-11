"""Stellar radial velocity from Doppler shifts of absorption lines.

The calibrated stellar spectrum ``estrella_calibrada.csv`` carries a
``lambda_rest`` column whose telluric features have already been brought
to rest (topocentric frame). Any residual shift of a stellar absorption
line w.r.t. its laboratory wavelength is then the star's topocentric
radial velocity::

    v = c * (lambda_obs - lambda_lab) / lambda_lab

The reported value is therefore the *topocentric* radial velocity: the
only frame transformation applied to the wavelength scale is the
telluric correction. A barycentric correction module exists in
``spectrogasm.barycentric`` for callers who want heliocentric values,
but the pipeline does not apply it by default.

**Estimator (heuristic, not a model fit).** For each laboratory line
we take the deepest pixel within ``±half_window`` of ``lambda_lab``
and refine its position with a parabolic interpolation through the
three points around the minimum. This is the spreadsheet-style "find
the dip, compute Doppler" approach: no Gaussian shape assumption, no
bounds on sigma or mu, no constraint that the line look like a clean
absorption. The per-line velocities are then combined with a MAD
outlier rejection.

Why heuristic over Gaussian: the previous fitter constrained ``mu``
to ``±0.30 A`` around the lab wavelength, which was too tight for
this dataset (the strongest features sit ~+0.7 A redward in
``lambda_rest`` for several lines), and the Gaussian shape biased
the centre toward whatever pixel had the lowest noise rather than
the actual line bottom.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


C_KMS = 299_792.458


# Laboratory rest wavelengths (air, Å, VALD-like values) for the stellar
# absorption lines used to measure the radial velocity. All six are in
# UVES echelle order 37 (5970-6180 Å).
STELLAR_LINES: list[tuple[str, float]] = [
    ("V",  6039.7256),
    ("V",  6058.1420),
    ("V",  6081.4422),
    ("V",  6090.2084),
    ("Ni", 6108.1159),
    ("Ti", 6126.2160),
]


# --- Helpers --------------------------------------------------------------

@dataclass
class LineFit:
    element: str
    lambda_lab: float
    lambda_obs: float
    sigma: float            # half-width estimate (Å), from FWHM/2.355
    depth: float            # continuum - intensity at minimum
    continuum: float
    v_kms: float
    rms: float              # not meaningful for the heuristic; left as NaN
    n_points: int
    ok: bool
    reason: str = ""


def _wing_continuum(
    x: np.ndarray, y: np.ndarray, mu: float, core_half: float, percentile: float = 80.0
) -> float:
    """Upper-percentile of the wing samples (outside ±core_half of ``mu``)."""
    wing = np.abs(x - mu) > core_half
    sample = y[wing] if wing.sum() >= 3 else y
    return float(np.percentile(sample, percentile))


def _parabolic_vertex(
    x_left: float, x_mid: float, x_right: float,
    y_left: float, y_mid: float, y_right: float,
) -> float:
    """Vertex of the parabola through three (x, y) points. Falls back
    to ``x_mid`` if the points are not concave-up."""
    denom = y_left - 2.0 * y_mid + y_right
    if denom <= 0:
        return x_mid
    # Standard 3-point parabolic interpolation, written in absolute x so
    # it tolerates non-uniform sampling (the calibrated grid is close to
    # uniform but not exactly).
    dx_left = x_mid - x_left
    dx_right = x_right - x_mid
    dx = 0.5 * (dx_left + dx_right)
    offset = 0.5 * (y_left - y_right) / denom
    return x_mid + offset * dx


def _half_width_at_half_min(
    x: np.ndarray, y: np.ndarray, i_min: int, continuum: float
) -> float:
    """Rough HWHM of the absorption profile (Å), used only for the plot."""
    depth = continuum - y[i_min]
    if depth <= 0:
        return float("nan")
    half_level = continuum - 0.5 * depth
    # Walk left/right from the minimum until we cross half_level.
    left = i_min
    while left > 0 and y[left] < half_level:
        left -= 1
    right = i_min
    while right < x.size - 1 and y[right] < half_level:
        right += 1
    return 0.5 * (x[right] - x[left])


# --- Per-line measurement -------------------------------------------------

def measure_line_shift(
    lam: np.ndarray,
    intensity: np.ndarray,
    lam_lab: float,
    half_window: float = 2.0,
    min_points: int = 5,
) -> LineFit:
    """Heuristic line-centre finder + Doppler velocity.

    1. Take the deepest pixel within ``±half_window`` of ``lam_lab``.
    2. Refine sub-pixel with a parabola through the deepest pixel and
       its two neighbours.
    3. Compute ``v = c (lambda_centre - lambda_lab) / lambda_lab``.

    No bounds on the search window beyond ``half_window`` — if the line
    really sits +0.7 A redward of the lab value because the spectrum's
    rest frame is mis-aligned, we still find it.
    """
    mask = (lam >= lam_lab - half_window) & (lam <= lam_lab + half_window)
    x = lam[mask]
    y = intensity[mask]

    if x.size < min_points:
        return LineFit(
            element="", lambda_lab=lam_lab, lambda_obs=np.nan, sigma=np.nan,
            depth=np.nan, continuum=np.nan, v_kms=np.nan, rms=np.nan,
            n_points=int(x.size), ok=False, reason="too few points in window",
        )

    i_min = int(np.argmin(y))
    if 0 < i_min < x.size - 1:
        lambda_obs = _parabolic_vertex(
            x[i_min - 1], x[i_min], x[i_min + 1],
            float(y[i_min - 1]), float(y[i_min]), float(y[i_min + 1]),
        )
    else:
        lambda_obs = float(x[i_min])

    continuum = _wing_continuum(x, y, mu=lam_lab, core_half=0.45, percentile=80.0)
    depth = float(max(continuum - float(y[i_min]), 0.0))
    hwhm = _half_width_at_half_min(x, y, i_min, continuum)
    sigma = hwhm / np.sqrt(2.0 * np.log(2.0)) if hwhm == hwhm else float("nan")
    v = C_KMS * (lambda_obs - lam_lab) / lam_lab

    return LineFit(
        element="", lambda_lab=lam_lab, lambda_obs=float(lambda_obs),
        sigma=float(sigma), depth=depth, continuum=float(continuum),
        v_kms=float(v), rms=float("nan"),
        n_points=int(x.size), ok=True,
    )


def measure_rv_night(
    star: pd.DataFrame,
    lines: list[tuple[str, float]] = STELLAR_LINES,
    half_window: float = 2.0,
    lambda_col: str = "lambda_rest",
) -> pd.DataFrame:
    """Heuristic shift for every line in ``lines``."""
    if lambda_col not in star.columns:
        raise KeyError(
            f"Star spectrum has no '{lambda_col}' column; "
            f"available: {list(star.columns)}"
        )

    lam = star[lambda_col].to_numpy()
    intensity = star["intensidad"].to_numpy()

    rows: list[dict] = []
    for element, lam_lab in lines:
        fit = measure_line_shift(lam, intensity, lam_lab, half_window=half_window)
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
    lines: list[tuple[str, float]] = STELLAR_LINES,
    context_half: float = 5.0,
    fit_half: float = 2.0,
    lambda_col: str = "lambda_rest",
    title: str = "Diagnóstico  ·  contexto de cada línea estelar",
) -> None:
    """One panel per line showing a *wider* spectral context."""
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
                   label="ventana de búsqueda")
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


# --- Per-line measurement plot --------------------------------------------

def plot_rv_fits(
    star: pd.DataFrame,
    per_line: pd.DataFrame,
    out_path: str | Path,
    half_window: float = 2.0,
    lambda_col: str = "lambda_rest",
    title: str = "Corrimientos Doppler  ·  V / Ni / Ti",
) -> None:
    """One panel per line: data points, λ_lab marker, measured centre."""
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
        x = lam[mask]
        y = intensity[mask]
        ax.plot(x, y, color=DARK, lw=1.2, alpha=0.7)
        ax.scatter(x, y, s=22, color=DARK, alpha=0.85, edgecolor="none")
        if row["ok"]:
            if np.isfinite(row["continuum"]):
                ax.axhline(row["continuum"], color=LIGHT, ls=":", lw=1.0)
            ax.axvline(row["lambda_obs"], color=PRIMARY, ls="-", lw=1.6,
                       label=f"λ_obs = {row['lambda_obs']:.3f}")
            shift = row["lambda_obs"] - lam_lab
            ax.set_title(
                f"{row['element']} {lam_lab:.3f}   "
                f"Δλ = {shift:+.3f} Å   v = {row['v_kms']:+.2f} km/s",
                fontsize=10,
            )
        else:
            ax.set_title(f"{row['element']} {lam_lab:.3f}   (sin datos)",
                         fontsize=10, color=DARK)
        ax.axvline(lam_lab, color=SECONDARY, ls="--", lw=1.2,
                   label=f"λ_lab = {lam_lab:.3f}")
        ax.set_xlabel("λ (Å)", fontsize=10)
        ax.set_ylabel("I", fontsize=10)
        ax.legend(loc="best", fontsize=7, framealpha=0.85)
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
    title: str = "Velocidad radial por noche  ·  V / Ni / Ti",
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
