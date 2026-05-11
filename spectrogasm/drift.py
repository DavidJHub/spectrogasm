"""Drift monitoring across calibrated nights.

After ``calibrate_all`` finishes, each ``results/<date>/solution.json``
holds the polynomial coefficients and final RMS for that night. This
module walks those files and produces:

  * a tidy ``DataFrame`` of (date, degree, rms, coef[0..3], telluric_kms,
    n_lines, seed_origin), and
  * a multi-panel plot of every coefficient and the RMS vs. date.

Sudden jumps in the curves flag real instrument events (focus changes,
grating moves); slow drifts justify per-night calibration.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import style as _style  # noqa: F401
from .style import PRIMARY, SECONDARY, style_axes


def collect_solutions(results_dir: str | Path) -> pd.DataFrame:
    """Load every ``solution.json`` under ``results_dir`` into one DataFrame."""
    rows = []
    for sol_path in sorted(Path(results_dir).glob("*/solution.json")):
        try:
            data = json.loads(sol_path.read_text())
        except json.JSONDecodeError:
            continue
        rows.append(data)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Align coefficients on the highest-degree row (low-order first).
    max_deg = max(len(c) for c in df["coef"])
    coef_matrix = np.array(
        [[0.0] * (max_deg - len(c)) + list(c) for c in df["coef"]]
    )
    for i in range(max_deg):
        df[f"c{max_deg - 1 - i}"] = coef_matrix[:, i]

    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def plot_drift(
    results_dir: str | Path,
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Plot polynomial coefficients and RMS vs date. Returns the underlying DataFrame.

    Pass ``out_path`` to write a PNG; otherwise the plot is shown interactively.
    """
    df = collect_solutions(results_dir)
    if df.empty:
        raise RuntimeError(f"No solution.json files found under {results_dir}.")
    if len(df) < 2:
        print(f"[drift] only {len(df)} night calibrated; nothing to plot yet.")
        return df

    coef_cols = sorted(
        (c for c in df.columns if c.startswith("c") and c[1:].isdigit()),
        key=lambda c: int(c[1:]),
        reverse=True,  # highest-order first, matches np.polyval convention
    )

    n_panels = len(coef_cols) + 1
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(12, 2.1 * n_panels), sharex=True
    )

    for ax, col in zip(axes[:-1], coef_cols):
        ax.plot(df["date"], df[col], "o-",
                color=PRIMARY, markerfacecolor=SECONDARY,
                markeredgecolor=PRIMARY, markeredgewidth=1.2)
        ax.set_ylabel(col)
        style_axes(ax)

    rms_ax = axes[-1]
    rms_ax.plot(df["date"], df["rms"], "o-",
                color=PRIMARY, markerfacecolor=SECONDARY,
                markeredgecolor=PRIMARY, markeredgewidth=1.2)
    rms_ax.set_ylabel("RMS (Å)")
    rms_ax.set_xlabel("Fecha")
    style_axes(rms_ax)

    fig.suptitle("Deriva de la solución de longitud de onda")
    fig.tight_layout()

    if out_path is None:
        plt.show()
    else:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    return df
