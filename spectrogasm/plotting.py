"""Diagnostic plots for the calibration pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import style as _style  # noqa: F401  (applies rcParams on import)
from .style import LIGHT, PRIMARY, SECONDARY, style_axes, with_alpha


def _save_or_show(fig: plt.Figure, out_path: str | Path | None) -> None:
    if out_path is None:
        plt.show()
    else:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def plot_spectrum(
    spectrum: pd.DataFrame,
    title: str,
    out_path: str | Path | None = None,
    fill: bool = True,
) -> None:
    """Line plot of a calibrated spectrum.

    ``fill=True`` (default) adds a soft fill down to zero — good for
    emission spectra (Th-Ar lamp). For absorption spectra over a bright
    continuum (stellar), pass ``fill=False``.
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    if fill:
        ax.fill_between(
            spectrum["lambda_cal"],
            spectrum["intensidad"],
            color=with_alpha(LIGHT, 0.55),
            linewidth=0,
        )
    ax.plot(spectrum["lambda_cal"], spectrum["intensidad"],
            color=PRIMARY, linewidth=1.2)
    ax.set_xlabel("Longitud de onda (Å)")
    ax.set_ylabel("Intensidad")
    ax.set_title(title)
    style_axes(ax)
    fig.tight_layout()
    _save_or_show(fig, out_path)


def plot_lines_used(
    thar: pd.DataFrame,
    lines: pd.DataFrame,
    out_path: str | Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.plot(thar["lambda_cal"], thar["intensidad"],
            color=PRIMARY, linewidth=1.1)

    y_marks = np.interp(lines["lambda_atlas"], thar["lambda_cal"], thar["intensidad"])
    ax.scatter(
        lines["lambda_atlas"], y_marks,
        s=55, marker="o",
        facecolor=SECONDARY, edgecolor="white", linewidth=1.0, zorder=3,
    )

    for _, fila in lines.iterrows():
        y = np.interp(fila["lambda_atlas"], thar["lambda_cal"], thar["intensidad"])
        ax.annotate(
            f'{fila["lambda_atlas"]:.3f}',
            (fila["lambda_atlas"], y),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            color=PRIMARY,
            fontweight="semibold",
        )

    ax.set_xlabel("Longitud de onda (Å)")
    ax.set_ylabel("Intensidad")
    ax.set_title("Líneas usadas en la calibración")
    style_axes(ax)
    fig.tight_layout()
    _save_or_show(fig, out_path)
