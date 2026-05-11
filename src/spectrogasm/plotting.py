"""Diagnostic plots for the calibration pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_or_show(fig: plt.Figure, out_path: str | Path | None) -> None:
    if out_path is None:
        plt.show()
    else:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def plot_spectrum(
    spectrum: pd.DataFrame,
    title: str,
    out_path: str | Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(spectrum["lambda_cal"], spectrum["intensidad"])
    ax.set_xlabel("Longitud de onda (A)")
    ax.set_ylabel("Intensidad")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, out_path)


def plot_lines_used(
    thar: pd.DataFrame,
    lines: pd.DataFrame,
    out_path: str | Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(thar["lambda_cal"], thar["intensidad"])

    y_marks = np.interp(lines["lambda_atlas"], thar["lambda_cal"], thar["intensidad"])
    ax.plot(lines["lambda_atlas"], y_marks, "x")

    for _, fila in lines.iterrows():
        y = np.interp(fila["lambda_atlas"], thar["lambda_cal"], thar["intensidad"])
        ax.annotate(
            f'{fila["lambda_atlas"]:.3f}',
            (fila["lambda_atlas"], y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=7,
        )

    ax.set_xlabel("Longitud de onda (A)")
    ax.set_ylabel("Intensidad")
    ax.set_title("Lineas usadas en la calibracion")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, out_path)
