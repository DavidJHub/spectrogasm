"""File I/O: spectra, atlas, and result tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_spectrum(path: str | Path) -> pd.DataFrame:
    """Load a two-column (pixel, intensity) ascii spectrum."""
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["pixel", "intensidad"],
        engine="python",
    )


def load_atlas(
    path: str | Path,
    lam_min: float = 3000.0,
    lam_max: float = 10000.0,
) -> pd.Series:
    """Load a Th-Ar line-list atlas and return the wavelengths in [lam_min, lam_max].

    The atlas file may have several columns and use either '.' or ',' as decimal
    separator; every numeric token is collected, filtered, sorted and deduplicated.
    The wavelength window defaults to the full optical range so the same atlas
    can be reused for any observation date.
    """
    raw = pd.read_csv(path, sep=r"\s+", header=None, dtype=str, engine="python")

    vals = pd.to_numeric(
        raw.stack().str.replace(",", ".", regex=False),
        errors="coerce",
    ).dropna()

    lines = vals[(vals > lam_min) & (vals < lam_max)]
    return lines.sort_values().drop_duplicates().reset_index(drop=True)


def atlas_window(atlas: pd.Series, lam_min: float, lam_max: float) -> np.ndarray:
    """Restrict an already-loaded atlas to a narrow wavelength window."""
    region = atlas[(atlas >= lam_min) & (atlas <= lam_max)]
    return region.reset_index(drop=True).to_numpy()


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
