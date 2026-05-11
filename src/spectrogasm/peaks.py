"""Peak detection and sub-pixel centroid refinement."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def detect_peaks(
    intensity: np.ndarray,
    prominence: float = 2000.0,
    distance: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Find peaks in a 1-D intensity array. Returns (indices, prominences)."""
    idx, props = find_peaks(intensity, prominence=prominence, distance=distance)
    return idx, props["prominences"]


def refine_centroids(
    pixel: np.ndarray,
    intensity: np.ndarray,
    peak_idx: np.ndarray,
    prominences: np.ndarray,
) -> pd.DataFrame:
    """Refine each integer-pixel peak with a 3-point parabolic fit.

    Peaks whose vertex falls more than 0.8 px away from the integer maximum
    (or whose quadratic coefficient is zero) are discarded as unreliable.
    """
    rows = []
    n = len(intensity)

    for p, prom in zip(peak_idx, prominences):
        if not (1 <= p < n - 1):
            continue

        x = np.array([p - 1, p, p + 1], dtype=float)
        y = intensity[p - 1 : p + 2].astype(float)

        a, b, c = np.polyfit(x, y, 2)
        if a == 0:
            continue

        x_centro = -b / (2 * a)
        y_centro = a * x_centro**2 + b * x_centro + c

        if (p - 0.8) <= x_centro <= (p + 0.8):
            rows.append(
                {
                    "pixel_bruto": int(p),
                    "pixel_centro": x_centro,
                    "intensidad": y_centro,
                    "prominencia": prom,
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values("pixel_centro")
        .reset_index(drop=True)
    )
