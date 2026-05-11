"""Peak detection and sub-pixel centroid refinement.

The centroid is refined with a Gaussian fit over +/- ``window`` pixels around
the integer-pixel peak. Compared to a 3-point parabola, this is far less
biased when the line is slightly asymmetric or when the true peak does not
fall close to the central pixel, at the cost of one curve_fit call per peak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def detect_peaks(
    intensity: np.ndarray,
    prominence: float = 2000.0,
    distance: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Find peaks in a 1-D intensity array. Returns (indices, prominences)."""
    idx, props = find_peaks(intensity, prominence=prominence, distance=distance)
    return idx, props["prominences"]


def _gauss(x: np.ndarray, A: float, mu: float, sigma: float, offset: float) -> np.ndarray:
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset


def refine_centroids(
    pixel: np.ndarray,
    intensity: np.ndarray,
    peak_idx: np.ndarray,
    prominences: np.ndarray,
    window: int = 3,
    max_sigma: float = 5.0,
) -> pd.DataFrame:
    """Refine each integer-pixel peak with a Gaussian fit on +/- ``window`` px.

    Peaks are discarded when:
      * the fitting window runs off the array,
      * ``curve_fit`` fails to converge,
      * the fitted centre drifts more than 1 px from the integer peak,
      * the fitted width is non-positive or larger than ``max_sigma`` px
        (suggests a blend or noise spike rather than a real line).
    """
    rows = []
    n = len(intensity)

    for p, prom in zip(peak_idx, prominences):
        lo = int(p) - window
        hi = int(p) + window + 1
        if lo < 0 or hi > n or (hi - lo) < 4:
            continue

        x = pixel[lo:hi].astype(float)
        y = intensity[lo:hi].astype(float)

        offset0 = float(y.min())
        A0 = float(y.max() - offset0)
        mu0 = float(p)
        sigma0 = 1.5

        try:
            popt, _ = curve_fit(
                _gauss, x, y, p0=[A0, mu0, sigma0, offset0], maxfev=500
            )
        except (RuntimeError, ValueError):
            continue

        A, mu, sigma, offset = popt
        sigma = abs(sigma)

        if abs(mu - p) > 1.0 or sigma <= 0 or sigma > max_sigma or A <= 0:
            continue

        rows.append(
            {
                "pixel_bruto": int(p),
                "pixel_centro": float(mu),
                "intensidad": float(A + offset),
                "prominencia": float(prom),
                "sigma_line": float(sigma),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("pixel_centro")
        .reset_index(drop=True)
    )
