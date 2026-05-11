"""Polynomial fit (degree 2 vs 3) with a single-pass outlier rejection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Solution:
    """Final wavelength solution and the data used to build it."""

    coef: np.ndarray
    degree: int
    rms: float
    lines: pd.DataFrame

    def apply(self, pixel: np.ndarray) -> np.ndarray:
        return np.polyval(self.coef, pixel)


def _fit(pix: np.ndarray, lam: np.ndarray, degree: int) -> tuple[np.ndarray, float]:
    coef = np.polyfit(pix, lam, degree)
    rms = float(np.sqrt(np.mean((np.polyval(coef, pix) - lam) ** 2)))
    return coef, rms


def choose_polynomial(
    lines: pd.DataFrame,
    prefer_high_degree_if: float = 0.8,
) -> tuple[np.ndarray, int, float]:
    """Fit degree 2 and 3 and pick the higher degree only if it cuts RMS hard
    enough (default: 20% better)."""
    pix = lines["pixel_centro"].to_numpy()
    lam = lines["lambda_atlas"].to_numpy()

    coef2, rms2 = _fit(pix, lam, 2)
    coef3, rms3 = _fit(pix, lam, 3)

    if rms3 < prefer_high_degree_if * rms2:
        return coef3, 3, rms3
    return coef2, 2, rms2


def reject_outliers(
    lines: pd.DataFrame,
    coef: np.ndarray,
    threshold: float = 0.08,
) -> pd.DataFrame:
    """Drop lines whose residual exceeds ``threshold`` Angstrom."""
    fit_lambda = np.polyval(coef, lines["pixel_centro"])
    residuals = fit_lambda - lines["lambda_atlas"]
    keep = lines[np.abs(residuals) < threshold].copy()
    keep["residual"] = residuals[np.abs(residuals) < threshold].to_numpy()
    return keep.reset_index(drop=True)


def fit_solution(
    matched: pd.DataFrame,
    outlier_threshold: float = 0.08,
) -> Solution:
    """Full fit: choose degree, drop outliers, refit, choose degree again."""
    coef0, _, _ = choose_polynomial(matched)
    good = reject_outliers(matched, coef0, threshold=outlier_threshold)

    if len(good) < 4:
        raise RuntimeError(
            f"Only {len(good)} lines survived the outlier cut; "
            "lower the threshold or revisit the seed."
        )

    coef, degree, rms = choose_polynomial(good)
    good["lambda_fit"] = np.polyval(coef, good["pixel_centro"])
    good["residual_final"] = good["lambda_fit"] - good["lambda_atlas"]

    return Solution(coef=coef, degree=degree, rms=rms, lines=good)
