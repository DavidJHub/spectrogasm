"""Polynomial fit with iterative kappa-sigma outlier rejection.

Instead of a single hard residual cut, we refit while iteratively masking
lines whose residual exceeds ``kappa * sigma`` of the current residual
distribution. This converges to the natural scatter of the surviving
lines, so neither the cut value nor the line list has to be tuned by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Solution:
    """Final wavelength solution and the lines that survived clipping."""

    coef: np.ndarray
    degree: int
    rms: float
    lines: pd.DataFrame
    n_iter: int = 0

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
    """Fit degree 2 and 3; pick degree 3 only if it cuts RMS by 20% or more."""
    pix = lines["pixel_centro"].to_numpy()
    lam = lines["lambda_atlas"].to_numpy()

    coef2, rms2 = _fit(pix, lam, 2)
    coef3, rms3 = _fit(pix, lam, 3)

    if rms3 < prefer_high_degree_if * rms2:
        return coef3, 3, rms3
    return coef2, 2, rms2


def sigma_clip_polyfit(
    pix: np.ndarray,
    lam: np.ndarray,
    degree: int,
    kappa: float = 3.0,
    max_iter: int = 10,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Iterative kappa-sigma clipped polynomial fit.

    Returns the final coefficients, a boolean mask of surviving points and
    the number of iterations actually performed. Stops early when the mask
    stops changing or when the surviving set would fall below ``degree + 2``.
    """
    mask = np.ones(len(pix), dtype=bool)
    coef = np.polyfit(pix, lam, degree)
    iters = 0

    for iters in range(1, max_iter + 1):
        resid = np.polyval(coef, pix) - lam
        sigma = float(np.std(resid[mask]))
        if sigma == 0.0:
            break

        new_mask = np.abs(resid) < kappa * sigma
        if new_mask.sum() < degree + 2:
            break
        if np.array_equal(new_mask, mask):
            break

        mask = new_mask
        coef = np.polyfit(pix[mask], lam[mask], degree)

    return coef, mask, iters


def fit_solution(
    matched: pd.DataFrame,
    kappa: float = 3.0,
    max_iter: int = 10,
) -> Solution:
    """Choose the polynomial degree, then sigma-clip iteratively at that degree."""
    pix_all = matched["pixel_centro"].to_numpy()
    lam_all = matched["lambda_atlas"].to_numpy()

    _, degree, _ = choose_polynomial(matched)
    coef, mask, n_iter = sigma_clip_polyfit(pix_all, lam_all, degree, kappa, max_iter)

    if mask.sum() < degree + 2:
        raise RuntimeError(
            f"Only {int(mask.sum())} lines survived sigma-clipping at kappa={kappa}; "
            "loosen kappa, lower the seed match tolerance, or revisit the seed."
        )

    good = matched.loc[mask].copy().reset_index(drop=True)
    good["lambda_fit"] = np.polyval(coef, good["pixel_centro"])
    good["residual_final"] = good["lambda_fit"] - good["lambda_atlas"]
    rms = float(np.sqrt(np.mean(good["residual_final"] ** 2)))

    return Solution(coef=coef, degree=degree, rms=rms, lines=good, n_iter=n_iter)
