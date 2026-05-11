"""Match refined peaks to atlas wavelengths using the seed polynomial."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io import atlas_window


@dataclass
class MatchReport:
    """Diagnostic counts emitted by ``match_to_atlas`` so callers can show
    *why* a match failed without poking at intermediate DataFrames."""

    lam_min: float
    lam_max: float
    n_atlas_in_window: int
    n_centroids: int
    n_within_tol: int
    nearest_absdiff: float  # smallest |diff| across all centroids (any tol)


def match_to_atlas(
    centroids: pd.DataFrame,
    atlas: pd.Series,
    seed_coef: np.ndarray,
    tol: float = 0.20,
    pad: float = 1.0,
) -> tuple[pd.DataFrame, MatchReport]:
    """For every refined peak, attach the nearest atlas line within ``tol`` A.

    ``pad`` extends the atlas window beyond the predicted lambda range so that
    peaks near the edge still find candidates. Returns the matched DataFrame
    and a ``MatchReport`` with the counts needed to debug an empty match.
    """
    centroids = centroids.copy()
    centroids["lambda_seed"] = np.polyval(seed_coef, centroids["pixel_centro"])

    lam_min = float(centroids["lambda_seed"].min() - pad)
    lam_max = float(centroids["lambda_seed"].max() + pad)
    region = atlas_window(atlas, lam_min, lam_max)

    if region.size == 0:
        report = MatchReport(lam_min, lam_max, 0, len(centroids), 0, float("nan"))
        return centroids.iloc[0:0].assign(
            lambda_atlas=np.nan, diff_seed=np.nan, absdiff=np.nan
        ), report

    rows = []
    for _, fila in centroids.iterrows():
        lam_est = fila["lambda_seed"]
        idx = int(np.argmin(np.abs(region - lam_est)))
        lam_atlas = float(region[idx])
        diff = lam_atlas - lam_est

        rows.append(
            {
                "pixel_centro": fila["pixel_centro"],
                "intensidad": fila["intensidad"],
                "prominencia": fila["prominencia"],
                "lambda_seed": lam_est,
                "lambda_atlas": lam_atlas,
                "diff_seed": diff,
                "absdiff": abs(diff),
            }
        )

    all_candidates = pd.DataFrame(rows)
    nearest = float(all_candidates["absdiff"].min())

    matched = all_candidates[all_candidates["absdiff"] < tol].copy()
    matched = matched.sort_values(
        ["lambda_atlas", "absdiff", "prominencia"],
        ascending=[True, True, False],
    ).drop_duplicates(subset="lambda_atlas", keep="first")
    matched = matched.reset_index(drop=True)

    report = MatchReport(
        lam_min=lam_min,
        lam_max=lam_max,
        n_atlas_in_window=int(region.size),
        n_centroids=int(len(centroids)),
        n_within_tol=int(len(matched)),
        nearest_absdiff=nearest,
    )
    return matched, report


def select_strong_lines(
    matched: pd.DataFrame,
    n_lines: int = 20,
    min_separation: float = 8.0,
) -> pd.DataFrame:
    """Pick up to ``n_lines`` strong, well-separated calibration lines.

    Lines are ranked by prominence then by atlas distance; any candidate
    within ``min_separation`` pixels of an already-chosen one is skipped.
    Returns an empty same-schema DataFrame if ``matched`` is empty.
    """
    if matched.empty:
        return matched.iloc[0:0].copy()

    chosen: list[pd.Series] = []

    for _, fila in matched.sort_values(
        ["prominencia", "absdiff"],
        ascending=[False, True],
    ).iterrows():
        if all(abs(fila["pixel_centro"] - s["pixel_centro"]) > min_separation
               for s in chosen):
            chosen.append(fila)
        if len(chosen) == n_lines:
            break

    if not chosen:
        return matched.iloc[0:0].copy()

    return (
        pd.DataFrame(chosen)
        .sort_values("pixel_centro")
        .reset_index(drop=True)
    )
