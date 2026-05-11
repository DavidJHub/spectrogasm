"""Match refined peaks to atlas wavelengths using the seed polynomial."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import atlas_window


def match_to_atlas(
    centroids: pd.DataFrame,
    atlas: pd.Series,
    seed_coef: np.ndarray,
    tol: float = 0.20,
    pad: float = 1.0,
) -> pd.DataFrame:
    """For every refined peak, attach the nearest atlas line within ``tol`` A.

    ``pad`` extends the atlas window beyond the predicted lambda range so that
    peaks near the edge still find candidates.
    """
    centroids = centroids.copy()
    centroids["lambda_seed"] = np.polyval(seed_coef, centroids["pixel_centro"])

    lam_min = centroids["lambda_seed"].min() - pad
    lam_max = centroids["lambda_seed"].max() + pad
    region = atlas_window(atlas, lam_min, lam_max)

    if region.size == 0:
        raise RuntimeError(
            "No atlas lines fall inside the seed-predicted range "
            f"[{lam_min:.2f}, {lam_max:.2f}]. Check the seed file or the atlas."
        )

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

    matched = pd.DataFrame(rows)
    matched = matched[matched["absdiff"] < tol].copy()

    # If two peaks claim the same atlas line, keep the closest / strongest one.
    matched = matched.sort_values(
        ["lambda_atlas", "absdiff", "prominencia"],
        ascending=[True, True, False],
    ).drop_duplicates(subset="lambda_atlas", keep="first")

    return matched.reset_index(drop=True)


def select_strong_lines(
    matched: pd.DataFrame,
    n_lines: int = 20,
    min_separation: float = 8.0,
) -> pd.DataFrame:
    """Pick up to ``n_lines`` strong, well-separated calibration lines.

    Lines are ranked by prominence then by atlas distance, and any new line
    that sits within ``min_separation`` pixels of one already chosen is
    dropped.
    """
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

    return (
        pd.DataFrame(chosen)
        .sort_values("pixel_centro")
        .reset_index(drop=True)
    )
