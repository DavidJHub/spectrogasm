"""Per-night seed identifications: a few (pixel, lambda) anchors used to
build a coarse pixel -> wavelength polynomial for the matching stage.

Seeds live as JSON files keyed by observation date so adding a new night is
just dropping a new file in ``seeds/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_seed(path: str | Path) -> pd.DataFrame:
    """Load a seed file. Format::

        {
          "date": "2024-03-12",
          "lines": [
            {"pixel": 71.685,  "lambda": 5973.664},
            {"pixel": 145.290, "lambda": 5994.128},
            ...
          ]
        }
    """
    data = json.loads(Path(path).read_text())
    return pd.DataFrame(data["lines"])


def seed_polynomial(seed: pd.DataFrame, degree: int = 2) -> np.ndarray:
    """Fit an initial polynomial pixel -> lambda from the manual seed lines."""
    if len(seed) <= degree:
        raise ValueError(
            f"Need more than {degree} seed lines to fit a degree-{degree} polynomial, "
            f"got {len(seed)}."
        )
    return np.polyfit(seed["pixel"], seed["lambda"], degree)
