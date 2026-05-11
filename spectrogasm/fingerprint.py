"""Lamp-fingerprint seeding.

Physical premise: a Th-Ar (or V/Ni/Ti comparison) lamp emits at fixed
laboratory wavelengths, and the *relative* intensities of strong lines are
nearly constant night-to-night. What drifts is only the pixel <-> wavelength
mapping. So once one night is calibrated by hand, the lamp's intensity
profile in pixel space serves as a reference fingerprint: any later night
needs only a global pixel shift to align with it, and the reference
(pixel, lambda) identifications can be carried over as a seed.

Saving the fingerprint:
    >>> save_fingerprint(thar_df, identified_lines, FINGERPRINT_PATH)

Bootstrapping a new night:
    >>> fp = load_fingerprint(FINGERPRINT_PATH)
    >>> seed_df = bootstrap_seed(new_thar_df, fp)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import correlate


@dataclass
class Fingerprint:
    """Reference lamp profile plus the identified (pixel, lambda) pairs."""

    pixel: np.ndarray
    intensity: np.ndarray
    lines: pd.DataFrame  # columns: pixel, lambda

    def save(self, path: str | Path) -> None:
        payload = {
            "pixel": self.pixel.tolist(),
            "intensity": self.intensity.tolist(),
            "lines": self.lines[["pixel", "lambda"]].to_dict(orient="records"),
        }
        Path(path).write_text(json.dumps(payload))


def save_fingerprint(
    thar: pd.DataFrame,
    identified_lines: pd.DataFrame,
    path: str | Path,
) -> Fingerprint:
    """Persist the reference lamp profile and its line identifications.

    ``identified_lines`` must contain ``pixel_centro`` and ``lambda_atlas``
    columns (the columns produced by the matching stage).
    """
    lines = pd.DataFrame(
        {
            "pixel": identified_lines["pixel_centro"].to_numpy(),
            "lambda": identified_lines["lambda_atlas"].to_numpy(),
        }
    )
    fp = Fingerprint(
        pixel=thar["pixel"].to_numpy(),
        intensity=thar["intensidad"].to_numpy(),
        lines=lines,
    )
    fp.save(path)
    return fp


def load_fingerprint(path: str | Path) -> Fingerprint:
    payload = json.loads(Path(path).read_text())
    return Fingerprint(
        pixel=np.asarray(payload["pixel"], dtype=float),
        intensity=np.asarray(payload["intensity"], dtype=float),
        lines=pd.DataFrame(payload["lines"]),
    )


def estimate_pixel_shift(reference: np.ndarray, new: np.ndarray) -> float:
    """Return the integer pixel shift (new - reference) via cross-correlation.

    Both inputs are 1-D intensity arrays of the same length sampled on the
    same integer pixel grid. The continuum is subtracted (mean) before
    correlating so the result is dominated by emission peaks.
    """
    if reference.shape != new.shape:
        raise ValueError(
            f"Fingerprint length {reference.shape} does not match the "
            f"new lamp length {new.shape}; trim or pad before fingerprinting."
        )
    a = reference - reference.mean()
    b = new - new.mean()
    corr = correlate(b, a, mode="full")
    lag = int(np.argmax(corr)) - (len(a) - 1)
    return float(lag)


def bootstrap_seed(thar: pd.DataFrame, fingerprint: Fingerprint) -> pd.DataFrame:
    """Generate a seed DataFrame for the new night from the reference fingerprint.

    The pixel shift is estimated by cross-correlation; identified lines from
    the reference are translated by that shift and returned in the same
    ``(pixel, lambda)`` schema that ``seeds.load_seed`` produces.
    """
    shift = estimate_pixel_shift(fingerprint.intensity, thar["intensidad"].to_numpy())
    print(f"[fingerprint] estimated pixel shift = {shift:+.1f}")
    seed = fingerprint.lines.copy()
    seed["pixel"] = seed["pixel"] + shift
    return seed
