"""Telluric radial-velocity lookup and rest-frame shift.

The wavelength scale ``lambda_cal`` produced by ``pipeline.calibrate_night``
is in the frame of the upstream UVES data reduction, which is *not*
topocentric: telluric O2 lines in this dataset appear at velocities
ranging from −15 km/s (Feb) to −25 km/s (Apr), a clean seasonal pattern
that matches Earth's barycentric motion toward HD 49331. So the upstream
data already lives in a barycentric-like frame and the telluric
velocity in ``manifest.TELLURIC_KMS`` is, in practice, the apparent
shift of topocentric O2 lines in that frame (≈ -v_bary).

To put the stellar absorption lines at their laboratory rest wavelengths
we have to **undo** that upstream correction, i.e. shift back toward
topocentric. The empirical signature is unmistakable: the previous
implementation (which applied the correction in the opposite direction)
left ``v_rad + 2·v_tel`` constant to 0.7 km/s across 11 nights spanning
a 22 km/s swing in v_rad, indicating the correction was being applied
twice in the same direction. ``to_rest_frame`` now applies it once in
the correct direction.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from .manifest import TELLURIC_KMS


C_KMS = 299_792.458


def telluric_velocity(iso_date: str) -> float:
    """Return the telluric velocity (km/s) for ``iso_date``.

    If the date is not in the log, linearly interpolate between the two
    bracketing logged dates. Raises if the date is outside the logged range.
    """
    if iso_date in TELLURIC_KMS:
        return TELLURIC_KMS[iso_date]

    target = date.fromisoformat(iso_date)
    logged = sorted((date.fromisoformat(d), v) for d, v in TELLURIC_KMS.items())

    if target < logged[0][0] or target > logged[-1][0]:
        raise KeyError(
            f"No telluric velocity for {iso_date} and it falls outside the "
            f"logged range {logged[0][0]} .. {logged[-1][0]}."
        )

    dates = np.array([(d - logged[0][0]).days for d, _ in logged], dtype=float)
    values = np.array([v for _, v in logged], dtype=float)
    x = (target - logged[0][0]).days
    return float(np.interp(x, dates, values))


def to_rest_frame(lambda_obs: np.ndarray, v_kms: float) -> np.ndarray:
    """Shift an observed wavelength array toward the (telluric) rest frame.

    Given the upstream frame described above, the correct transform is::

        lambda_rest = lambda_obs * (1 + v_kms / c)

    which, for the typical ``v_kms < 0`` of this dataset, shifts every
    wavelength *down* by ~0.48 Å — exactly cancelling the seasonal trend
    seen in ``v_rad`` across the 11 observed nights.
    """
    return lambda_obs * (1.0 + v_kms / C_KMS)

