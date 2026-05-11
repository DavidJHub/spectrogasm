"""Telluric radial-velocity lookup and rest-frame shift.

Telluric absorption lines should sit at known laboratory wavelengths; any
non-zero radial velocity measured from them is Earth's motion projected
onto the line of sight. We undo it on the calibrated spectrum so stellar
features land at rest-frame wavelengths.
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
    """Shift an observed wavelength array to the rest frame.

    If telluric lines appear at v_kms (negative = blueshift), the observed
    spectrum is multiplied by 1/(1 + v/c) to put those lines back at rest.
    """
    return lambda_obs / (1.0 + v_kms / C_KMS)
