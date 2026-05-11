"""Regression test for the March-12 reference night.

Skipped automatically if either the Th-Ar atlas or the night's .dat files
are missing, so the test suite remains green in environments without data.
"""

from __future__ import annotations

import pytest

from spectrogasm import CalibrationParams, REFERENCE_NIGHT, calibrate_night
from spectrogasm.manifest import ATLAS_PATH


def _data_present() -> bool:
    return (
        ATLAS_PATH.exists()
        and REFERENCE_NIGHT.star_file.exists()
        and REFERENCE_NIGHT.thar_file.exists()
        and REFERENCE_NIGHT.seed_file is not None
        and REFERENCE_NIGHT.seed_file.exists()
    )


@pytest.mark.skipif(not _data_present(), reason="data files not present")
def test_marzo12_calibration_rms():
    result = calibrate_night(
        REFERENCE_NIGHT, CalibrationParams(make_plots=False)
    )

    assert result.solution.degree in (2, 3)
    assert len(result.solution.lines) >= 10
    assert result.solution.rms < 0.05, (
        f"RMS regressed to {result.solution.rms:.4f} A "
        "(expected < 0.05 A for the reference night)."
    )
    assert result.seed_origin == "manual"
