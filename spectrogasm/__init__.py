"""Wavelength calibration pipeline for echelle/grating spectroscopy."""

from . import style  # noqa: F401  — installs poster-grade matplotlib defaults
from .manifest import NIGHTS, REFERENCE_NIGHT, Night
from .pipeline import (
    CalibrationParams,
    CalibrationResult,
    calibrate_all,
    calibrate_night,
)
from .radial_velocity import (
    STELLAR_LINES,
    RVSummary,
    measure_rv_night,
    summarize_rv,
)

__all__ = [
    "NIGHTS",
    "REFERENCE_NIGHT",
    "Night",
    "CalibrationParams",
    "CalibrationResult",
    "calibrate_night",
    "calibrate_all",
    "STELLAR_LINES",
    "RVSummary",
    "measure_rv_night",
    "summarize_rv",
]
