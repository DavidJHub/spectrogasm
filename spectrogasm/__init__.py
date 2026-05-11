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
    CCFResult,
    RVSummary,
    measure_rv_night,
    measure_rv_night_ccf,
    measure_rv_night_joint,
    summarize_rv,
)
from .science import (
    ResolutionSummary,
    equivalent_widths,
    estimate_snr,
    per_element_rv,
    spectral_resolution,
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
    "CCFResult",
    "RVSummary",
    "measure_rv_night",
    "measure_rv_night_ccf",
    "measure_rv_night_joint",
    "summarize_rv",
    "ResolutionSummary",
    "equivalent_widths",
    "estimate_snr",
    "per_element_rv",
    "spectral_resolution",
]
