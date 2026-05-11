"""Hardcoded inventory of observing nights and per-night metadata.

Adding a new night is just a matter of appending a ``Night`` entry to
``NIGHTS`` and dropping its ``.dat`` files in ``data/``. If a manual seed
file does not exist for the new night, the pipeline falls back to the
fingerprint-based seed bootstrapped from ``REFERENCE_NIGHT``.

Telluric radial velocities (km/s) come from the night log; they shift the
calibrated wavelengths back to the rest frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SEED_DIR = ROOT / "seeds"
RESULTS_DIR = ROOT / "results"

ATLAS_PATH = DATA_DIR / "thar_uves.dat"
FINGERPRINT_PATH = ROOT / "fingerprint.json"


@dataclass(frozen=True)
class Night:
    date: str               # ISO date, used for the result folder name
    star_file: Path
    thar_file: Path
    seed_file: Path | None  # None -> use the fingerprint bootstrap


# Telluric radial velocities measured from the science spectra. Values are
# transcribed from the observing log; gaps are filled by linear interpolation
# inside ``telluric.telluric_velocity``.
TELLURIC_KMS: dict[str, float] = {
    "2024-02-09": -14.908,
    "2024-02-10": -15.280,
    "2024-02-11": -16.169,
    "2024-02-18": -18.423,
    "2024-02-21": -19.016,
    "2024-03-02": -21.343,
    "2024-03-11": -23.326,
    "2024-03-13": -24.022,
    "2024-03-19": -24.545,
    "2024-03-26": -25.259,
    "2024-04-06": -25.370,
    "2024-04-08": -25.583,
    "2024-04-09": -25.336,
}


# The reference night is the one whose seed lines were measured by hand and
# whose lamp spectrum becomes the fingerprint template for every other night.
REFERENCE_NIGHT = Night(
    date="2024-03-12",
    star_file=DATA_DIR / "estrella_marzo12.dat",
    thar_file=DATA_DIR / "thor_marzo12.dat",
    seed_file=SEED_DIR / "marzo12.json",
)


# Full batch. New nights without a hand-built seed get ``seed_file=None`` and
# the pipeline will bootstrap them from the fingerprint.
NIGHTS: list[Night] = [
    REFERENCE_NIGHT,
    # Example of a future night (no data on disk yet, here only as a template):
    # Night(
    #     date="2024-03-19",
    #     star_file=DATA_DIR / "estrella_marzo19.dat",
    #     thar_file=DATA_DIR / "thor_marzo19.dat",
    #     seed_file=None,
    # ),
]
