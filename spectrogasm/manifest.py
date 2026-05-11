"""Inventory of observing nights, auto-discovered from ``data/``.

Naming convention for each night's spectra::

    data/estrella_YY-MM-DD.dat
    data/thar_YY-MM-DD.dat

Two-digit years are mapped to the 21st century (``YY`` -> ``20YY``). The
atlas ``data/thar_uves.dat`` is shared across all nights and is therefore
excluded from the per-night discovery.

A night gets a manual seed if a matching ``seeds/YY-MM-DD.json`` file
exists; otherwise the pipeline falls back to the fingerprint bootstrap
from the reference night.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SEED_DIR = ROOT / "seeds"
RESULTS_DIR = ROOT / "results"

ATLAS_PATH = DATA_DIR / "thar_uves.dat"
FINGERPRINT_PATH = ROOT / "fingerprint.json"

# The reference night supplies the manual seed and the fingerprint template
# every other night is bootstrapped from.
REFERENCE_DATE = "2024-03-12"

_DATE_RE = re.compile(r"^estrella_(\d{2}-\d{2}-\d{2})\.dat$")


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


# --- Target & observatory metadata ----------------------------------------
#
# All nights observe the same star from the same site, so these are global
# constants rather than per-night fields. Used by the barycentric correction
# in ``barycentric.py`` to convert the topocentric v_rad into heliocentric.

# HD 49331 (M1 III), SIMBAD ICRS J2000.
TARGET_NAME = "HD 49331"
TARGET_RA_DEG = 101.9055
TARGET_DEC_DEG = -16.8457

# Universidad de los Andes, Bogotá (UTC-5, no DST).
OBSERVATORY_NAME = "Universidad de los Andes, Bogotá"
OBSERVATORY_LAT_DEG = 4.602
OBSERVATORY_LON_DEG = -74.066
OBSERVATORY_ALT_M = 2625.0

# Approximate local hour at which exposures were taken. The exact DATE-OBS
# from the FITS headers is not in the dataset; using ~19:00 local
# introduces < 0.5 km/s of error in v_bary, well below the per-line scatter.
OBSERVATION_LOCAL_HOUR = 19


def _iso_from_short(short: str) -> str:
    """Convert ``YY-MM-DD`` to ``20YY-MM-DD``."""
    yy, mm, dd = short.split("-")
    return f"20{yy}-{mm}-{dd}"


def discover_nights(
    data_dir: Path = DATA_DIR,
    seed_dir: Path = SEED_DIR,
) -> list[Night]:
    """Scan ``data_dir`` for ``estrella_YY-MM-DD.dat`` / ``thar_YY-MM-DD.dat``
    pairs and return one :class:`Night` per matched date.

    A star file without a matching Th-Ar (or vice versa) is skipped silently;
    the atlas ``thar_uves.dat`` is excluded by the strict ``YY-MM-DD`` regex.
    """
    nights: list[Night] = []
    for star_path in sorted(data_dir.glob("estrella_*.dat")):
        match = _DATE_RE.match(star_path.name)
        if match is None:
            continue
        short = match.group(1)
        thar_path = data_dir / f"thar_{short}.dat"
        if not thar_path.exists():
            continue
        seed_path = seed_dir / f"{short}.json"
        nights.append(
            Night(
                date=_iso_from_short(short),
                star_file=star_path,
                thar_file=thar_path,
                seed_file=seed_path if seed_path.exists() else None,
            )
        )
    return nights


NIGHTS: list[Night] = discover_nights()


def _find_reference(nights: list[Night]) -> Night | None:
    for n in nights:
        if n.date == REFERENCE_DATE and n.seed_file is not None:
            return n
    return None


REFERENCE_NIGHT: Night | None = _find_reference(NIGHTS)
