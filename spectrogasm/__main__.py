"""Batch entry point.

Usage:

    python -m spectrogasm                # run every night in manifest.NIGHTS
    python -m spectrogasm 2024-03-12     # run only the listed dates

All paths (data, seeds, atlas, results) are hardcoded in ``manifest.py``.
"""

from __future__ import annotations

import sys

from .manifest import NIGHTS
from .pipeline import calibrate_all


def main(argv: list[str]) -> None:
    if argv:
        wanted = set(argv)
        nights = [n for n in NIGHTS if n.date in wanted]
        missing = wanted - {n.date for n in nights}
        if missing:
            raise SystemExit(f"Unknown dates: {sorted(missing)}")
    else:
        nights = NIGHTS

    results = calibrate_all(nights)
    print(f"\nCalibrated {len(results)} / {len(nights)} nights.")
    for r in results:
        print(
            f"  {r.night.date}  deg={r.solution.degree}  "
            f"rms={r.solution.rms:.4f} A  v_tel={r.telluric_kms:+.3f} km/s  "
            f"seed={r.seed_origin}  -> {r.out_dir}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
