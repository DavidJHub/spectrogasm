"""CLI entry point for the spectrogasm wavelength-calibration pipeline.

Examples
--------

Run the supplied March-12 night with the bundled seed file::

    python run_calibration.py \\
        --star  estrella_marzo12.dat \\
        --thar  thor_marzo12.dat \\
        --atlas thar_uves.dat \\
        --seed  seeds/marzo12.json \\
        --out   results/marzo12

A new night is added by dropping its ``.dat`` files anywhere and creating a
``seeds/<date>.json`` with 4-6 manual identifications. The atlas file does
not change between nights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src/`` importable without requiring an install.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spectrogasm.pipeline import CalibrationConfig, calibrate_night  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Th-Ar wavelength calibration pipeline.")
    p.add_argument("--star", required=True, help="Star spectrum (.dat).")
    p.add_argument("--thar", required=True, help="Th-Ar lamp spectrum (.dat).")
    p.add_argument("--atlas", required=True, help="Th-Ar atlas line list.")
    p.add_argument("--seed", required=True, help="Seed JSON for this night.")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--atlas-lam-min", type=float, default=3000.0)
    p.add_argument("--atlas-lam-max", type=float, default=10000.0)
    p.add_argument("--prominence", type=float, default=2000.0)
    p.add_argument("--distance", type=int, default=4)
    p.add_argument("--match-tol", type=float, default=0.20)
    p.add_argument("--n-lines", type=int, default=20)
    p.add_argument("--min-separation", type=float, default=8.0)
    p.add_argument("--outlier-threshold", type=float, default=0.08)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = CalibrationConfig(
        star_path=args.star,
        thar_path=args.thar,
        atlas_path=args.atlas,
        seed_path=args.seed,
        out_dir=args.out,
        atlas_lam_min=args.atlas_lam_min,
        atlas_lam_max=args.atlas_lam_max,
        peak_prominence=args.prominence,
        peak_distance=args.distance,
        match_tol=args.match_tol,
        n_lines=args.n_lines,
        min_line_separation=args.min_separation,
        outlier_threshold=args.outlier_threshold,
        make_plots=not args.no_plots,
    )
    calibrate_night(cfg)


if __name__ == "__main__":
    main()
