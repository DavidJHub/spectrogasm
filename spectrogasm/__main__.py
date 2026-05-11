"""Batch entry point.

Usage:

    python -m spectrogasm                # run every night discovered in data/
    python -m spectrogasm 2024-03-12     # run only the listed dates

Nights are auto-discovered from ``data/`` by ``manifest.discover_nights``
(pattern ``estrella_YY-MM-DD.dat`` + ``thar_YY-MM-DD.dat``); the atlas
``data/thar_uves.dat`` is shared across all nights. Output goes to
``results/<ISO date>/``.
"""

from __future__ import annotations

import sys

from .drift import plot_drift
from .manifest import NIGHTS, RESULTS_DIR
from .pipeline import calibrate_all
from .radial_velocity import plot_rv_drift


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
            f"v_rad(per-line)={r.rv_summary.v_mean_kms:+.3f}"
            f"+/-{r.rv_summary.v_sem_kms:.3f}  "
            f"v_rad(joint)={r.rv_summary_joint.v_mean_kms:+.3f}  "
            f"seed={r.seed_origin}  -> {r.out_dir}"
        )

    if len(results) >= 2:
        drift_path = RESULTS_DIR / "drift.png"
        plot_drift(RESULTS_DIR, drift_path)
        print(f"\nDrift plot: {drift_path}")

        rv_path = RESULTS_DIR / "rv_drift.png"
        plot_rv_drift(RESULTS_DIR, rv_path)
        print(f"Radial-velocity plot (per-line): {rv_path}")

        rv_joint_path = RESULTS_DIR / "rv_drift_joint.png"
        plot_rv_drift(
            RESULTS_DIR, rv_joint_path,
            filename="rv_joint.json",
            title="Velocidad radial estelar por noche  ·  conjunto (σ compartido)",
        )
        print(f"Radial-velocity plot (joint):    {rv_joint_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
