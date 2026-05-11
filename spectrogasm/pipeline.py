"""End-to-end orchestration of the wavelength calibration pipeline.

One ``Night`` from ``manifest.NIGHTS`` -> one ``CalibrationResult``. All
tunables live in ``CalibrationParams`` with sensible defaults; the manifest
keeps paths, dates and seeds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import fingerprint as fp_mod
from . import fitting, io, matching, peaks, plotting, seeds, telluric
from .manifest import ATLAS_PATH, FINGERPRINT_PATH, RESULTS_DIR, Night


@dataclass
class CalibrationParams:
    atlas_lam_min: float = 3000.0
    atlas_lam_max: float = 10000.0
    peak_prominence: float = 2000.0
    peak_distance: int = 4
    match_tol: float = 0.20
    n_lines: int = 20
    min_line_separation: float = 8.0
    sigma_clip_kappa: float = 3.0
    sigma_clip_max_iter: int = 10
    make_plots: bool = True


@dataclass
class CalibrationResult:
    night: Night
    star: pd.DataFrame
    thar: pd.DataFrame
    solution: fitting.Solution
    telluric_kms: float
    seed: pd.DataFrame
    seed_origin: str  # "manual" or "fingerprint"
    matched: pd.DataFrame
    out_dir: Path


def calibrate_night(night: Night, params: CalibrationParams | None = None) -> CalibrationResult:
    params = params or CalibrationParams()
    out_dir = RESULTS_DIR / night.date
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load spectra and atlas.
    star = io.load_spectrum(night.star_file)
    thar = io.load_spectrum(night.thar_file)
    atlas = io.load_atlas(ATLAS_PATH, params.atlas_lam_min, params.atlas_lam_max)
    print(f"[{night.date}] atlas lines: {len(atlas)}")

    # 2) Detect and refine lamp peaks.
    idx, prom = peaks.detect_peaks(
        thar["intensidad"].to_numpy(),
        prominence=params.peak_prominence,
        distance=params.peak_distance,
    )
    centroids = peaks.refine_centroids(
        thar["pixel"].to_numpy(),
        thar["intensidad"].to_numpy(),
        idx,
        prom,
    )
    print(f"[{night.date}] peaks: {len(idx)}  refined: {len(centroids)}")

    # 3) Seed: manual JSON if available, otherwise fingerprint bootstrap.
    seed_df, seed_origin = _resolve_seed(night, thar)
    seed_coef = seeds.seed_polynomial(seed_df, degree=2)
    print(f"[{night.date}] seed: {seed_origin} ({len(seed_df)} lines)")

    # 4) Match every peak to the atlas using the seed.
    matched = matching.match_to_atlas(
        centroids, atlas, seed_coef, tol=params.match_tol
    )
    selected = matching.select_strong_lines(
        matched,
        n_lines=params.n_lines,
        min_separation=params.min_line_separation,
    )
    print(f"[{night.date}] matched: {len(matched)}  selected: {len(selected)}")

    # 5) Final polynomial fit with iterative kappa-sigma clipping.
    solution = fitting.fit_solution(
        selected,
        kappa=params.sigma_clip_kappa,
        max_iter=params.sigma_clip_max_iter,
    )
    print(
        f"[{night.date}] degree={solution.degree}  rms={solution.rms:.4f} A  "
        f"lines used={len(solution.lines)}  iters={solution.n_iter}"
    )

    # 6) Apply pixel -> lambda to lamp and star.
    thar["lambda_cal"] = solution.apply(thar["pixel"].to_numpy())
    star["lambda_cal"] = solution.apply(star["pixel"].to_numpy())

    # 7) Telluric rest-frame correction (star only; lamp is laboratory frame).
    v_tel = telluric.telluric_velocity(night.date)
    star["lambda_rest"] = telluric.to_rest_frame(star["lambda_cal"].to_numpy(), v_tel)
    print(f"[{night.date}] telluric v = {v_tel:+.3f} km/s")

    # 8) Persist artifacts.
    io.save_table(thar, out_dir / "thar_calibrado.csv")
    io.save_table(star, out_dir / "estrella_calibrada.csv")
    io.save_table(solution.lines, out_dir / "lineas_calibracion.csv")
    _save_solution_metadata(night, solution, v_tel, seed_origin, out_dir / "solution.json")

    # 9) If this is the reference (manual) night, refresh the fingerprint.
    if seed_origin == "manual":
        fp_mod.save_fingerprint(thar, solution.lines, FINGERPRINT_PATH)
        print(f"[{night.date}] fingerprint refreshed -> {FINGERPRINT_PATH}")

    # 10) Diagnostic plots.
    if params.make_plots:
        plotting.plot_spectrum(thar, f"Th-Ar {night.date}", out_dir / "thar.png")
        plotting.plot_spectrum(star, f"Estrella {night.date}", out_dir / "estrella.png")
        plotting.plot_lines_used(thar, solution.lines, out_dir / "lineas.png")

    return CalibrationResult(
        night=night,
        star=star,
        thar=thar,
        solution=solution,
        telluric_kms=v_tel,
        seed=seed_df,
        seed_origin=seed_origin,
        matched=matched,
        out_dir=out_dir,
    )


def calibrate_all(
    nights: list[Night], params: CalibrationParams | None = None
) -> list[CalibrationResult]:
    """Run the pipeline on every night in the manifest.

    Nights with a manual seed are processed first so the fingerprint is
    available before any fingerprint-only night runs.
    """
    ordered = sorted(nights, key=lambda n: (n.seed_file is None, n.date))
    results: list[CalibrationResult] = []
    for night in ordered:
        try:
            results.append(calibrate_night(night, params))
        except Exception as exc:
            print(f"[{night.date}] FAILED: {exc}")
    return results


def _resolve_seed(night: Night, thar: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if night.seed_file is not None and Path(night.seed_file).exists():
        return seeds.load_seed(night.seed_file), "manual"

    if not FINGERPRINT_PATH.exists():
        raise FileNotFoundError(
            f"No seed file for {night.date} and no fingerprint at "
            f"{FINGERPRINT_PATH}. Calibrate the reference night first."
        )
    fp = fp_mod.load_fingerprint(FINGERPRINT_PATH)
    return fp_mod.bootstrap_seed(thar, fp), "fingerprint"


def _save_solution_metadata(
    night: Night,
    solution: fitting.Solution,
    v_tel: float,
    seed_origin: str,
    path: Path,
) -> None:
    payload = {
        "date": night.date,
        "degree": solution.degree,
        "coef": solution.coef.tolist(),
        "rms": solution.rms,
        "n_lines": int(len(solution.lines)),
        "telluric_kms": v_tel,
        "seed_origin": seed_origin,
    }
    path.write_text(json.dumps(payload, indent=2))
