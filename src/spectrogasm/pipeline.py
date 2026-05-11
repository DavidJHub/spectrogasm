"""End-to-end orchestration of the wavelength calibration pipeline.

This walks the same sixteen steps as the original notebook, but each step is
delegated to a small focused module so the pipeline can be re-run on any
night by swapping the input paths and the seed file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import io, peaks, seeds, matching, fitting, plotting


@dataclass
class CalibrationConfig:
    star_path: str | Path
    thar_path: str | Path
    atlas_path: str | Path
    seed_path: str | Path
    out_dir: str | Path
    atlas_lam_min: float = 3000.0
    atlas_lam_max: float = 10000.0
    peak_prominence: float = 2000.0
    peak_distance: int = 4
    match_tol: float = 0.20
    n_lines: int = 20
    min_line_separation: float = 8.0
    outlier_threshold: float = 0.08
    make_plots: bool = True


@dataclass
class CalibrationResult:
    star: pd.DataFrame
    thar: pd.DataFrame
    solution: fitting.Solution
    seed_lines: pd.DataFrame
    matched_lines: pd.DataFrame


def calibrate_night(cfg: CalibrationConfig) -> CalibrationResult:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load spectra and atlas.
    star = io.load_spectrum(cfg.star_path)
    thar = io.load_spectrum(cfg.thar_path)
    atlas = io.load_atlas(cfg.atlas_path, cfg.atlas_lam_min, cfg.atlas_lam_max)
    print(f"[1] atlas lines loaded: {len(atlas)}")

    # 2) Detect and refine lamp peaks.
    idx, prom = peaks.detect_peaks(
        thar["intensidad"].to_numpy(),
        prominence=cfg.peak_prominence,
        distance=cfg.peak_distance,
    )
    centroids = peaks.refine_centroids(
        thar["pixel"].to_numpy(),
        thar["intensidad"].to_numpy(),
        idx,
        prom,
    )
    print(f"[2] peaks detected: {len(idx)}  refined: {len(centroids)}")

    # 3) Seed polynomial from manual identifications.
    seed = seeds.load_seed(cfg.seed_path)
    seed_coef = seeds.seed_polynomial(seed, degree=2)
    print(f"[3] seed lines: {len(seed)}")

    # 4) Match every peak to the nearest atlas line.
    matched = matching.match_to_atlas(
        centroids, atlas, seed_coef, tol=cfg.match_tol
    )
    print(f"[4] matched candidates: {len(matched)}")

    # 5) Pick a clean subset of well-separated strong lines.
    selected = matching.select_strong_lines(
        matched,
        n_lines=cfg.n_lines,
        min_separation=cfg.min_line_separation,
    )
    io.save_table(selected, out_dir / "lineas_calibracion_iniciales.csv")
    print(f"[5] lines selected: {len(selected)}")

    # 6) Fit the final polynomial with one outlier-rejection pass.
    solution = fitting.fit_solution(
        selected, outlier_threshold=cfg.outlier_threshold
    )
    print(
        f"[6] final degree={solution.degree}  rms={solution.rms:.4f} A  "
        f"lines used={len(solution.lines)}"
    )

    # 7) Apply solution to lamp and star.
    thar["lambda_cal"] = solution.apply(thar["pixel"].to_numpy())
    star["lambda_cal"] = solution.apply(star["pixel"].to_numpy())

    # 8) Persist artifacts.
    io.save_table(thar, out_dir / "thor_calibrado.csv")
    io.save_table(star, out_dir / "estrella_calibrada.csv")
    io.save_table(solution.lines, out_dir / "lineas_calibracion_finales.csv")
    _save_solution_metadata(solution, out_dir / "solution.json")

    # 9) Diagnostic plots.
    if cfg.make_plots:
        plotting.plot_spectrum(thar, "Th-Ar calibrado", out_dir / "thar_calibrado.png")
        plotting.plot_spectrum(star, "Estrella calibrada", out_dir / "estrella_calibrada.png")
        plotting.plot_lines_used(thar, solution.lines, out_dir / "lineas_usadas.png")

    return CalibrationResult(
        star=star,
        thar=thar,
        solution=solution,
        seed_lines=seed,
        matched_lines=matched,
    )


def _save_solution_metadata(solution: fitting.Solution, path: Path) -> None:
    payload = {
        "degree": solution.degree,
        "coef": solution.coef.tolist(),
        "rms": solution.rms,
        "n_lines": int(len(solution.lines)),
    }
    path.write_text(json.dumps(payload, indent=2))
