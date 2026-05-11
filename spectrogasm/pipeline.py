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
from . import fitting, io, matching, peaks, plotting, radial_velocity as rv_mod, science, seeds, telluric
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
    rv_half_window: float = 2.0
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
    rv_lines: pd.DataFrame
    rv_summary: rv_mod.RVSummary
    rv_lines_joint: pd.DataFrame
    rv_summary_joint: rv_mod.RVSummary
    rv_joint_sigma: float
    ew: pd.DataFrame
    resolution: pd.DataFrame
    resolution_summary: science.ResolutionSummary
    snr: float
    per_element_rv: pd.DataFrame
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
    matched, report = matching.match_to_atlas(
        centroids, atlas, seed_coef, tol=params.match_tol
    )
    print(
        f"[{night.date}] seed range: {report.lam_min:.2f} - {report.lam_max:.2f} A  "
        f"atlas in window: {report.n_atlas_in_window}  "
        f"nearest |diff|: {report.nearest_absdiff:.3f} A"
    )

    if matched.empty:
        raise RuntimeError(
            f"0 atlas lines fell within match_tol={params.match_tol} A. "
            f"Atlas density in the seed-predicted window "
            f"[{report.lam_min:.2f}, {report.lam_max:.2f}] = "
            f"{report.n_atlas_in_window} lines; the closest atlas line was "
            f"{report.nearest_absdiff:.3f} A from the nearest peak. "
            "Either the atlas is too sparse, the seed is mis-identified, or "
            "the wavelength range is wrong. Try CalibrationParams(match_tol=...) "
            f">= {max(report.nearest_absdiff * 1.5, 0.5):.2f}."
        )

    selected = matching.select_strong_lines(
        matched,
        n_lines=params.n_lines,
        min_separation=params.min_line_separation,
    )
    print(f"[{night.date}] matched: {len(matched)}  selected: {len(selected)}")

    if len(selected) < 4:
        raise RuntimeError(
            f"Only {len(selected)} usable lines after selection (need >= 4). "
            "Lower CalibrationParams.min_line_separation or relax match_tol."
        )

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

    # 8) Stellar radial velocity from V / Ni / Ti Doppler shifts.
    # Two estimators run side-by-side so we can compare and later deprecate one.
    rv_lines = rv_mod.measure_rv_night(star, half_window=params.rv_half_window)
    rv_summary = rv_mod.summarize_rv(rv_lines)
    print(
        f"[{night.date}] v_rad (per-line) = {rv_summary.v_mean_kms:+.3f} +/- "
        f"{rv_summary.v_sem_kms:.3f} km/s  (n={rv_summary.n_used})"
    )

    rv_lines_joint, v_joint, sigma_joint = rv_mod.measure_rv_night_joint(
        star, half_window=params.rv_half_window
    )
    rv_summary_joint = rv_mod.summarize_rv(rv_lines_joint)
    print(
        f"[{night.date}] v_rad (joint)    = {v_joint:+.3f} km/s  "
        f"sigma_inst = {sigma_joint:.3f} A  "
        f"(n={rv_summary_joint.n_used})"
    )

    # 9) Extra science measurements: EW, spectral R, SNR, per-element RV.
    ew = science.equivalent_widths(rv_lines)
    res = science.spectral_resolution(thar, solution.lines)
    res_summary = science.summarize_resolution(res)
    snr = science.estimate_snr(star)
    per_elem = science.per_element_rv(rv_lines)
    print(
        f"[{night.date}] R = {res_summary.R_median:.0f}  "
        f"SNR ~ {snr:.0f}  "
        f"EW(V/Ni/Ti) ok = {int(ew['ok'].sum())}/{len(ew)}"
    )

    # 10) Persist artifacts.
    io.save_table(thar, out_dir / "thar_calibrado.csv")
    io.save_table(star, out_dir / "estrella_calibrada.csv")
    io.save_table(solution.lines, out_dir / "lineas_calibracion.csv")
    io.save_table(rv_lines, out_dir / "rv_lines.csv")
    io.save_table(rv_lines_joint, out_dir / "rv_lines_joint.csv")
    io.save_table(ew, out_dir / "equivalent_widths.csv")
    io.save_table(res, out_dir / "resolution_per_line.csv")
    _save_solution_metadata(night, solution, v_tel, seed_origin, out_dir / "solution.json")
    _save_rv_metadata(night, rv_summary, out_dir / "rv.json")
    _save_rv_metadata(
        night, rv_summary_joint, out_dir / "rv_joint.json",
        extra={"sigma_inst_A": sigma_joint, "v_joint_kms": v_joint},
    )
    science.save_science_json(
        night.date, ew, res, res_summary, snr, per_elem,
        out_dir / "science.json",
    )

    # 11) If this is the reference (manual) night, refresh the fingerprint.
    if seed_origin == "manual":
        fp_mod.save_fingerprint(thar, solution.lines, FINGERPRINT_PATH)
        print(f"[{night.date}] fingerprint refreshed -> {FINGERPRINT_PATH}")

    # 12) Diagnostic plots.
    if params.make_plots:
        plotting.plot_spectrum(thar, f"Th-Ar  ·  {night.date}",
                               out_dir / "thar.png", fill=True)
        plotting.plot_spectrum(star, f"Estrella  ·  {night.date}",
                               out_dir / "estrella.png", fill=False)
        plotting.plot_lines_used(thar, solution.lines, out_dir / "lineas.png")
        rv_mod.plot_rv_fits(
            star, rv_lines, out_dir / "rv_fits.png",
            half_window=params.rv_half_window,
            title="Ajustes Doppler · per-línea  ·  V / Ni / Ti",
        )
        rv_mod.plot_rv_fits(
            star, rv_lines_joint, out_dir / "rv_fits_joint.png",
            half_window=params.rv_half_window,
            title=(
                "Ajustes Doppler · conjunto (σ compartido)  ·  V / Ni / Ti  "
                f"·  v = {v_joint:+.2f} km/s,  σ = {sigma_joint:.3f} Å"
            ),
        )

    return CalibrationResult(
        night=night,
        star=star,
        thar=thar,
        solution=solution,
        telluric_kms=v_tel,
        seed=seed_df,
        seed_origin=seed_origin,
        matched=matched,
        rv_lines=rv_lines,
        rv_summary=rv_summary,
        rv_lines_joint=rv_lines_joint,
        rv_summary_joint=rv_summary_joint,
        rv_joint_sigma=sigma_joint,
        ew=ew,
        resolution=res,
        resolution_summary=res_summary,
        snr=snr,
        per_element_rv=per_elem,
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


def _save_rv_metadata(
    night: Night,
    summary: rv_mod.RVSummary,
    path: Path,
    extra: dict | None = None,
) -> None:
    payload = {
        "date": night.date,
        "n_used": summary.n_used,
        "v_mean_kms": summary.v_mean_kms,
        "v_median_kms": summary.v_median_kms,
        "v_std_kms": summary.v_std_kms,
        "v_sem_kms": summary.v_sem_kms,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2))
