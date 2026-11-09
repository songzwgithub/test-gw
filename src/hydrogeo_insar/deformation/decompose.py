from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import h5py
import numpy as np
import rasterio

from ..common import (
    block_slices,
    days_to_dates,
    ensure_dir,
    h5_grid_metadata,
    write_json,
)
from ..config import ProjectConfig
from ..temporal.fit import fit_block
from ..temporal.model import (
    TimeModel,
    coefficient_indices,
    design_matrix,
)


def _bic_from_rss(rss: np.ndarray, n: np.ndarray, k: int) -> np.ndarray:
    """Bayesian information criterion for independent Gaussian residuals."""
    rss = np.asarray(rss, dtype=float)
    n = np.asarray(n, dtype=float)
    out = np.full(rss.shape, np.nan, dtype=float)
    ok = np.isfinite(rss) & np.isfinite(n) & (n > k) & (rss >= 0.0)
    if ok.any():
        variance = np.maximum(
            rss[ok] / n[ok],
            np.finfo(float).tiny,
        )
        out[ok] = n[ok] * np.log(variance) + float(k) * np.log(n[ok])
    return out


def _bic_classes(delta_bic: np.ndarray, threshold: float):
    """Return best-model and evidence classes.

    delta_bic = BIC_linear - BIC_quadratic
      > 0 : quadratic has lower BIC
      < 0 : linear has lower BIC

    preferred_model:
      0 nodata, 1 linear+annual, 2 quadratic+annual

    evidence_class:
      0 nodata, 1 linear supported, 2 inconclusive, 3 quadratic supported
    """
    d = np.asarray(delta_bic, dtype=float)
    preferred = np.zeros(d.shape, dtype="uint8")
    evidence = np.zeros(d.shape, dtype="uint8")
    ok = np.isfinite(d)

    preferred[ok & (d <= 0.0)] = 1
    preferred[ok & (d > 0.0)] = 2

    evidence[ok & (d <= -float(threshold))] = 1
    evidence[ok & (np.abs(d) < float(threshold))] = 2
    evidence[ok & (d >= float(threshold))] = 3
    return preferred, evidence


def _writers(out_dir: Path, grid: dict[str, Any], include_quadratic: bool):
    names = {
        "intercept_mm": "intercept_mm.tif",
        "linear_coeff_mm_yr": "linear_coeff_mm_yr.tif",
        "start_rate_mm_yr": "start_rate_mm_yr.tif",
        "end_rate_mm_yr": "end_rate_mm_yr.tif",
        "annual_sin_mm": "annual_sin_mm.tif",
        "annual_cos_mm": "annual_cos_mm.tif",
        "annual_amplitude_mm": "annual_amplitude_mm.tif",
        "annual_phase_day": "annual_phase_day.tif",
        "fit_rmse_mm": "fit_rmse_mm.tif",
        "n_observations": "n_observations.tif",
    }
    if include_quadratic:
        names.update({
            "quadratic_coeff_mm_yr2": "quadratic_coeff_mm_yr2.tif",
            "rate_change_mm_yr": "rate_change_mm_yr.tif",
            "vertex_time_year": "vertex_time_year.tif",
            "vertex_feature_year": "vertex_feature_year.tif",
            "linear_annual_rmse_mm": "linear_annual_rmse_mm.tif",
            "quadratic_annual_rmse_mm": "quadratic_annual_rmse_mm.tif",
            "delta_bic_linear_minus_quadratic": "delta_bic_linear_minus_quadratic.tif",
            "preferred_model": "preferred_model.tif",
            "bic_evidence_class": "bic_evidence_class.tif",
        })

    writers = {}
    base = {
        "driver": "GTiff",
        "height": grid["height"],
        "width": grid["width"],
        "count": 1,
        "crs": grid["crs"],
        "transform": grid["transform"],
        "compress": "deflate",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }

    for key, fn in names.items():
        profile = base.copy()
        if key == "n_observations":
            profile.update(dtype="uint16", nodata=0)
        elif key in {"preferred_model", "bic_evidence_class"}:
            profile.update(dtype="uint8", nodata=0)
        else:
            profile.update(dtype="float32", nodata=np.nan)
        writers[key] = rasterio.open(out_dir / fn, "w", **profile)

    return writers, names


def decompose_insar(cfg: ProjectConfig) -> dict[str, Any]:
    sec = cfg.section("deformation")
    period = float(sec.get("annual_period_days", 365.2425))
    degree = int(sec.get("polynomial_degree", 2))
    if degree not in {1, 2}:
        raise ValueError(
            "deformation.polynomial_degree supports 1 or 2"
        )

    min_obs = int(sec.get("min_observations", 24))
    block_size = int(sec.get("block_size", 256))
    bic_threshold = float(sec.get("bic_evidence_threshold", 2.0))

    stack_path = cfg.outputs / "canonical" / "insar_stack.h5"
    grid = h5_grid_metadata(stack_path)
    out_dir = ensure_dir(cfg.outputs / "deformation")

    stage_t0 = time.perf_counter()

    with h5py.File(stack_path, "r") as h5:
        all_dates = days_to_dates(h5["date_days"][:])
        analysis = cfg.section("analysis")
        start = np.datetime64(
            str(analysis.get("start_date", all_dates[0])),
            "D",
        )
        end = np.datetime64(
            str(analysis.get("end_date", all_dates[-1])),
            "D",
        )
        use = (all_dates >= start) & (all_dates <= end)
        dates = all_dates[use]

        if len(dates) < max(min_obs, 8):
            raise ValueError("Too few InSAR epochs in analysis period")

        model_main = TimeModel(
            polynomial_degree=degree,
            periods_days=(period,),
        )
        X_main, t_year = design_matrix(dates, model_main)
        idx_main = coefficient_indices(model_main)
        duration = float(t_year[-1])

        if degree >= 2:
            model_linear = TimeModel(
                polynomial_degree=1,
                periods_days=(period,),
            )
            X_linear, _ = design_matrix(dates, model_linear)
            k_linear = int(X_linear.shape[1])
            k_quadratic = int(X_main.shape[1])
        else:
            X_linear = None
            k_linear = None
            k_quadratic = None

        writers, names = _writers(
            out_dir,
            grid,
            include_quadratic=(degree >= 2),
        )

        blocks = list(
            block_slices(
                grid["height"],
                grid["width"],
                block_size,
            )
        )
        total_blocks = len(blocks)
        report_every = max(1, total_blocks // 50)

        bic_valid = 0
        bic_linear_preferred = 0
        bic_quadratic_preferred = 0
        bic_linear_supported = 0
        bic_inconclusive = 0
        bic_quadratic_supported = 0

        print(
            f"[DECOMP] start epochs={len(dates)}, "
            f"grid={grid['height']}x{grid['width']}, "
            f"blocks={total_blocks}, degree={degree}",
            flush=True,
        )
        if degree >= 2:
            print(
                f"[DECOMP] BIC diagnostic: "
                f"M1=linear+annual (k={k_linear}), "
                f"M2=quadratic+annual (k={k_quadratic}), "
                f"evidence threshold=±{bic_threshold:g}",
                flush=True,
            )

        try:
            for block_no, (r0, r1, c0, c1) in enumerate(
                blocks,
                start=1,
            ):
                arr = h5[
                    "displacement_mm"
                ][
                    use,
                    r0:r1,
                    c0:c1,
                ].astype(float)

                T, bh, bw = arr.shape
                y = arr.reshape(T, -1)

                beta, rmse, n, rss = fit_block(
                    y,
                    X_main,
                    min_obs=min_obs,
                )

                intercept = beta[:, 0]
                b = beta[:, 1]

                if degree >= 2:
                    a = beta[:, 2]
                    start_rate = b
                    end_rate = b + 2.0 * a * duration
                    rate_change = end_rate - start_rate

                    eps = 1e-12
                    mathematical_vertex = np.where(
                        np.abs(a) > eps,
                        -b / (2.0 * a),
                        np.nan,
                    )

                    raw_vertex = np.where(
                        np.abs(a)
                        > float(
                            sec.get(
                                "vertex_min_abs_curvature",
                                1e-4,
                            )
                        ),
                        mathematical_vertex,
                        np.nan,
                    )

                    low = -duration
                    high = 2.0 * duration
                    fallback = np.where(
                        b < 0,
                        high,
                        low,
                    )
                    vertex_feature = np.where(
                        np.isfinite(mathematical_vertex),
                        np.clip(
                            mathematical_vertex,
                            low,
                            high,
                        ),
                        fallback,
                    )

                    _, rmse_linear, n_linear, rss_linear = fit_block(
                        y,
                        X_linear,
                        min_obs=min_obs,
                    )

                    bic_linear = _bic_from_rss(
                        rss_linear,
                        n_linear,
                        k_linear,
                    )
                    bic_quadratic = _bic_from_rss(
                        rss,
                        n,
                        k_quadratic,
                    )
                    delta_bic = bic_linear - bic_quadratic

                    preferred, evidence = _bic_classes(
                        delta_bic,
                        bic_threshold,
                    )

                    ok_bic = np.isfinite(delta_bic)
                    bic_valid += int(ok_bic.sum())
                    bic_linear_preferred += int(
                        np.sum(preferred == 1)
                    )
                    bic_quadratic_preferred += int(
                        np.sum(preferred == 2)
                    )
                    bic_linear_supported += int(
                        np.sum(evidence == 1)
                    )
                    bic_inconclusive += int(
                        np.sum(evidence == 2)
                    )
                    bic_quadratic_supported += int(
                        np.sum(evidence == 3)
                    )

                else:
                    a = None
                    start_rate = b
                    end_rate = b
                    rate_change = None
                    raw_vertex = None
                    vertex_feature = None

                pinfo = idx_main["periodic"][0]
                s = beta[:, pinfo["sin"]]
                c = beta[:, pinfo["cos"]]
                amp = np.hypot(s, c)
                phase = (
                    np.arctan2(s, c)
                    * period
                    / (2.0 * np.pi)
                ) % period

                out = {
                    "intercept_mm": intercept,
                    "linear_coeff_mm_yr": b,
                    "start_rate_mm_yr": start_rate,
                    "end_rate_mm_yr": end_rate,
                    "annual_sin_mm": s,
                    "annual_cos_mm": c,
                    "annual_amplitude_mm": amp,
                    "annual_phase_day": phase,
                    "fit_rmse_mm": rmse,
                    "n_observations": n,
                }

                if degree >= 2:
                    out.update({
                        "quadratic_coeff_mm_yr2": a,
                        "rate_change_mm_yr": rate_change,
                        "vertex_time_year": raw_vertex,
                        "vertex_feature_year": vertex_feature,
                        "linear_annual_rmse_mm": rmse_linear,
                        "quadratic_annual_rmse_mm": rmse,
                        "delta_bic_linear_minus_quadratic": delta_bic,
                        "preferred_model": preferred,
                        "bic_evidence_class": evidence,
                    })

                win = rasterio.windows.Window(
                    c0,
                    r0,
                    c1 - c0,
                    r1 - r0,
                )

                for key, values in out.items():
                    data = values.reshape(bh, bw)
                    if key == "n_observations":
                        writers[key].write(
                            data.astype("uint16"),
                            1,
                            window=win,
                        )
                    elif key in {
                        "preferred_model",
                        "bic_evidence_class",
                    }:
                        writers[key].write(
                            data.astype("uint8"),
                            1,
                            window=win,
                        )
                    else:
                        writers[key].write(
                            data.astype("float32"),
                            1,
                            window=win,
                        )

                if (
                    block_no == 1
                    or block_no % report_every == 0
                    or block_no == total_blocks
                ):
                    elapsed = time.perf_counter() - stage_t0
                    frac = block_no / total_blocks
                    eta = elapsed * (1.0 / frac - 1.0)
                    print(
                        f"[DECOMP] {block_no}/{total_blocks} "
                        f"({100*frac:5.1f}%) "
                        f"elapsed={elapsed/60:.1f}m "
                        f"ETA={eta/60:.1f}m",
                        flush=True,
                    )
        finally:
            for dst in writers.values():
                dst.close()

    summary = {
        "status": "ok",
        "output_directory": str(out_dir),
        "model": (
            f"polynomial_degree_{degree}_plus_annual_harmonic"
        ),
        "n_epochs": int(len(dates)),
        "first_date": str(dates[0]),
        "last_date": str(dates[-1]),
        "period_days": period,
        "duration_years": duration,
        "products": names,
    }

    if degree >= 2:
        denom = max(bic_valid, 1)
        summary["bic_diagnostic"] = {
            "delta_definition": (
                "BIC_linear_annual_minus_BIC_quadratic_annual"
            ),
            "positive_favors": "quadratic_plus_annual",
            "evidence_threshold_abs_delta_bic": bic_threshold,
            "valid_pixels": bic_valid,
            "linear_preferred_fraction": (
                bic_linear_preferred / denom
            ),
            "quadratic_preferred_fraction": (
                bic_quadratic_preferred / denom
            ),
            "linear_supported_fraction": (
                bic_linear_supported / denom
            ),
            "inconclusive_fraction": (
                bic_inconclusive / denom
            ),
            "quadratic_supported_fraction": (
                bic_quadratic_supported / denom
            ),
        }

    write_json(
        out_dir / "deformation_summary.json",
        summary,
    )

    print(
        f"[DECOMP] done total="
        f"{(time.perf_counter()-stage_t0)/60:.1f}m",
        flush=True,
    )

    return summary
