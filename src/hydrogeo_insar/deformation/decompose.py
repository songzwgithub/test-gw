from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rasterio

from ..common import block_slices, days_to_dates, ensure_dir, h5_grid_metadata, write_json
from ..config import ProjectConfig
from ..temporal.fit import fit_block
from ..temporal.model import TimeModel, coefficient_indices, design_matrix


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
        else:
            profile.update(dtype="float32", nodata=np.nan)
        writers[key] = rasterio.open(out_dir / fn, "w", **profile)
    return writers, names


def decompose_insar(cfg: ProjectConfig) -> dict[str, Any]:
    sec = cfg.section("deformation")
    period = float(sec.get("annual_period_days", 365.2425))
    degree = int(sec.get("polynomial_degree", 2))
    if degree not in {1, 2}:
        raise ValueError("v0.2 deformation.polynomial_degree supports 1 or 2")
    min_obs = int(sec.get("min_observations", 24))
    block_size = int(sec.get("block_size", 256))
    stack_path = cfg.outputs / "canonical" / "insar_stack.h5"
    grid = h5_grid_metadata(stack_path)
    out_dir = ensure_dir(cfg.outputs / "deformation")

    with h5py.File(stack_path, "r") as h5:
        all_dates = days_to_dates(h5["date_days"][:])
        analysis = cfg.section("analysis")
        start = np.datetime64(str(analysis.get("start_date", all_dates[0])), "D")
        end = np.datetime64(str(analysis.get("end_date", all_dates[-1])), "D")
        use = (all_dates >= start) & (all_dates <= end)
        dates = all_dates[use]
        if len(dates) < max(min_obs, 8):
            raise ValueError("Too few InSAR epochs in analysis period")
        model = TimeModel(polynomial_degree=degree, periods_days=(period,))
        X, t_year = design_matrix(dates, model)
        idx = coefficient_indices(model)
        duration = float(t_year[-1])
        writers, names = _writers(out_dir, grid, include_quadratic=(degree >= 2))
        try:
            for r0, r1, c0, c1 in block_slices(grid["height"], grid["width"], block_size):
                arr = h5["displacement_mm"][use, r0:r1, c0:c1].astype(float)
                T, bh, bw = arr.shape
                beta, rmse, n, _rss = fit_block(arr.reshape(T, -1), X, min_obs=min_obs)
                intercept = beta[:, 0]
                b = beta[:, 1]
                if degree >= 2:
                    a = beta[:, 2]
                    start_rate = b
                    end_rate = b + 2.0 * a * duration
                    rate_change = end_rate - start_rate
                    eps = 1e-12
                    mathematical_vertex = np.where(np.abs(a) > eps, -b / (2.0 * a), np.nan)
                    raw_vertex = np.where(
                        np.abs(a) > float(sec.get("vertex_min_abs_curvature", 1e-4)),
                        mathematical_vertex,
                        np.nan,
                    )
                    # Clustering keeps the direction of far-away mathematical vertices.
                    low = -duration
                    high = 2.0 * duration
                    fallback = np.where(b < 0, high, low)
                    vertex_feature = np.where(
                        np.isfinite(mathematical_vertex),
                        np.clip(mathematical_vertex, low, high),
                        fallback,
                    )
                else:
                    a = None
                    start_rate = b
                    end_rate = b
                    rate_change = None
                    raw_vertex = None
                    vertex_feature = None

                pinfo = idx["periodic"][0]
                s = beta[:, pinfo["sin"]]
                c = beta[:, pinfo["cos"]]
                amp = np.hypot(s, c)
                phase = (np.arctan2(s, c) * period / (2.0 * np.pi)) % period
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
                    })
                win = rasterio.windows.Window(c0, r0, c1-c0, r1-r0)
                for key, values in out.items():
                    data = values.reshape(bh, bw)
                    if key == "n_observations":
                        writers[key].write(data.astype("uint16"), 1, window=win)
                    else:
                        writers[key].write(data.astype("float32"), 1, window=win)
        finally:
            for dst in writers.values():
                dst.close()

    summary = {
        "status": "ok",
        "output_directory": str(out_dir),
        "model": f"polynomial_degree_{degree}_plus_annual_harmonic",
        "n_epochs": int(len(dates)),
        "first_date": str(dates[0]),
        "last_date": str(dates[-1]),
        "period_days": period,
        "duration_years": duration,
        "products": names,
    }
    write_json(out_dir / "deformation_summary.json", summary)
    return summary
