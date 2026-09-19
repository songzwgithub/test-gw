from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rasterio

from ..common import block_slices, days_to_dates, ensure_dir, h5_grid_metadata, write_json
from ..config import ProjectConfig


def design_matrix(dates: np.ndarray, period_days: float = 365.2425) -> tuple[np.ndarray, np.ndarray]:
    dates = np.asarray(dates, dtype="datetime64[D]")
    t_days = (dates - dates[0]).astype("timedelta64[D]").astype(float)
    t_year = t_days / period_days
    angle = 2.0 * np.pi * t_days / period_days
    X = np.column_stack([
        np.ones(len(dates)),
        t_year,
        t_year**2,
        np.sin(angle),
        np.cos(angle),
    ])
    return X, t_year


def fit_block(y: np.ndarray, X: np.ndarray, min_obs: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit c + b*t + a*t² + s*sin + c*cos for a T×P block."""
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y)
    n = valid.sum(axis=0)
    y0 = np.where(valid, y, 0.0)
    w = valid.astype(float)
    xtx = np.einsum("tp,ti,tj->pij", w, X, X, optimize=True)
    xty = np.einsum("tp,ti,tp->pi", w, X, y0, optimize=True)
    ridge = 1e-10 * np.eye(X.shape[1])[None, :, :]
    beta = np.full((y.shape[1], X.shape[1]), np.nan, dtype=float)
    good = n >= min_obs
    if good.any():
        beta[good] = np.linalg.solve(xtx[good] + ridge, xty[good][..., None])[..., 0]
    pred = X @ np.nan_to_num(beta.T, nan=0.0)
    residual = np.where(valid, y - pred, np.nan)
    rmse = np.sqrt(np.nanmean(residual**2, axis=0))
    rmse[~good] = np.nan
    return beta, rmse, n


def _writers(out_dir: Path, grid: dict[str, Any]):
    names = {
        "intercept_mm": "intercept_mm.tif",
        "linear_rate_mm_yr": "linear_rate_mm_yr.tif",
        "quadratic_coeff_mm_yr2": "quadratic_coeff_mm_yr2.tif",
        "start_rate_mm_yr": "start_rate_mm_yr.tif",
        "end_rate_mm_yr": "end_rate_mm_yr.tif",
        "rate_change_mm_yr": "rate_change_mm_yr.tif",
        "vertex_time_year": "vertex_time_year.tif",
        "annual_sin_mm": "annual_sin_mm.tif",
        "annual_cos_mm": "annual_cos_mm.tif",
        "annual_amplitude_mm": "annual_amplitude_mm.tif",
        "annual_phase_day": "annual_phase_day.tif",
        "fit_rmse_mm": "fit_rmse_mm.tif",
        "n_observations": "n_observations.tif",
    }
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
    min_obs = int(sec.get("min_observations", 24))
    block_size = int(sec.get("block_size", 256))
    stack_path = cfg.outputs / "canonical" / "insar_stack.h5"
    grid = h5_grid_metadata(stack_path)
    out_dir = ensure_dir(cfg.outputs / "deformation")

    with h5py.File(stack_path, "r") as h5:
        dates = days_to_dates(h5["date_days"][:])
        X, t_year = design_matrix(dates, period)
        duration = float(t_year[-1])
        writers, names = _writers(out_dir, grid)
        try:
            for r0, r1, c0, c1 in block_slices(grid["height"], grid["width"], block_size):
                arr = h5["displacement_mm"][:, r0:r1, c0:c1].astype(float)
                T, bh, bw = arr.shape
                beta, rmse, n = fit_block(arr.reshape(T, -1), X, min_obs=min_obs)
                intercept = beta[:, 0]
                b = beta[:, 1]
                a = beta[:, 2]
                s = beta[:, 3]
                c = beta[:, 4]
                start_rate = b
                end_rate = b + 2.0 * a * duration
                rate_change = end_rate - start_rate
                vertex = np.where(np.abs(a) > 1e-12, -b / (2.0 * a), np.nan)
                amp = np.hypot(s, c)
                phase = (np.arctan2(s, c) * period / (2.0*np.pi)) % period
                block = {
                    "intercept_mm": intercept,
                    "linear_rate_mm_yr": b,
                    "quadratic_coeff_mm_yr2": a,
                    "start_rate_mm_yr": start_rate,
                    "end_rate_mm_yr": end_rate,
                    "rate_change_mm_yr": rate_change,
                    "vertex_time_year": vertex,
                    "annual_sin_mm": s,
                    "annual_cos_mm": c,
                    "annual_amplitude_mm": amp,
                    "annual_phase_day": phase,
                    "fit_rmse_mm": rmse,
                    "n_observations": n,
                }
                win = rasterio.windows.Window(c0, r0, c1-c0, r1-r0)
                for key, values in block.items():
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
        "model": "quadratic_plus_annual_harmonic",
        "n_epochs": int(len(dates)),
        "period_days": period,
        "duration_years": duration,
        "products": names,
    }
    write_json(out_dir / "deformation_summary.json", summary)
    return summary
