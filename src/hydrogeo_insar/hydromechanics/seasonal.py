from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import rasterio

from ..common import block_slices, days_to_dates, ensure_dir, h5_grid_metadata, read_tif, write_json, write_tif
from ..config import ProjectConfig
from ..temporal.fit import fit_block
from ..temporal.model import TimeModel, coefficient_indices, design_matrix
from ..temporal.select import choose_global_polynomial_degree


def rotate_coefficients(sin_coef: np.ndarray, cos_coef: np.ndarray, lag_days: float, period_days: float):
    """Coefficients of y(t-lag) for y=s*sin(wt)+c*cos(wt). Positive lag delays the response."""
    angle = 2.0 * np.pi * float(lag_days) / float(period_days)
    ca, sa = np.cos(angle), np.sin(angle)
    return sin_coef * ca + cos_coef * sa, cos_coef * ca - sin_coef * sa


def _common_dates(insar_dates: np.ndarray, head_dates: np.ndarray):
    common, i_idx, h_idx = np.intersect1d(insar_dates, head_dates, assume_unique=True, return_indices=True)
    return common, i_idx, h_idx


def _sample_head_series(field_path: Path, h_idx: np.ndarray, max_series: int, block_size: int):
    grid = h5_grid_metadata(field_path)
    collected = []
    with h5py.File(field_path, "r") as h5:
        for r0, r1, c0, c1 in block_slices(grid["height"], grid["width"], block_size):
            arr = h5["head_anomaly_m"][h_idx, r0:r1, c0:c1].astype(float)
            T = arr.shape[0]
            flat = arr.reshape(T, -1)
            good = np.isfinite(flat).sum(axis=0) >= max(12, T // 2)
            if good.any():
                take = flat[:, good]
                if take.shape[1] > 200:
                    take = take[:, :: max(1, take.shape[1] // 200)]
                collected.append(take)
            if sum(x.shape[1] for x in collected) >= max_series:
                break
    if not collected:
        raise ValueError("No groundwater field pixels are available for temporal model selection")
    out = np.concatenate(collected, axis=1)
    return out[:, :max_series]


def compute_joint_harmonics(cfg: ProjectConfig) -> dict[str, Any]:
    """Fit InSAR and groundwater annual harmonics on exactly the same acquisition dates."""
    sec = cfg.section("seasonal_response")
    period = float(sec.get("annual_period_days", 365.2425))
    min_obs = int(sec.get("min_observations", 24))
    block_size = int(sec.get("block_size", 256))
    insar_path = cfg.outputs / "canonical" / "insar_stack.h5"
    head_path = cfg.outputs / "groundwater" / "groundwater_field.h5"
    grid = h5_grid_metadata(insar_path)
    out_dir = ensure_dir(cfg.outputs / "seasonal")

    with h5py.File(insar_path, "r") as ih5, h5py.File(head_path, "r") as hh5:
        insar_dates = days_to_dates(ih5["date_days"][:])
        head_dates = days_to_dates(hh5["date_days"][:])
        dates, i_idx, h_idx = _common_dates(insar_dates, head_dates)
    if len(dates) < min_obs:
        raise ValueError("Too few common InSAR-groundwater epochs for seasonal analysis")

    d_degree = int(sec.get("deformation_polynomial_degree", 2))
    g_degree_cfg = sec.get("groundwater_polynomial_degree", "auto")
    model_selection_rows = []
    if str(g_degree_cfg).lower() == "auto":
        sample = _sample_head_series(head_path, h_idx, int(sec.get("model_selection_sample_size", 3000)), block_size)
        candidates = [int(v) for v in sec.get("groundwater_polynomial_candidates", [1, 2])]
        g_degree, model_selection_rows = choose_global_polynomial_degree(
            dates, sample, candidates=candidates, periods_days=(period,), min_obs=min_obs
        )
    else:
        g_degree = int(g_degree_cfg)

    d_model = TimeModel(polynomial_degree=d_degree, periods_days=(period,))
    g_model = TimeModel(polynomial_degree=g_degree, periods_days=(period,))
    Xd, _ = design_matrix(dates, d_model)
    Xg, _ = design_matrix(dates, g_model)
    didx = coefficient_indices(d_model)
    gidx = coefficient_indices(g_model)
    dp = didx["periodic"][0]
    gp = gidx["periodic"][0]

    products = {
        "deformation_annual_sin_mm": np.full((grid["height"], grid["width"]), np.nan, dtype="float32"),
        "deformation_annual_cos_mm": np.full((grid["height"], grid["width"]), np.nan, dtype="float32"),
        "deformation_annual_amplitude_mm": np.full((grid["height"], grid["width"]), np.nan, dtype="float32"),
        "deformation_annual_phase_day": np.full((grid["height"], grid["width"]), np.nan, dtype="float32"),
        "head_intercept_m": np.full((grid["height"], grid["width"]), np.nan, dtype="float32"),
        "head_linear_m_yr": np.full((grid["height"], grid["width"]), np.nan, dtype="float32"),
        "head_annual_sin_m": np.full((grid["height"], grid["width"]), np.nan, dtype="float32"),
        "head_annual_cos_m": np.full((grid["height"], grid["width"]), np.nan, dtype="float32"),
        "head_annual_amplitude_m": np.full((grid["height"], grid["width"]), np.nan, dtype="float32"),
        "head_annual_phase_day": np.full((grid["height"], grid["width"]), np.nan, dtype="float32"),
    }
    if g_degree >= 2:
        products["head_quadratic_m_yr2"] = np.full((grid["height"], grid["width"]), np.nan, dtype="float32")

    with h5py.File(insar_path, "r") as ih5, h5py.File(head_path, "r") as hh5:
        for r0, r1, c0, c1 in block_slices(grid["height"], grid["width"], block_size):
            darr = ih5["displacement_mm"][i_idx, r0:r1, c0:c1].astype(float)
            harr = hh5["head_anomaly_m"][h_idx, r0:r1, c0:c1].astype(float)
            T, bh, bw = darr.shape
            dbeta, _drmse, _dn, _ = fit_block(darr.reshape(T, -1), Xd, min_obs=min_obs)
            hbeta, _hrmse, _hn, _ = fit_block(harr.reshape(T, -1), Xg, min_obs=min_obs)
            ds, dc = dbeta[:, dp["sin"]], dbeta[:, dp["cos"]]
            hs, hc = hbeta[:, gp["sin"]], hbeta[:, gp["cos"]]
            vals = {
                "deformation_annual_sin_mm": ds,
                "deformation_annual_cos_mm": dc,
                "deformation_annual_amplitude_mm": np.hypot(ds, dc),
                "deformation_annual_phase_day": (np.arctan2(ds, dc) * period / (2*np.pi)) % period,
                "head_intercept_m": hbeta[:, 0],
                "head_linear_m_yr": hbeta[:, 1],
                "head_annual_sin_m": hs,
                "head_annual_cos_m": hc,
                "head_annual_amplitude_m": np.hypot(hs, hc),
                "head_annual_phase_day": (np.arctan2(hs, hc) * period / (2*np.pi)) % period,
            }
            if g_degree >= 2:
                vals["head_quadratic_m_yr2"] = hbeta[:, 2]
            for key, x in vals.items():
                products[key][r0:r1, c0:c1] = x.reshape(bh, bw).astype("float32")

    for key, arr in products.items():
        write_tif(out_dir / f"{key}.tif", arr, grid["crs"], grid["transform"])
    if model_selection_rows:
        pd.DataFrame(model_selection_rows).to_csv(out_dir / "groundwater_temporal_model_selection.csv", index=False)
    result = {
        "status": "ok",
        "output_directory": str(out_dir),
        "common_epochs": int(len(dates)),
        "first_date": str(dates[0]),
        "last_date": str(dates[-1]),
        "deformation_polynomial_degree": d_degree,
        "groundwater_polynomial_degree": g_degree,
        "period_days": period,
    }
    write_json(out_dir / "joint_harmonics_summary.json", result)
    return result


def estimate_lag(cfg: ProjectConfig) -> dict[str, Any]:
    sec = cfg.section("seasonal_response")
    period = float(sec.get("annual_period_days", 365.2425))
    search = sec.get("lag_search_days", [0.0, 180.0])
    step = float(sec.get("lag_step_days", 0.5))
    lags = np.arange(float(search[0]), float(search[1]) + step*0.5, step)
    out_dir = ensure_dir(cfg.outputs / "seasonal")

    ds = read_tif(out_dir / "deformation_annual_sin_mm.tif") / 1000.0
    dc = read_tif(out_dir / "deformation_annual_cos_mm.tif") / 1000.0
    hs = read_tif(out_dir / "head_annual_sin_m.tif")
    hc = read_tif(out_dir / "head_annual_cos_m.tif")
    damp_mm = np.hypot(ds, dc) * 1000.0
    hamp_m = np.hypot(hs, hc)
    valid = np.isfinite(ds) & np.isfinite(dc) & np.isfinite(hs) & np.isfinite(hc)
    valid &= hamp_m >= float(sec.get("min_head_amplitude_m", 0.2))
    valid &= damp_mm >= float(sec.get("min_deformation_amplitude_mm", 0.5))
    support_path = cfg.outputs / "groundwater" / "groundwater_support_mask.tif"
    if support_path.exists():
        valid &= read_tif(support_path) > 0
    if valid.sum() < 10:
        raise ValueError("Too few pixels for seasonal lag estimation")

    phase_d = (np.arctan2(ds, dc) * period / (2*np.pi)) % period
    phase_h = (np.arctan2(hs, hc) * period / (2*np.pi)) % period
    phase_lag = (phase_d - phase_h) % period
    phase_lag[~valid] = np.nan
    with rasterio.open(out_dir / "head_annual_sin_m.tif") as ref:
        write_tif(out_dir / "phase_lag_days.tif", phase_lag.astype("float32"), str(ref.crs), ref.transform)

    idx = np.flatnonzero(valid)
    max_samples = int(sec.get("lag_sample_size", 200000))
    if len(idx) > max_samples:
        rng = np.random.default_rng(int(sec.get("random_state", 20260919)))
        idx = rng.choice(idx, max_samples, replace=False)
    dsv, dcv = ds.ravel()[idx], dc.ravel()[idx]
    hsv, hcv = hs.ravel()[idx], hc.ravel()[idx]
    dnorm = np.maximum(np.hypot(dsv, dcv), 1e-12)
    rows = []
    best_lag, best_score = None, -np.inf
    for lag in lags:
        rs, rc = rotate_coefficients(hsv, hcv, lag, period)
        hnorm = np.maximum(np.hypot(rs, rc), 1e-12)
        cosine = (dsv*rs + dcv*rc) / (dnorm*hnorm)
        score = float(np.nanmedian(cosine))
        rows.append((lag, score))
        if score > best_score:
            best_lag, best_score = float(lag), score
    np.savetxt(out_dir / "lag_scan.csv", np.asarray(rows), delimiter=",", header="lag_days,median_cosine_similarity", comments="")
    summary = {
        "status": "ok",
        "lag_days": best_lag,
        "median_cosine_similarity": best_score,
        "pixel_phase_lag_median_days": float(np.nanmedian(phase_lag)),
        "valid_pixels": int(valid.sum()),
    }
    write_json(out_dir / "lag_summary.json", summary)
    return summary
