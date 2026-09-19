from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter

from ..common import block_slices, days_to_dates, ensure_dir, h5_grid_metadata, write_json, write_tif
from ..config import ProjectConfig


def rotate_coefficients(sin_coef: np.ndarray, cos_coef: np.ndarray, lag_days: float, period_days: float):
    """Return coefficients of y(t-lag_days) for y=s*sin(wt)+c*cos(wt)."""
    angle = 2.0 * np.pi * float(lag_days) / float(period_days)
    ca, sa = np.cos(angle), np.sin(angle)
    out_sin = sin_coef * ca + cos_coef * sa
    out_cos = cos_coef * ca - sin_coef * sa
    return out_sin, out_cos


def _harmonic_design(dates: np.ndarray, period_days: float):
    dates = np.asarray(dates, dtype="datetime64[D]")
    t_days = (dates - dates[0]).astype("timedelta64[D]").astype(float)
    t_year = t_days / period_days
    a = 2.0 * np.pi * t_days / period_days
    return np.column_stack([np.ones(len(dates)), t_year, np.sin(a), np.cos(a)])


def _fit_harmonic_block(y: np.ndarray, X: np.ndarray, min_obs: int = 24):
    valid = np.isfinite(y)
    n = valid.sum(axis=0)
    y0 = np.where(valid, y, 0.0)
    w = valid.astype(float)
    xtx = np.einsum("tp,ti,tj->pij", w, X, X, optimize=True)
    xty = np.einsum("tp,ti,tp->pi", w, X, y0, optimize=True)
    beta = np.full((y.shape[1], X.shape[1]), np.nan, dtype=float)
    good = n >= min_obs
    if good.any():
        beta[good] = np.linalg.solve(xtx[good] + 1e-10*np.eye(X.shape[1])[None], xty[good][..., None])[..., 0]
    return beta, n


def compute_groundwater_harmonics(cfg: ProjectConfig) -> dict[str, Any]:
    sec = cfg.section("seasonal_response")
    period = float(sec.get("annual_period_days", cfg.section("deformation").get("annual_period_days", 365.2425)))
    min_obs = int(sec.get("min_observations", 24))
    block_size = int(sec.get("block_size", 256))
    field_path = cfg.outputs / "groundwater" / "groundwater_field.h5"
    grid = h5_grid_metadata(field_path)
    out_dir = ensure_dir(cfg.outputs / "seasonal")

    products = {
        "head_trend_m_yr": np.full((grid["height"], grid["width"]), np.nan, dtype="float32"),
        "head_annual_sin_m": np.full((grid["height"], grid["width"]), np.nan, dtype="float32"),
        "head_annual_cos_m": np.full((grid["height"], grid["width"]), np.nan, dtype="float32"),
        "head_annual_amplitude_m": np.full((grid["height"], grid["width"]), np.nan, dtype="float32"),
        "head_annual_phase_day": np.full((grid["height"], grid["width"]), np.nan, dtype="float32"),
    }
    with h5py.File(field_path, "r") as h5:
        dates = days_to_dates(h5["date_days"][:])
        X = _harmonic_design(dates, period)
        for r0, r1, c0, c1 in block_slices(grid["height"], grid["width"], block_size):
            arr = h5["head_anomaly_m"][:, r0:r1, c0:c1].astype(float)
            T, bh, bw = arr.shape
            beta, _n = _fit_harmonic_block(arr.reshape(T, -1), X, min_obs=min_obs)
            trend = beta[:, 1]
            s = beta[:, 2]
            c = beta[:, 3]
            amp = np.hypot(s, c)
            phase = (np.arctan2(s, c) * period / (2*np.pi)) % period
            for key, vals in {
                "head_trend_m_yr": trend,
                "head_annual_sin_m": s,
                "head_annual_cos_m": c,
                "head_annual_amplitude_m": amp,
                "head_annual_phase_day": phase,
            }.items():
                products[key][r0:r1, c0:c1] = vals.reshape(bh, bw).astype("float32")

    for key, arr in products.items():
        write_tif(out_dir / f"{key}.tif", arr, grid["crs"], grid["transform"])
    result = {"status": "ok", "output_directory": str(out_dir), "period_days": period, "products": list(products)}
    write_json(out_dir / "groundwater_harmonics_summary.json", result)
    return result


def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


def estimate_lag_and_ske(cfg: ProjectConfig) -> dict[str, Any]:
    sec = cfg.section("seasonal_response")
    ske_sec = cfg.section("ske")
    period = float(sec.get("annual_period_days", cfg.section("deformation").get("annual_period_days", 365.2425)))
    search = sec.get("lag_search_days", [0.0, 180.0])
    step = float(sec.get("lag_step_days", 0.5))
    lags = np.arange(float(search[0]), float(search[1]) + step*0.5, step)

    deformation_dir = cfg.outputs / "deformation"
    seasonal_dir = ensure_dir(cfg.outputs / "seasonal")
    ds = _read(deformation_dir / "annual_sin_mm.tif") / 1000.0
    dc = _read(deformation_dir / "annual_cos_mm.tif") / 1000.0
    hs = _read(seasonal_dir / "head_annual_sin_m.tif")
    hc = _read(seasonal_dir / "head_annual_cos_m.tif")
    damp = np.hypot(ds, dc) * 1000.0
    hamp = np.hypot(hs, hc)

    valid = np.isfinite(ds) & np.isfinite(dc) & np.isfinite(hs) & np.isfinite(hc)
    valid &= hamp >= float(sec.get("min_head_amplitude_m", 0.2))
    valid &= damp >= float(sec.get("min_deformation_amplitude_mm", 0.5))
    if valid.sum() < 10:
        raise ValueError("Too few pixels for seasonal lag estimation")

    max_samples = int(sec.get("lag_sample_size", 200000))
    idx = np.flatnonzero(valid)
    if len(idx) > max_samples:
        rng = np.random.default_rng(int(sec.get("random_state", 20260919)))
        idx = rng.choice(idx, max_samples, replace=False)
    dsv, dcv = ds.ravel()[idx], dc.ravel()[idx]
    hsv, hcv = hs.ravel()[idx], hc.ravel()[idx]
    dnorm = np.maximum(np.hypot(dsv, dcv), 1e-12)

    rows = []
    best_lag = None
    best_score = -np.inf
    for lag in lags:
        rs, rc = rotate_coefficients(hsv, hcv, lag, period)
        hnorm = np.maximum(np.hypot(rs, rc), 1e-12)
        cosine = (dsv*rs + dcv*rc) / (dnorm*hnorm)
        score = float(np.nanmedian(cosine))
        rows.append((lag, score))
        if score > best_score:
            best_score = score
            best_lag = float(lag)

    np.savetxt(seasonal_dir / "lag_scan.csv", np.asarray(rows), delimiter=",", header="lag_days,median_cosine_similarity", comments="")

    rhs, rhc = rotate_coefficients(hs, hc, best_lag, period)
    numerator = ds*rhs + dc*rhc
    denominator = rhs*rhs + rhc*rhc
    sigma = float(ske_sec.get("support_sigma_pixels", 0.0))
    if sigma > 0:
        m = np.isfinite(numerator) & np.isfinite(denominator)
        w = gaussian_filter(m.astype(float), sigma=sigma, mode="nearest")
        numerator = gaussian_filter(np.where(m, numerator, 0.0), sigma=sigma, mode="nearest") / np.maximum(w, 1e-12)
        denominator = gaussian_filter(np.where(m, denominator, 0.0), sigma=sigma, mode="nearest") / np.maximum(w, 1e-12)
    ske = numerator / np.maximum(denominator, 1e-12)
    ske[~valid] = np.nan
    ske_min = float(ske_sec.get("min", 0.0))
    ske_max = float(ske_sec.get("max", 0.05))
    ske = np.clip(ske, ske_min, ske_max)

    with rasterio.open(deformation_dir / "annual_sin_mm.tif") as ref:
        crs, transform = str(ref.crs), ref.transform
    write_tif(seasonal_dir / "ske_effective.tif", ske.astype("float32"), crs, transform)

    summary = {
        "status": "ok",
        "lag_days": best_lag,
        "median_cosine_similarity": best_score,
        "valid_pixels": int(valid.sum()),
        "ske_median": float(np.nanmedian(ske)),
        "ske_p10": float(np.nanpercentile(ske, 10)),
        "ske_p90": float(np.nanpercentile(ske, 90)),
        "ske_min_bound": ske_min,
        "ske_max_bound": ske_max,
        "support_sigma_pixels": sigma,
    }
    write_json(seasonal_dir / "seasonal_response_summary.json", summary)
    return summary
