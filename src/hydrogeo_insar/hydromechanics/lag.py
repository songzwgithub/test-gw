from __future__ import annotations

from typing import Any

import numpy as np
import rasterio

from ..common import ensure_dir, read_tif, write_json, write_tif
from ..config import ProjectConfig
from .seasonal import rotate_coefficients


def estimate_lag(cfg: ProjectConfig) -> dict[str, Any]:
    """Estimate pixel phase lag for diagnosis and one regional lag for Ske inversion.

    Regional lag maximizes quality-weighted harmonic-vector similarity. The
    quality term uses harmonic fit residuals relative to annual amplitudes.
    """
    sec = cfg.section("seasonal_response")
    period = float(sec.get("annual_period_days", 365.2425))
    search = sec.get("lag_search_days", [0.0, 180.0])
    step = float(sec.get("lag_step_days", 0.5))
    lags = np.arange(float(search[0]), float(search[1]) + 0.5 * step, step)
    out_dir = ensure_dir(cfg.outputs / "seasonal")

    ds_mm = read_tif(out_dir / "deformation_annual_sin_mm.tif")
    dc_mm = read_tif(out_dir / "deformation_annual_cos_mm.tif")
    hs = read_tif(out_dir / "head_annual_sin_m.tif")
    hc = read_tif(out_dir / "head_annual_cos_m.tif")
    drmse = read_tif(out_dir / "deformation_fit_rmse_mm.tif")
    hrmse = read_tif(out_dir / "head_fit_rmse_m.tif")

    damp = np.hypot(ds_mm, dc_mm)
    hamp = np.hypot(hs, hc)
    valid = np.isfinite(ds_mm) & np.isfinite(dc_mm) & np.isfinite(hs) & np.isfinite(hc)
    valid &= np.isfinite(drmse) & np.isfinite(hrmse)
    valid &= damp >= float(sec.get("min_deformation_amplitude_mm", 0.5))
    valid &= hamp >= float(sec.get("min_head_amplitude_m", 0.2))
    support_path = cfg.outputs / "groundwater" / "groundwater_support_mask.tif"
    if support_path.exists():
        valid &= read_tif(support_path) > 0
    if valid.sum() < 10:
        raise ValueError("Too few pixels for seasonal lag estimation")

    phase_d = (np.arctan2(ds_mm, dc_mm) * period / (2.0 * np.pi)) % period
    phase_h = (np.arctan2(hs, hc) * period / (2.0 * np.pi)) % period
    phase_lag = (phase_d - phase_h) % period
    phase_lag[~valid] = np.nan

    with rasterio.open(out_dir / "head_annual_sin_m.tif") as ref:
        write_tif(out_dir / "phase_lag_days.tif", phase_lag.astype("float32"), str(ref.crs), ref.transform)

    quality = 1.0 / (
        1.0
        + (drmse / np.maximum(damp, 1e-6)) ** 2
        + (hrmse / np.maximum(hamp, 1e-6)) ** 2
    )
    weight = quality * damp * hamp
    weight[~valid] = np.nan

    idx = np.flatnonzero(valid)
    max_samples = int(sec.get("lag_sample_size", 200000))
    if len(idx) > max_samples:
        rng = np.random.default_rng(int(sec.get("random_state", 20260919)))
        idx = rng.choice(idx, max_samples, replace=False)

    ds = ds_mm.ravel()[idx] / 1000.0
    dc = dc_mm.ravel()[idx] / 1000.0
    hsv = hs.ravel()[idx]
    hcv = hc.ravel()[idx]
    w = weight.ravel()[idx]
    dnorm = np.maximum(np.hypot(ds, dc), 1e-12)

    rows = []
    best_lag, best_score = None, -np.inf
    for lag in lags:
        rs, rc = rotate_coefficients(hsv, hcv, lag, period)
        hnorm = np.maximum(np.hypot(rs, rc), 1e-12)
        cosine = (ds * rs + dc * rc) / (dnorm * hnorm)
        score = float(np.nansum(w * cosine) / np.nansum(w))
        rows.append((lag, score))
        if score > best_score:
            best_lag, best_score = float(lag), score

    np.savetxt(
        out_dir / "lag_scan.csv",
        np.asarray(rows),
        delimiter=",",
        header="lag_days,weighted_cosine_similarity",
        comments="",
    )
    summary = {
        "status": "ok",
        "lag_days": best_lag,
        "weighted_cosine_similarity": best_score,
        "pixel_phase_lag_median_days": float(np.nanmedian(phase_lag)),
        "valid_pixels": int(valid.sum()),
    }
    write_json(out_dir / "lag_summary.json", summary)
    return summary
