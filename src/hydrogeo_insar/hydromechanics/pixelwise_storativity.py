from __future__ import annotations

import json
from typing import Any

import numpy as np
import rasterio

from ..common import ensure_dir, read_tif, write_json, write_tif
from ..config import ProjectConfig
from .seasonal import rotate_coefficients


def _qstats(a: np.ndarray) -> dict[str, Any]:
    x = np.asarray(a, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "p01": float(np.percentile(x, 1)),
        "p10": float(np.percentile(x, 10)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "p90": float(np.percentile(x, 90)),
        "p99": float(np.percentile(x, 99)),
    }


def estimate_pixelwise_ske(cfg: ProjectConfig) -> dict[str, Any]:
    """Primary v0.4 Ske estimator: pixelwise lag-aligned harmonic LS."""
    seasonal = ensure_dir(cfg.outputs / "seasonal")
    sresp = cfg.section("seasonal_response")
    sec = cfg.section("ske")

    period = float(sresp.get("annual_period_days", 365.2425))
    min_damp_mm = float(sresp.get("min_deformation_amplitude_mm", 0.5))
    min_hamp_m = float(sresp.get("min_head_amplitude_m", 0.2))
    min_ske = float(sec.get("min", 0.0))
    max_ske = float(sec.get("max", 0.05))
    min_cosine = float(sec.get("min_vector_cosine", 0.0))
    hc_cosine = float(sec.get("high_confidence_min_vector_cosine", 0.7))

    lag_days = float(json.loads(
        (seasonal / "lag_summary.json").read_text(encoding="utf-8")
    )["lag_days"])

    ds = read_tif(seasonal / "deformation_annual_sin_mm.tif") / 1000.0
    dc = read_tif(seasonal / "deformation_annual_cos_mm.tif") / 1000.0
    hs = read_tif(seasonal / "head_annual_sin_m.tif")
    hc = read_tif(seasonal / "head_annual_cos_m.tif")

    rhs, rhc = rotate_coefficients(hs, hc, lag_days, period)
    damp = np.hypot(ds, dc)
    hamp = np.hypot(rhs, rhc)
    dot = ds * rhs + dc * rhc
    denom = rhs * rhs + rhc * rhc
    cosine = dot / np.maximum(damp * hamp, 1e-12)
    raw = dot / np.maximum(denom, 1e-12)

    gw_support = read_tif(
        cfg.outputs / "groundwater" / "groundwater_support_mask.tif"
    ) > 0

    base = (
        np.isfinite(raw)
        & np.isfinite(cosine)
        & np.isfinite(damp)
        & np.isfinite(hamp)
        & gw_support
        & (damp * 1000.0 >= min_damp_mm)
        & (hamp >= min_hamp_m)
        & (cosine > min_cosine)
    )
    support = base & (raw >= min_ske) & (raw <= max_ske)
    high = support & (cosine >= hc_cosine)

    raw_out = np.full(raw.shape, np.nan, dtype="float32")
    raw_out[base] = raw[base].astype("float32")
    ske = np.full(raw.shape, np.nan, dtype="float32")
    ske[support] = raw[support].astype("float32")
    ske_high = np.full(raw.shape, np.nan, dtype="float32")
    ske_high[high] = raw[high].astype("float32")

    with rasterio.open(seasonal / "deformation_annual_sin_mm.tif") as ref:
        crs, transform = str(ref.crs), ref.transform

    write_tif(seasonal / "ske_pixelwise_raw_ratio.tif", raw_out, crs, transform)
    write_tif(seasonal / "ske_pixelwise.tif", ske, crs, transform)
    write_tif(
        seasonal / "ske_pixelwise_support_mask.tif",
        support.astype("uint8"), crs, transform, nodata=0, dtype="uint8",
    )
    write_tif(
        seasonal / "ske_pixelwise_high_confidence.tif",
        ske_high, crs, transform,
    )
    write_tif(
        seasonal / "ske_pixelwise_high_confidence_mask.tif",
        high.astype("uint8"), crs, transform, nodata=0, dtype="uint8",
    )
    write_tif(
        seasonal / "seasonal_vector_cosine.tif",
        cosine.astype("float32"), crs, transform,
    )

    n_gw = int(gw_support.sum())
    result = {
        "status": "ok",
        "method": "pixelwise_lag_aligned_harmonic_vector_least_squares",
        "equation": "Ske=(d_s*h_s_tau+d_c*h_c_tau)/(h_s_tau^2+h_c_tau^2)",
        # Backward-compatible scalar aliases used by v0.3 integration tests
        # and lightweight downstream summaries. Detailed statistics remain
        # available under primary_ske/high_confidence_ske.
        "lag_days": lag_days,
        "regional_lag_days": lag_days,
        "ske_median": float(np.nanmedian(ske)),
        "ske_p10": float(np.nanpercentile(ske, 10)),
        "ske_p90": float(np.nanpercentile(ske, 90)),
        "support_pixels": int(support.sum()),
        "period_days": period,
        "min_deformation_amplitude_mm": min_damp_mm,
        "min_head_amplitude_m": min_hamp_m,
        "physical_bounds": [min_ske, max_ske],
        "primary_min_vector_cosine": min_cosine,
        "high_confidence_min_vector_cosine": hc_cosine,
        "groundwater_support_pixels": n_gw,
        "primary_support_pixels": int(support.sum()),
        "primary_support_fraction_of_groundwater_domain":
            float(support.sum() / n_gw) if n_gw else np.nan,
        "high_confidence_support_pixels": int(high.sum()),
        "high_confidence_support_fraction_of_groundwater_domain":
            float(high.sum() / n_gw) if n_gw else np.nan,
        "primary_ske": _qstats(ske),
        "high_confidence_ske": _qstats(ske_high),
        "vector_cosine_on_primary_support":
            _qstats(np.where(support, cosine, np.nan)),
    }
    write_json(seasonal / "ske_pixelwise_summary.json", result)
    return result
