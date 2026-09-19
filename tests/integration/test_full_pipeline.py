from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.transform import from_origin

from hydrogeo_insar.common import pixel_area_rows, read_tif
from hydrogeo_insar.config import load_config
from hydrogeo_insar.pipeline import run_pipeline
from hydrogeo_insar.visualization import plot_stage


def _make_case(root: Path):
    raw = root / "data" / "raw"
    insar_dir = raw / "insar"
    insar_dir.mkdir(parents=True)
    h = w = 20
    transform = from_origin(115.0, 38.5, 0.02, 0.02)
    crs = "EPSG:4326"
    ref = np.datetime64("2018-01-01", "D")
    dates = np.arange(np.datetime64("2018-01-13"), np.datetime64("2022-01-01"), np.timedelta64(24, "D"))
    period = 365.2425
    lag = 50.0
    ske = 0.002
    yy, xx = np.mgrid[0:h, 0:w]
    amp = 5.0 + 0.03 * xx + 0.02 * yy
    trend = 0.35 + 0.002 * xx
    phase = 0.4 + 0.002 * yy

    def head_field(date):
        t = float((date - ref).astype("timedelta64[D]").astype(int))
        return trend * (t / period) + amp * np.cos(2 * np.pi * t / period - phase)

    def irreversible_mm(date):
        t = max(0.0, float((date - ref).astype("timedelta64[D]").astype(int)) / period)
        left = [-10.0, -8.0, -6.0, -4.0, -3.0]
        right = [-4.0, -3.0, -2.0, -1.0, -0.5]
        out = np.zeros((h, w), dtype=float)
        for j in range(len(left)):
            dur = max(0.0, min(t - j, 1.0))
            if dur <= 0:
                continue
            out += np.where(xx < 10, left[j], right[j]) * dur
        return out

    def deformation_abs(date):
        delayed = head_field(date - np.timedelta64(int(round(lag)), "D"))
        return irreversible_mm(date) + ske * delayed * 1000.0

    d0 = deformation_abs(ref)
    profile = {"driver": "GTiff", "height": h, "width": w, "count": 1, "dtype": "float32", "crs": crs, "transform": transform, "nodata": -9999.0}
    for d in dates:
        arr = (deformation_abs(d) - d0).astype("float32")
        fn = insar_dir / f"geo_20180101_{str(d).replace('-', '')}.tif"
        with rasterio.open(fn, "w", **profile) as dst:
            dst.write(arr, 1)

    well_rc = [(2, 2), (2, 8), (2, 14), (2, 18), (7, 4), (7, 11), (7, 17), (12, 2), (12, 8), (12, 15), (17, 5), (17, 17)]
    daily = np.arange(ref, np.datetime64("2022-01-01"), np.timedelta64(1, "D"))
    rows = []
    for k, (r, c) in enumerate(well_rc):
        lon = 115.0 + (c + 0.5) * 0.02
        lat = 38.5 - (r + 0.5) * 0.02
        for d in daily:
            rows.append({"well_id": f"W{k:02d}", "date": str(d), "lon": lon, "lat": lat, "layer": "confined", "head": float(head_field(d)[r, c])})
    pd.DataFrame(rows).to_csv(raw / "groundwater.csv", index=False)

    cfg = {
        "project": {"name": "synthetic", "root": "."},
        "outputs": "outputs",
        "analysis": {"start_date": "2018-01-01", "end_date": "2021-12-31"},
        "insar": {"path": "data/raw/insar", "pattern": "geo_*.tif", "filename_regex": r"geo_(\d{8})_(\d{8})\.tif$", "unit": "mm", "quantity": "vertical_displacement", "positive": "uplift", "block_size": 10},
        "groundwater": {"path": "data/raw/groundwater.csv", "format": "long", "variable": "head", "fields": {"station_id": "well_id", "date": "date", "value": "head", "lon": "lon", "lat": "lat", "aquifer_class": "layer"}, "aquifer_labels": {"confined": "confined"}},
        "groundwater_field": {
            "aquifer": "confined", "max_gap_days": 0, "min_coverage_fraction": 0.9, "min_active_well_fraction": 0.9,
            "min_baseline_observations": 30, "svd_iterations": 10, "grid_block_size": 10,
            "baseline": {"start": "2018-01-01", "end": "2018-03-31"}, "rank_candidates": [2],
            "rbf": {"max_centers": 12, "sigma_km_candidates": [25], "ridge_candidates": [0.001]},
            "cross_validation": {"folds": 3, "block_km": 20, "random_state": 1, "harmonic_polynomial_degree": 1, "rmse_shortlist_fraction": 0.05},
            "support": {"max_nearest_well_km": None},
        },
        "deformation": {"polynomial_degree": 2, "annual_period_days": period, "min_observations": 24, "block_size": 10},
        "regimes": {"feature_scheme": "meng2026", "k_candidates": [2], "k": 2, "training_spacing_km": 4.0, "selection_sample_size": 1000, "stable_rate_mm_yr": 1.0, "deceleration_threshold_mm_yr": 0.5, "random_state": 1},
        "seasonal_response": {"annual_period_days": period, "min_observations": 24, "block_size": 10, "deformation_polynomial_degree": 2, "groundwater_polynomial_degree": 1, "min_head_amplitude_m": 0.2, "min_deformation_amplitude_mm": 0.2, "lag_search_days": [0, 100], "lag_step_days": 1.0, "lag_sample_size": 10000, "random_state": 1},
        "ske": {"min": 0.0, "max": 0.01, "observation_cell_km": 4.0, "node_spacing_km_candidates": [12.0], "basis_sigma_factor": 1.5, "max_basis_neighbors": 8, "max_extrapolation_km": 35.0, "lambda_candidates": [1.0], "cv_block_km": 15.0, "aggregation_block_size": 10, "rasterize_block_size": 10},
        "storage": {"partition_model": "jiang2018", "baseline_date": "2018-01-13", "valid_fraction": 0.8, "min_observations": 24, "block_size": 10},
        "hydrostratigraphy": {"enabled": False},
        "extensometer": {"enabled": False},
    }
    path = root / "project.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path, irreversible_mm, transform, crs


def test_full_pipeline_recovers_lag_ske_and_irreversible_storage(tmp_path):
    cfg_path, irreversible_mm, transform, crs = _make_case(tmp_path)
    cfg = load_config(cfg_path)
    result = run_pipeline(cfg, stop="storage-budget")

    assert abs(result["estimate-lag"]["lag_days"] - 50.0) <= 3.0
    assert abs(result["estimate-ske"]["ske_median"] - 0.002) <= 6e-4

    storage = result["storage-budget"]
    domain = read_tif(cfg.outputs / "storage" / "storage_domain_mask.tif") > 0
    start = np.datetime64(storage["baseline_date"], "D")
    end = np.datetime64(storage["end_date"], "D")
    true_mm = irreversible_mm(end) - irreversible_mm(start)
    area = pixel_area_rows(domain.shape[0], domain.shape[1], crs, transform)[:, None]
    truth = float(np.sum((true_mm[domain] / 1000.0) * np.broadcast_to(area, domain.shape)[domain]))
    estimate = float(storage["irreversible_gws_change_m3"])
    assert abs(estimate - truth) <= 0.05 * max(1.0, abs(truth))
    assert abs(storage["total_gws_change_m3"] - storage["recoverable_gws_change_m3"] - estimate) < 1e-8 * max(1.0, abs(storage["total_gws_change_m3"]))

    annual = pd.read_csv(cfg.outputs / "storage" / "storage_annual_change.csv")
    assert len(annual) >= 3
    assert np.isfinite(annual["irreversible_gws_change_m3"]).all()

    ske_plot = plot_stage(cfg, "estimate-ske")
    storage_plot = plot_stage(cfg, "storage-budget")
    assert ske_plot["status"] == "ok"
    assert storage_plot["status"] == "ok"
    assert all(Path(x).exists() for x in ske_plot["files"] + storage_plot["files"])
