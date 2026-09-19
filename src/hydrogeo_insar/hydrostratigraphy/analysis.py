from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import rasterio
from scipy.stats import spearmanr

from ..common import aligned_raster, ensure_dir, h5_grid_metadata, write_json, write_tif
from ..config import ProjectConfig


def analyze_hydrostratigraphy(cfg: ProjectConfig) -> dict[str, Any]:
    sec = cfg.section("hydrostratigraphy")
    if not sec.get("enabled", False):
        return {"status": "skipped"}
    layers = sec.get("layers", [])
    if not layers:
        raise ValueError("hydrostratigraphy.layers is empty")

    grid = h5_grid_metadata(cfg.outputs / "canonical" / "insar_stack.h5")
    out_dir = ensure_dir(cfg.outputs / "hydrostratigraphy")
    with rasterio.open(cfg.outputs / "seasonal" / "ske_effective.tif") as src:
        ske = src.read(1).astype(float)
        if src.nodata is not None:
            ske[ske == src.nodata] = np.nan
    with rasterio.open(cfg.outputs / "storage" / "irreversible_storage_loss_mm.tif") as src:
        loss = src.read(1).astype(float)
        if src.nodata is not None:
            loss[loss == src.nodata] = np.nan
    cluster_path = cfg.outputs / "regimes" / "deformation_regime_id.tif"
    clusters = None
    if cluster_path.exists():
        with rasterio.open(cluster_path) as src:
            clusters = src.read(1).astype(int)

    correlation_rows = []
    cluster_rows = []
    total_clay = np.zeros((grid["height"], grid["width"]), dtype="float64")
    total_sand = np.zeros_like(total_clay)
    total_valid = np.zeros_like(total_clay, dtype=bool)

    for layer in layers:
        lid = str(layer["id"])
        sand = aligned_raster(cfg.resolve(layer["sand"]), grid["height"], grid["width"], grid["crs"], grid["transform"])
        clay = aligned_raster(cfg.resolve(layer["clay"]), grid["height"], grid["width"], grid["crs"], grid["transform"])
        denom = sand + clay
        clay_fraction = np.where(np.isfinite(denom) & (denom > 0), clay / denom, np.nan)
        write_tif(out_dir / f"sand_thickness_{lid}.tif", sand, grid["crs"], grid["transform"])
        write_tif(out_dir / f"clay_thickness_{lid}.tif", clay, grid["crs"], grid["transform"])
        write_tif(out_dir / f"clay_fraction_{lid}.tif", clay_fraction, grid["crs"], grid["transform"])
        v = np.isfinite(sand) & np.isfinite(clay)
        total_clay[v] += clay[v]
        total_sand[v] += sand[v]
        total_valid |= v

        for target_name, target in [("ske_effective", ske), ("irreversible_storage_loss_mm", loss)]:
            for metric_name, metric in [("sand_thickness_m", sand), ("clay_thickness_m", clay), ("clay_fraction", clay_fraction)]:
                m = np.isfinite(metric) & np.isfinite(target)
                rho, p = spearmanr(metric[m], target[m]) if m.sum() >= 10 else (np.nan, np.nan)
                correlation_rows.append({
                    "layer_id": lid,
                    "metric": metric_name,
                    "target": target_name,
                    "n": int(m.sum()),
                    "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                    "p_value": float(p) if np.isfinite(p) else np.nan,
                })
        if clusters is not None:
            for cid in sorted(int(v) for v in np.unique(clusters) if v > 0):
                m = (clusters == cid) & np.isfinite(sand) & np.isfinite(clay)
                cluster_rows.append({
                    "cluster_id": cid,
                    "layer_id": lid,
                    "n": int(m.sum()),
                    "sand_median_m": float(np.nanmedian(sand[m])) if m.any() else np.nan,
                    "clay_median_m": float(np.nanmedian(clay[m])) if m.any() else np.nan,
                    "clay_fraction_median": float(np.nanmedian(clay_fraction[m])) if m.any() else np.nan,
                })

    total_clay[~total_valid] = np.nan
    total_sand[~total_valid] = np.nan
    total_frac = total_clay / np.maximum(total_clay + total_sand, 1e-12)
    write_tif(out_dir / "total_clay_thickness_m.tif", total_clay.astype("float32"), grid["crs"], grid["transform"])
    write_tif(out_dir / "total_sand_thickness_m.tif", total_sand.astype("float32"), grid["crs"], grid["transform"])
    write_tif(out_dir / "total_clay_fraction.tif", total_frac.astype("float32"), grid["crs"], grid["transform"])
    pd.DataFrame(correlation_rows).to_csv(out_dir / "hydrostratigraphy_correlations.csv", index=False)
    if cluster_rows:
        pd.DataFrame(cluster_rows).to_csv(out_dir / "hydrostratigraphy_by_cluster.csv", index=False)
    result = {"status": "ok", "layers": [str(x["id"]) for x in layers], "output_directory": str(out_dir)}
    write_json(out_dir / "hydrostratigraphy_summary.json", result)
    return result
