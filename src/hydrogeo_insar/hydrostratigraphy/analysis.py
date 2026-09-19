from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..common import aligned_raster, ensure_dir, h5_grid_metadata, read_tif, write_json, write_tif
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
    ske = read_tif(cfg.outputs / "seasonal" / "ske_effective.tif")
    irr = read_tif(cfg.outputs / "storage" / "irreversible_gws_change_equivalent_mm.tif")
    clusters_path = cfg.outputs / "regimes" / "deformation_regime_id.tif"
    clusters = read_tif(clusters_path) if clusters_path.exists() else None

    h, w = grid["height"], grid["width"]
    total_clay = np.zeros((h, w), dtype=float)
    total_sand = np.zeros((h, w), dtype=float)
    valid_count = np.zeros((h, w), dtype=np.uint16)
    corr_rows, cluster_rows = [], []

    for layer in layers:
        lid = str(layer["id"])
        sand = aligned_raster(cfg.resolve(layer["sand"]), h, w, grid["crs"], grid["transform"])
        clay = aligned_raster(cfg.resolve(layer["clay"]), h, w, grid["crs"], grid["transform"])
        valid = np.isfinite(sand) & np.isfinite(clay)
        denom = sand + clay
        frac = np.where(valid & (denom > 0), clay / denom, np.nan)
        write_tif(out_dir / f"sand_thickness_{lid}.tif", sand, grid["crs"], grid["transform"])
        write_tif(out_dir / f"clay_thickness_{lid}.tif", clay, grid["crs"], grid["transform"])
        write_tif(out_dir / f"clay_fraction_{lid}.tif", frac, grid["crs"], grid["transform"])
        total_sand[valid] += sand[valid]
        total_clay[valid] += clay[valid]
        valid_count[valid] += 1

        targets = [("ske_effective", ske), ("irreversible_gws_change_mm", irr)]
        metrics = [("sand_thickness_m", sand), ("clay_thickness_m", clay), ("clay_fraction", frac)]
        for tname, target in targets:
            for mname, metric in metrics:
                m = np.isfinite(metric) & np.isfinite(target)
                rho = float(spearmanr(metric[m], target[m]).statistic) if m.sum() >= 10 else np.nan
                corr_rows.append({"layer_id": lid, "metric": mname, "target": tname, "n": int(m.sum()), "spearman_rho": rho})
        if clusters is not None:
            for cid in sorted(int(v) for v in np.unique(clusters[np.isfinite(clusters)]) if v > 0):
                m = valid & (clusters == cid)
                cluster_rows.append({
                    "cluster_id": cid, "layer_id": lid, "n": int(m.sum()),
                    "sand_median_m": float(np.nanmedian(sand[m])) if m.any() else np.nan,
                    "clay_median_m": float(np.nanmedian(clay[m])) if m.any() else np.nan,
                    "clay_fraction_median": float(np.nanmedian(frac[m])) if m.any() else np.nan,
                })

    required = len(layers)
    complete = valid_count == required
    total_clay[~complete] = np.nan
    total_sand[~complete] = np.nan
    total_frac = np.where(complete & ((total_clay + total_sand) > 0), total_clay / (total_clay + total_sand), np.nan)
    write_tif(out_dir / "valid_layer_count.tif", valid_count, grid["crs"], grid["transform"], nodata=0, dtype="uint16")
    write_tif(out_dir / "total_clay_thickness_m.tif", total_clay.astype("float32"), grid["crs"], grid["transform"])
    write_tif(out_dir / "total_sand_thickness_m.tif", total_sand.astype("float32"), grid["crs"], grid["transform"])
    write_tif(out_dir / "total_clay_fraction.tif", total_frac.astype("float32"), grid["crs"], grid["transform"])
    pd.DataFrame(corr_rows).to_csv(out_dir / "hydrostratigraphy_correlations.csv", index=False)
    if cluster_rows:
        pd.DataFrame(cluster_rows).to_csv(out_dir / "hydrostratigraphy_by_cluster.csv", index=False)
    result = {"status": "ok", "layers": [str(x["id"]) for x in layers], "complete_layer_pixels": int(complete.sum()), "output_directory": str(out_dir)}
    write_json(out_dir / "hydrostratigraphy_summary.json", result)
    return result
