from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import rasterio

from .common import ensure_dir, write_json
from .config import ProjectConfig


def _read(path):
    with rasterio.open(path) as src:
        a = src.read(1).astype(float)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
    return a


def synthesize(cfg: ProjectConfig) -> dict[str, Any]:
    out_dir = ensure_dir(cfg.outputs / "synthesis")
    cluster_path = cfg.outputs / "regimes" / "deformation_regime_id.tif"
    if not cluster_path.exists():
        return {"status": "skipped_no_regimes"}
    with rasterio.open(cluster_path) as src:
        clusters = src.read(1).astype(int)
    end_rate = _read(cfg.outputs / "deformation" / "end_rate_mm_yr.tif")
    rate_change = _read(cfg.outputs / "deformation" / "rate_change_mm_yr.tif")
    ske = _read(cfg.outputs / "seasonal" / "ske_effective.tif")
    loss = _read(cfg.outputs / "storage" / "irreversible_storage_loss_mm.tif")
    recoverable = _read(cfg.outputs / "storage" / "recoverable_storage_equivalent_mm.tif")
    head_trend = _read(cfg.outputs / "seasonal" / "head_trend_m_yr.tif")

    clay_total = None
    clay_path = cfg.outputs / "hydrostratigraphy" / "total_clay_thickness_m.tif"
    if clay_path.exists():
        clay_total = _read(clay_path)

    regime_names = {}
    summary_csv = cfg.outputs / "regimes" / "cluster_summary.csv"
    if summary_csv.exists():
        tmp = pd.read_csv(summary_csv)
        regime_names = {int(r.cluster_id): str(r.suggested_label) for _, r in tmp.iterrows()}

    rows = []
    for cid in sorted(int(v) for v in np.unique(clusters) if v > 0):
        m = clusters == cid
        row = {
            "cluster_id": cid,
            "regime": regime_names.get(cid, f"cluster_{cid}"),
            "pixel_count": int(m.sum()),
            "end_rate_median_mm_yr": float(np.nanmedian(end_rate[m])),
            "rate_change_median_mm_yr": float(np.nanmedian(rate_change[m])),
            "head_trend_median_m_yr": float(np.nanmedian(head_trend[m])),
            "ske_median": float(np.nanmedian(ske[m])),
            "recoverable_storage_equivalent_median_mm": float(np.nanmedian(recoverable[m])),
            "irreversible_storage_loss_median_mm": float(np.nanmedian(loss[m])),
        }
        if clay_total is not None:
            row["total_clay_thickness_median_m"] = float(np.nanmedian(clay_total[m]))
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "aquifer_system_response_by_regime.csv", index=False)
    result = {"status": "ok", "regimes": rows, "output": str(out_dir / "aquifer_system_response_by_regime.csv")}
    write_json(out_dir / "synthesis_summary.json", result)
    return result
