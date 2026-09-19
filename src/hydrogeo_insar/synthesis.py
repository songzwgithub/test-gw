from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import ensure_dir, read_tif, write_json
from .config import ProjectConfig


def synthesize(cfg: ProjectConfig) -> dict[str, Any]:
    out_dir = ensure_dir(cfg.outputs / "synthesis")
    cluster_path = cfg.outputs / "regimes" / "deformation_regime_id.tif"
    if not cluster_path.exists():
        return {"status": "skipped_no_regimes"}
    clusters = read_tif(cluster_path)
    end_rate = read_tif(cfg.outputs / "deformation" / "end_rate_mm_yr.tif")
    ske = read_tif(cfg.outputs / "seasonal" / "ske_effective.tif")
    irr = read_tif(cfg.outputs / "storage" / "irreversible_gws_change_equivalent_mm.tif")
    rec = read_tif(cfg.outputs / "storage" / "recoverable_gws_change_equivalent_mm.tif")
    htrend = read_tif(cfg.outputs / "seasonal" / "head_linear_m_yr.tif")
    storage_domain = read_tif(cfg.outputs / "storage" / "storage_domain_mask.tif") > 0

    clay_total = None
    clay_path = cfg.outputs / "hydrostratigraphy" / "total_clay_thickness_m.tif"
    if clay_path.exists():
        clay_total = read_tif(clay_path)

    names = {}
    summary_csv = cfg.outputs / "regimes" / "cluster_summary.csv"
    if summary_csv.exists():
        tmp = pd.read_csv(summary_csv)
        names = {int(r.cluster_id): str(r.suggested_label) for _, r in tmp.iterrows()}

    rows = []
    for cid in sorted(int(v) for v in np.unique(clusters[np.isfinite(clusters)]) if v > 0):
        m = (clusters == cid) & storage_domain
        row = {
            "cluster_id": cid,
            "regime": names.get(cid, f"cluster_{cid}"),
            "storage_domain_pixel_count": int(m.sum()),
            "end_rate_median_mm_yr": float(np.nanmedian(end_rate[m])) if m.any() else np.nan,
            "head_linear_trend_median_m_yr": float(np.nanmedian(htrend[m])) if m.any() else np.nan,
            "ske_median": float(np.nanmedian(ske[m])) if m.any() else np.nan,
            "recoverable_gws_change_median_mm": float(np.nanmedian(rec[m])) if m.any() else np.nan,
            "irreversible_gws_change_median_mm": float(np.nanmedian(irr[m])) if m.any() else np.nan,
        }
        if clay_total is not None:
            row["total_clay_thickness_median_m"] = float(np.nanmedian(clay_total[m])) if m.any() else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "aquifer_system_response_by_regime.csv", index=False)
    result = {"status": "ok", "regimes": rows, "output": str(out_dir / "aquifer_system_response_by_regime.csv")}
    write_json(out_dir / "synthesis_summary.json", result)
    return result
