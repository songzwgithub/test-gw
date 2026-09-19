from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import rasterio

from ..common import days_to_dates, ensure_dir, h5_grid_metadata, pixel_area_rows, write_json, write_tif
from ..config import ProjectConfig


def _nearest_index(dates: np.ndarray, target: str | np.datetime64 | None, default: int) -> int:
    if target is None:
        return default
    t = np.datetime64(str(target), "D")
    return int(np.argmin(np.abs(dates.astype("datetime64[D]") - t)))


def compute_storage_budget(cfg: ProjectConfig) -> dict[str, Any]:
    sec = cfg.section("storage")
    insar_path = cfg.outputs / "canonical" / "insar_stack.h5"
    head_path = cfg.outputs / "groundwater" / "groundwater_field.h5"
    ske_path = cfg.outputs / "seasonal" / "ske_effective.tif"
    cluster_path = cfg.outputs / "regimes" / "deformation_regime_id.tif"
    out_dir = ensure_dir(cfg.outputs / "storage")

    grid = h5_grid_metadata(insar_path)
    area_rows = pixel_area_rows(grid["height"], grid["width"], grid["crs"], grid["transform"])
    with rasterio.open(ske_path) as src:
        ske = src.read(1).astype(float)
        if src.nodata is not None:
            ske[ske == src.nodata] = np.nan
    clusters = None
    if cluster_path.exists():
        with rasterio.open(cluster_path) as src:
            clusters = src.read(1).astype(int)

    with h5py.File(insar_path, "r") as ih5, h5py.File(head_path, "r") as hh5:
        dates_i = days_to_dates(ih5["date_days"][:])
        dates_h = days_to_dates(hh5["date_days"][:])
        if len(dates_i) != len(dates_h) or np.any(dates_i != dates_h):
            raise ValueError("InSAR and groundwater field dates must match")
        dates = dates_i
        ib = _nearest_index(dates, sec.get("baseline_date"), 0)
        ie = _nearest_index(dates, sec.get("end_date"), len(dates)-1)

        disp0 = ih5["displacement_mm"][ib].astype(float)
        head0 = hh5["head_anomaly_m"][ib].astype(float)
        rows = []
        cluster_rows = []
        unique_clusters = [] if clusters is None else sorted(int(v) for v in np.unique(clusters) if v > 0)

        for it, date in enumerate(dates):
            disp = ih5["displacement_mm"][it].astype(float)
            head = hh5["head_anomaly_m"][it].astype(float)
            d_total_m = (disp - disp0) / 1000.0
            d_head_m = head - head0
            d_recoverable_m = ske * d_head_m
            d_irreversible_m = d_total_m - d_recoverable_m
            finite = np.isfinite(d_total_m) & np.isfinite(d_recoverable_m)
            area = area_rows[:, None]
            vt = float(np.nansum(np.where(finite, d_total_m * area, np.nan)))
            vr = float(np.nansum(np.where(finite, d_recoverable_m * area, np.nan)))
            vi = float(vt - vr)
            rows.append({
                "date": str(date),
                "total_storage_change_m3": vt,
                "recoverable_storage_change_m3": vr,
                "irreversible_storage_change_m3": vi,
                "irreversible_storage_loss_m3": max(0.0, -vi),
            })
            if clusters is not None:
                for cid in unique_clusters:
                    m = finite & (clusters == cid)
                    cvt = float(np.nansum(np.where(m, d_total_m * area, np.nan)))
                    cvr = float(np.nansum(np.where(m, d_recoverable_m * area, np.nan)))
                    cvi = cvt - cvr
                    cluster_rows.append({
                        "date": str(date),
                        "cluster_id": cid,
                        "total_storage_change_m3": cvt,
                        "recoverable_storage_change_m3": cvr,
                        "irreversible_storage_change_m3": cvi,
                        "irreversible_storage_loss_m3": max(0.0, -cvi),
                    })

        # Final spatial fields.
        disp_end = ih5["displacement_mm"][ie].astype(float)
        head_end = hh5["head_anomaly_m"][ie].astype(float)
        total_mm = disp_end - disp0
        recoverable_mm = ske * (head_end - head0) * 1000.0
        irreversible_mm = total_mm - recoverable_mm

    write_tif(out_dir / "total_storage_equivalent_mm.tif", total_mm.astype("float32"), grid["crs"], grid["transform"])
    write_tif(out_dir / "recoverable_storage_equivalent_mm.tif", recoverable_mm.astype("float32"), grid["crs"], grid["transform"])
    write_tif(out_dir / "irreversible_storage_equivalent_mm.tif", irreversible_mm.astype("float32"), grid["crs"], grid["transform"])
    write_tif(out_dir / "irreversible_storage_loss_mm.tif", np.maximum(0.0, -irreversible_mm).astype("float32"), grid["crs"], grid["transform"])

    ts = pd.DataFrame(rows)
    ts.to_csv(out_dir / "storage_budget_timeseries.csv", index=False)
    if cluster_rows:
        pd.DataFrame(cluster_rows).to_csv(out_dir / "storage_budget_by_cluster.csv", index=False)

    annual = ts.copy()
    annual["date"] = pd.to_datetime(annual["date"])
    annual["year"] = annual["date"].dt.year
    annual = annual.sort_values("date").groupby("year", as_index=False).tail(1)
    annual.to_csv(out_dir / "storage_budget_annual.csv", index=False)

    final = ts.iloc[ie]
    result = {
        "status": "ok",
        "baseline_date": str(dates[ib]),
        "end_date": str(dates[ie]),
        "total_storage_change_m3": float(final["total_storage_change_m3"]),
        "recoverable_storage_change_m3": float(final["recoverable_storage_change_m3"]),
        "irreversible_storage_change_m3": float(final["irreversible_storage_change_m3"]),
        "irreversible_storage_loss_m3": float(final["irreversible_storage_loss_m3"]),
        "identity": "V_total = V_recoverable + V_irreversible",
    }
    write_json(out_dir / "storage_budget_summary.json", result)
    return result
