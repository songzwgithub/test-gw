from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import pandas as pd
import rasterio

from ..common import days_to_dates, ensure_dir, h5_grid_metadata, write_json
from ..config import ProjectConfig


def _canonicalize_intervals(frame: pd.DataFrame, measurement_type: str) -> pd.DataFrame:
    out = frame.sort_values(["date", "depth_bottom_m", "depth_top_m"]).copy()
    if measurement_type == "interval_compaction":
        return out
    if measurement_type != "cumulative_marker_displacement":
        raise ValueError("extensometer.measurement_type must be interval_compaction or cumulative_marker_displacement")
    rows = []
    for date, g in out.groupby("date"):
        g = g.sort_values("depth_bottom_m")
        prev_depth, prev_value = 0.0, 0.0
        for _, r in g.iterrows():
            bottom = float(r["depth_bottom_m"])
            cumulative = float(r["compaction_mm"])
            rows.append({"date": date, "depth_top_m": prev_depth, "depth_bottom_m": bottom, "compaction_mm": cumulative - prev_value})
            prev_depth, prev_value = bottom, cumulative
    return pd.DataFrame(rows)


def analyze_extensometer(cfg: ProjectConfig) -> dict[str, Any]:
    sec = cfg.section("extensometer")
    if not sec.get("enabled", False):
        return {"status": "skipped"}
    path = cfg.resolve(sec["path"])
    frame = pd.read_excel(path, sheet_name=sec.get("sheet_name", 0)) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    fields = sec.get("fields", {})
    required = ["date", "depth_top_m", "depth_bottom_m", "compaction_mm"]
    missing = [k for k in required if k not in fields]
    if missing:
        raise ValueError(f"extensometer.fields missing: {missing}")
    out = frame[[fields[k] for k in fields]].rename(columns={v: k for k, v in fields.items()}).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["depth_top_m", "depth_bottom_m", "compaction_mm"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=required).sort_values(["depth_top_m", "depth_bottom_m", "date"]).reset_index(drop=True)
    out = _canonicalize_intervals(out, str(sec.get("measurement_type", "interval_compaction")))
    out = out.sort_values(["depth_top_m", "depth_bottom_m", "date"]).reset_index(drop=True)

    out_dir = ensure_dir(cfg.outputs / "extensometer")
    out.to_csv(out_dir / "extensometer_canonical.csv", index=False)
    interval = out.groupby(["depth_top_m", "depth_bottom_m"], as_index=False).agg(
        first_date=("date", "first"), last_date=("date", "last"),
        initial_mm=("compaction_mm", "first"), final_mm=("compaction_mm", "last"),
    )
    interval["change_mm"] = interval["final_mm"] - interval["initial_mm"]
    total = float(interval["change_mm"].sum())
    interval["fraction_of_total"] = interval["change_mm"] / total if total != 0 else np.nan
    interval.to_csv(out_dir / "depth_interval_contributions.csv", index=False)

    lon, lat = sec.get("lon"), sec.get("lat")
    if lon is not None and lat is not None:
        stack_path = cfg.outputs / "canonical" / "insar_stack.h5"
        grid = h5_grid_metadata(stack_path)
        from pyproj import CRS, Transformer
        if CRS.from_user_input(grid["crs"]).is_geographic:
            x, y = float(lon), float(lat)
        else:
            tr = Transformer.from_crs("EPSG:4326", grid["crs"], always_xy=True)
            x, y = tr.transform(float(lon), float(lat))
        row, col = rasterio.transform.rowcol(grid["transform"], x, y)
        if 0 <= row < grid["height"] and 0 <= col < grid["width"]:
            with h5py.File(stack_path, "r") as ih5, h5py.File(cfg.outputs / "groundwater" / "groundwater_field.h5", "r") as gh5:
                idates = days_to_dates(ih5["date_days"][:]); hdates = days_to_dates(gh5["date_days"][:])
                dates, ii, hi = np.intersect1d(idates, hdates, assume_unique=True, return_indices=True)
                ts = pd.DataFrame({
                    "date": dates.astype(str),
                    "insar_displacement_mm": ih5["displacement_mm"][ii, row, col],
                    "groundwater_head_anomaly_m": gh5["head_anomaly_m"][hi, row, col],
                })
                ts.to_csv(out_dir / "collocated_insar_groundwater.csv", index=False)

    result = {"status": "ok", "measurement_type": sec.get("measurement_type", "interval_compaction"), "intervals": int(len(interval)), "total_interval_change_mm": total, "output_directory": str(out_dir)}
    write_json(out_dir / "extensometer_summary.json", result)
    return result
