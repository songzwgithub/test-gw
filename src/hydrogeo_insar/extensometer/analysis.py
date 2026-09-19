from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import pandas as pd
import rasterio

from ..common import days_to_dates, ensure_dir, h5_grid_metadata, write_json
from ..config import ProjectConfig


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
    for col in ["depth_top_m", "depth_bottom_m", "compaction_mm", "lon", "lat"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "depth_top_m", "depth_bottom_m", "compaction_mm"])

    out_dir = ensure_dir(cfg.outputs / "extensometer")
    out.to_csv(out_dir / "extensometer_canonical.csv", index=False)
    interval = out.groupby(["depth_top_m", "depth_bottom_m"], as_index=False).agg(
        first_date=("date", "min"), last_date=("date", "max"),
        initial_mm=("compaction_mm", "first"), final_mm=("compaction_mm", "last")
    )
    interval["change_mm"] = interval["final_mm"] - interval["initial_mm"]
    total = float(interval["change_mm"].sum())
    interval["fraction_of_total"] = interval["change_mm"] / total if total != 0 else np.nan
    interval.to_csv(out_dir / "depth_interval_contributions.csv", index=False)

    # Optional collocation with InSAR and groundwater if lon/lat are provided in config.
    lon = sec.get("lon")
    lat = sec.get("lat")
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
                dates = days_to_dates(ih5["date_days"][:])
                ts = pd.DataFrame({
                    "date": dates.astype(str),
                    "insar_displacement_mm": ih5["displacement_mm"][:, row, col],
                    "groundwater_head_anomaly_m": gh5["head_anomaly_m"][:, row, col],
                })
                ts.to_csv(out_dir / "collocated_insar_groundwater.csv", index=False)

    result = {"status": "ok", "intervals": int(len(interval)), "total_interval_change_mm": total, "output_directory": str(out_dir)}
    write_json(out_dir / "extensometer_summary.json", result)
    return result
