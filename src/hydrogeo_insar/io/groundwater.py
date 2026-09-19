from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common import ensure_dir, write_json
from ..config import ProjectConfig


def _read_table(path: Path, sheet_name: str | int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0 if sheet_name is None else sheet_name)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported groundwater table format: {path.suffix}")


def _date_like_columns(columns) -> list[str]:
    out: list[str] = []
    rx = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}")
    for col in columns:
        text = str(col)
        if rx.search(text) and pd.notna(pd.to_datetime(text, errors="coerce")):
            out.append(col)
    return out


def _standardize_aquifer(series: pd.Series, labels: dict[str, str]) -> pd.Series:
    if not labels:
        return series.astype(str).str.strip().str.lower()
    mapped = series.map(labels)
    return mapped.where(mapped.notna(), series.astype(str).str.strip().str.lower())


def _build_long_from_wide(frame: pd.DataFrame, sec: dict[str, Any]) -> pd.DataFrame:
    fields = sec.get("fields", {})
    required = ["station_id", "lon", "lat"]
    missing = [key for key in required if key not in fields]
    if missing:
        raise ValueError(f"groundwater.fields missing keys: {missing}")
    source_meta = [fields[k] for k in fields if k not in {"value", "date"}]
    date_cols = _date_like_columns(frame.columns)
    if not date_cols:
        raise ValueError("No date columns detected in wide groundwater table")
    meta = frame[source_meta].copy()
    rename = {v: k for k, v in fields.items() if v in meta.columns}
    meta = meta.rename(columns=rename)
    wide = pd.concat([meta, frame[date_cols]], axis=1)
    return wide.melt(id_vars=list(meta.columns), value_vars=date_cols, var_name="date", value_name="value")


def _build_long_from_long(frame: pd.DataFrame, sec: dict[str, Any]) -> pd.DataFrame:
    fields = sec.get("fields", {})
    required = ["station_id", "date", "value", "lon", "lat"]
    missing = [key for key in required if key not in fields]
    if missing:
        raise ValueError(f"groundwater.fields missing keys: {missing}")
    source = [fields[k] for k in fields]
    return frame[source].rename(columns={v: k for k, v in fields.items()})


def prepare_groundwater(cfg: ProjectConfig) -> dict[str, Any]:
    """Read groundwater using the proven v0.1 wide/long contracts, without new input formats."""
    sec = cfg.section("groundwater")
    path = cfg.resolve(sec["path"])
    frame = _read_table(path, sec.get("sheet_name"))
    fmt = str(sec.get("format", "long")).lower()
    if fmt == "wide":
        long = _build_long_from_wide(frame, sec)
    elif fmt == "long":
        long = _build_long_from_long(frame, sec)
    else:
        raise ValueError("groundwater.format must be 'wide' or 'long'")

    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    for col in ["value", "lon", "lat", "elevation_m", "well_depth_m"]:
        if col in long.columns:
            long[col] = pd.to_numeric(long[col], errors="coerce")
    long = long.dropna(subset=["station_id", "date", "lon", "lat"]).copy()
    long["station_id"] = long["station_id"].astype(str)

    if "aquifer_class" in long.columns:
        long["aquifer_class"] = _standardize_aquifer(long["aquifer_class"], sec.get("aquifer_labels", {}))
    else:
        long["aquifer_class"] = "unknown"

    variable = str(sec.get("variable", "head")).lower()
    if variable == "head":
        long["head_m"] = long["value"]
    elif variable == "depth_to_water":
        if "elevation_m" not in long.columns:
            raise ValueError("elevation_m is required when groundwater.variable=depth_to_water")
        long["head_m"] = long["elevation_m"] - long["value"]
        long.loc[long["value"] < float(sec.get("minimum_water_depth_m", -0.5)), "head_m"] = np.nan
        if "well_depth_m" in long.columns:
            long.loc[long["value"] > long["well_depth_m"], "head_m"] = np.nan
    else:
        raise ValueError("groundwater.variable must be 'head' or 'depth_to_water'")

    keep = ["station_id", "date", "lon", "lat", "head_m", "aquifer_class"]
    for col in ["well_depth_m", "elevation_m"]:
        if col in long.columns:
            keep.append(col)
    out = long[keep].sort_values(["station_id", "date"]).reset_index(drop=True)

    out_dir = ensure_dir(cfg.outputs / "canonical")
    out_path = out_dir / "groundwater.csv"
    out.to_csv(out_path, index=False)

    manifest = {
        "status": "ok",
        "input": str(path),
        "output": str(out_path),
        "rows": int(len(out)),
        "stations": int(out["station_id"].nunique()),
        "first_date": str(out["date"].min().date()) if len(out) else None,
        "last_date": str(out["date"].max().date()) if len(out) else None,
        "aquifer_counts": out[["station_id", "aquifer_class"]].drop_duplicates()["aquifer_class"].value_counts().to_dict(),
    }
    write_json(out_dir / "groundwater_manifest.json", manifest)
    return manifest
