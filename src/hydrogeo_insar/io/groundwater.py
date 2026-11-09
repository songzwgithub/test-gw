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
    return wide.melt(
        id_vars=list(meta.columns),
        value_vars=date_cols,
        var_name="date",
        value_name="value",
    )


def _build_long_from_long(frame: pd.DataFrame, sec: dict[str, Any]) -> pd.DataFrame:
    fields = sec.get("fields", {})
    required = ["station_id", "date", "value", "lon", "lat"]
    missing = [key for key in required if key not in fields]
    if missing:
        raise ValueError(f"groundwater.fields missing keys: {missing}")
    source = [fields[k] for k in fields]
    return frame[source].rename(columns={v: k for k, v in fields.items()})


def _local_spike_statistics(
    dates: pd.Series,
    values: pd.Series,
    *,
    window_days: int,
    sigma: float,
    absolute_floor_m: float,
    min_neighbors: int,
    max_neighbor_gap_days: int,
):
    """Conservative local gross-error detector.

    A point is flagged only when:
      1) it has enough local observations,
      2) both nearest valid temporal neighbours are close in time, and
      3) its residual from the local median exceeds both a robust local
         scale threshold and an absolute floor.

    This targets isolated/short gross errors. It is not intended to remove
    seasonal drawdown or long-term groundwater trends.
    """
    d = pd.to_datetime(dates).reset_index(drop=True)
    x = pd.to_numeric(values, errors="coerce").reset_index(drop=True)

    s = pd.Series(x.to_numpy(float), index=pd.DatetimeIndex(d))
    window = f"{int(window_days)}D"

    local_median = s.rolling(
        window, center=True, min_periods=int(min_neighbors)
    ).median()
    residual = (s - local_median).abs()

    local_mad = residual.rolling(
        window, center=True, min_periods=int(min_neighbors)
    ).median()
    robust_sigma = 1.4826 * local_mad
    threshold = np.maximum(
        float(absolute_floor_m),
        float(sigma) * robust_sigma.to_numpy(float),
    )

    finite = np.isfinite(x.to_numpy(float))
    prev_gap = np.full(len(x), np.inf, dtype=float)
    next_gap = np.full(len(x), np.inf, dtype=float)

    valid_idx = np.flatnonzero(finite)
    for j in range(1, len(valid_idx)):
        i0, i1 = valid_idx[j - 1], valid_idx[j]
        gap = (d.iloc[i1] - d.iloc[i0]).days
        prev_gap[i1] = gap
        next_gap[i0] = gap

    well_supported = (
        (prev_gap <= int(max_neighbor_gap_days))
        & (next_gap <= int(max_neighbor_gap_days))
    )

    med = local_median.to_numpy(float)
    res = residual.to_numpy(float)
    flag = (
        finite
        & np.isfinite(med)
        & np.isfinite(threshold)
        & well_supported
        & (res > threshold)
    )
    return flag, med, res, threshold


def prepare_groundwater(cfg: ProjectConfig) -> dict[str, Any]:
    """Read groundwater and apply conservative, auditable observation QC."""
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
        long["aquifer_class"] = _standardize_aquifer(
            long["aquifer_class"], sec.get("aquifer_labels", {})
        )
    else:
        long["aquifer_class"] = "unknown"

    qc_rows: list[pd.DataFrame] = []
    zero_depth_count = 0

    variable = str(sec.get("variable", "head")).lower()
    if variable == "head":
        long["head_m"] = long["value"]
    elif variable == "depth_to_water":
        if "elevation_m" not in long.columns:
            raise ValueError(
                "elevation_m is required when groundwater.variable=depth_to_water"
            )

        zero_is_missing = bool(sec.get("zero_is_missing", False))
        if zero_is_missing:
            zero_mask = long["value"].notna() & np.isclose(
                long["value"].to_numpy(float), 0.0, atol=1e-12
            )
            zero_depth_count = int(zero_mask.sum())
            if zero_depth_count:
                z = long.loc[
                    zero_mask,
                    ["station_id", "date", "aquifer_class", "value"],
                ].copy()
                z["reason"] = "zero_depth_sentinel"
                qc_rows.append(z)
                long.loc[zero_mask, "value"] = np.nan

        long["head_m"] = long["elevation_m"] - long["value"]

        minimum_depth = float(sec.get("minimum_water_depth_m", -0.5))
        long.loc[long["value"] < minimum_depth, "head_m"] = np.nan

        if "well_depth_m" in long.columns:
            long.loc[long["value"] > long["well_depth_m"], "head_m"] = np.nan
    else:
        raise ValueError("groundwater.variable must be 'head' or 'depth_to_water'")

    spike_cfg = sec.get("qc", {}).get("spike_filter", {})
    spike_count = 0
    spike_wells = 0

    if bool(spike_cfg.get("enabled", False)):
        all_spikes = []
        for sid, idx in long.groupby("station_id").groups.items():
            g = long.loc[idx].sort_values("date")
            flag, med, residual, threshold = _local_spike_statistics(
                g["date"],
                g["head_m"],
                window_days=int(spike_cfg.get("window_days", 31)),
                sigma=float(spike_cfg.get("sigma", 8.0)),
                absolute_floor_m=float(
                    spike_cfg.get("absolute_floor_m", 15.0)
                ),
                min_neighbors=int(spike_cfg.get("min_neighbors", 7)),
                max_neighbor_gap_days=int(
                    spike_cfg.get("max_neighbor_gap_days", 7)
                ),
            )

            if flag.any():
                bad = g.loc[
                    flag,
                    ["station_id", "date", "aquifer_class", "head_m"],
                ].copy()
                bad["local_median_m"] = med[flag]
                bad["abs_residual_m"] = residual[flag]
                bad["threshold_m"] = threshold[flag]
                bad["reason"] = "local_gross_spike"
                all_spikes.append(bad)
                long.loc[g.index[flag], "head_m"] = np.nan

        if all_spikes:
            spikes = pd.concat(all_spikes, ignore_index=True)
            spike_count = int(len(spikes))
            spike_wells = int(spikes["station_id"].nunique())
            qc_rows.append(spikes)

    keep = [
        "station_id",
        "date",
        "lon",
        "lat",
        "head_m",
        "aquifer_class",
    ]
    for col in ["well_depth_m", "elevation_m"]:
        if col in long.columns:
            keep.append(col)

    out = (
        long[keep]
        .sort_values(["station_id", "date"])
        .reset_index(drop=True)
    )

    out_dir = ensure_dir(cfg.outputs / "canonical")
    out_path = out_dir / "groundwater.csv"
    out.to_csv(out_path, index=False)

    qc_path = out_dir / "groundwater_qc_report.csv"
    if qc_rows:
        report = pd.concat(qc_rows, ignore_index=True, sort=False)
    else:
        report = pd.DataFrame(
            columns=["station_id", "date", "aquifer_class", "reason"]
        )
    report.to_csv(qc_path, index=False)

    valid = out["head_m"].notna()
    manifest = {
        "status": "ok",
        "input": str(path),
        "output": str(out_path),
        "qc_report": str(qc_path),
        "rows": int(len(out)),
        "valid_head_rows": int(valid.sum()),
        "stations": int(out["station_id"].nunique()),
        "first_date": str(out["date"].min().date()) if len(out) else None,
        "last_date": str(out["date"].max().date()) if len(out) else None,
        "aquifer_counts": (
            out[["station_id", "aquifer_class"]]
            .drop_duplicates()["aquifer_class"]
            .value_counts()
            .to_dict()
        ),
        "zero_depth_sentinel_count": zero_depth_count,
        "local_gross_spike_count": spike_count,
        "local_gross_spike_wells": spike_wells,
    }
    write_json(out_dir / "groundwater_manifest.json", manifest)
    return manifest

