from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import rowcol, xy as raster_xy

from ..common import days_to_dates, ensure_dir, h5_grid_metadata, write_json
from ..config import ProjectConfig


def _canonicalize_intervals(frame: pd.DataFrame, measurement_type: str) -> pd.DataFrame:
    """Canonicalize the legacy long-form interval input.

    Output compaction_mm is positive for compression when the input already
    follows that convention. The legacy behavior is retained unchanged.
    """
    out = frame.sort_values(["date", "depth_bottom_m", "depth_top_m"]).copy()
    if measurement_type == "interval_compaction":
        return out
    if measurement_type != "cumulative_marker_displacement":
        raise ValueError(
            "extensometer.measurement_type must be interval_compaction or "
            "cumulative_marker_displacement for canonical_interval input"
        )
    rows = []
    for date, g in out.groupby("date"):
        g = g.sort_values("depth_bottom_m")
        prev_depth, prev_value = 0.0, 0.0
        for _, r in g.iterrows():
            bottom = float(r["depth_bottom_m"])
            cumulative = float(r["compaction_mm"])
            rows.append(
                {
                    "date": date,
                    "depth_top_m": prev_depth,
                    "depth_bottom_m": bottom,
                    "compaction_mm": cumulative - prev_value,
                }
            )
            prev_depth, prev_value = bottom, cumulative
    return pd.DataFrame(rows)


def _read_frame(path: Path, sheet_name=0, encoding: str | None = None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    return pd.read_csv(path, encoding=encoding or "utf-8-sig")


def _marker_sign(value: str) -> float:
    text = str(value).strip().lower()
    if text in {"uplift", "up", "upward", "positive_up"}:
        return 1.0
    if text in {"subsidence", "down", "downward", "positive_down"}:
        return -1.0
    raise ValueError("extensometer.marker_positive must be uplift or subsidence")


def _read_marker_table(path: Path, sec: dict[str, Any]):
    """Read a marker table of the form used by the Hengshui extensometer.

    Example::

        标孔,F1,F2,F3,F4
        深度（m）,41,150,267,401
        2013/1/15,-176.32,-115.72,-59.17,0
        ...

    Marker displacement is normalized to positive-upward. Layer compaction
    between adjacent markers is then:

        compaction(z1,z2) = u(z2) - u(z1)

    so positive values mean compression when the shallower marker settles more
    than the deeper marker.
    """
    frame = _read_frame(path, sec.get("sheet_name", 0), sec.get("encoding"))
    date_col = sec.get("date_column", frame.columns[0])
    marker_cols = sec.get("marker_columns") or [c for c in frame.columns if c != date_col]
    depth_label = str(sec.get("depth_row_label", "深度（m）")).strip()

    labels = frame[date_col].astype(str).str.strip()
    hit = frame.loc[labels == depth_label]
    if hit.empty:
        raise ValueError(f"Extensometer marker table has no depth row {depth_label!r}")
    depth_row = hit.iloc[0]

    markers = []
    for col in marker_cols:
        depth = pd.to_numeric(pd.Series([depth_row[col]]), errors="coerce").iloc[0]
        if np.isfinite(depth):
            markers.append((str(col), float(depth)))
    if len(markers) < 2:
        raise ValueError("At least two marker depths are required")
    markers.sort(key=lambda x: x[1])

    data = frame.loc[labels != depth_label, [date_col] + [m[0] for m in markers]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    sign = _marker_sign(sec.get("marker_positive", "uplift"))
    for col, _depth in markers:
        data[col] = pd.to_numeric(data[col], errors="coerce") * sign
    data = data.dropna(subset=[m[0] for m in markers], how="all")

    marker_rows = []
    for _, r in data.iterrows():
        for marker_id, depth in markers:
            marker_rows.append(
                {
                    "date": r[date_col],
                    "marker_id": marker_id,
                    "depth_m": depth,
                    "marker_displacement_mm": float(r[marker_id]) if np.isfinite(r[marker_id]) else np.nan,
                }
            )
    marker_long = pd.DataFrame(marker_rows)

    interval_rows = []
    profile_rows = []
    shallow_id, shallow_depth = markers[0]
    deep_id, deep_depth = markers[-1]
    for _, r in data.iterrows():
        for (upper_id, upper_depth), (lower_id, lower_depth) in zip(markers[:-1], markers[1:]):
            u_upper = float(r[upper_id]) if np.isfinite(r[upper_id]) else np.nan
            u_lower = float(r[lower_id]) if np.isfinite(r[lower_id]) else np.nan
            comp = u_lower - u_upper if np.isfinite(u_upper) and np.isfinite(u_lower) else np.nan
            interval_rows.append(
                {
                    "date": r[date_col],
                    "depth_top_m": upper_depth,
                    "depth_bottom_m": lower_depth,
                    "upper_marker": upper_id,
                    "lower_marker": lower_id,
                    "compaction_mm": comp,
                }
            )
        u_shallow = float(r[shallow_id]) if np.isfinite(r[shallow_id]) else np.nan
        u_deep = float(r[deep_id]) if np.isfinite(r[deep_id]) else np.nan
        profile_rows.append(
            {
                "date": r[date_col],
                "monitoring_top_m": shallow_depth,
                "monitoring_bottom_m": deep_depth,
                "profile_compaction_mm": u_deep - u_shallow
                if np.isfinite(u_shallow) and np.isfinite(u_deep)
                else np.nan,
            }
        )

    return marker_long, pd.DataFrame(interval_rows), pd.DataFrame(profile_rows), markers


def _profile_from_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    out = (
        intervals.groupby("date", as_index=False)["compaction_mm"]
        .sum(min_count=1)
        .rename(columns={"compaction_mm": "profile_compaction_mm"})
    )
    if not intervals.empty:
        out["monitoring_top_m"] = float(intervals["depth_top_m"].min())
        out["monitoring_bottom_m"] = float(intervals["depth_bottom_m"].max())
    return out


def _interval_contributions(intervals: pd.DataFrame) -> pd.DataFrame:
    g = intervals.sort_values(["depth_top_m", "depth_bottom_m", "date"])
    table = g.groupby(["depth_top_m", "depth_bottom_m"], as_index=False).agg(
        first_date=("date", "first"),
        last_date=("date", "last"),
        initial_mm=("compaction_mm", "first"),
        final_mm=("compaction_mm", "last"),
    )
    table["change_mm"] = table["final_mm"] - table["initial_mm"]
    total = float(table["change_mm"].sum()) if len(table) else np.nan
    table["fraction_of_profile_change"] = table["change_mm"] / total if np.isfinite(total) and total != 0 else np.nan
    return table


def _coverage(dates: np.ndarray | pd.Series):
    x = pd.to_datetime(np.asarray(dates), errors="coerce")
    x = x[~pd.isna(x)]
    if len(x) == 0:
        return None, None
    return pd.Timestamp(x.min()), pd.Timestamp(x.max())


def _overlap(a0, a1, b0, b1):
    if None in {a0, a1, b0, b1}:
        return None, None
    start, end = max(a0, b0), min(a1, b1)
    return (start, end) if start <= end else (None, None)


def _auto_utm_epsg(lon: float, lat: float) -> str:
    zone = int((lon + 180.0) // 6.0) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def _neighborhood_window(grid: dict[str, Any], lon: float, lat: float, radius_m: float):
    """Return raster window and a boolean circular mask around one site."""
    transform = grid["transform"]
    grid_crs = CRS.from_user_input(grid["crs"])
    to_grid = Transformer.from_crs("EPSG:4326", grid_crs, always_xy=True)
    gx0, gy0 = to_grid.transform(float(lon), float(lat))
    r0, c0 = rowcol(transform, gx0, gy0)
    if not (0 <= r0 < grid["height"] and 0 <= c0 < grid["width"]):
        return None

    if radius_m <= 0:
        return int(r0), int(r0 + 1), int(c0), int(c0 + 1), np.ones((1, 1), dtype=bool)

    local_crs = _auto_utm_epsg(float(lon), float(lat))
    grid_to_local = Transformer.from_crs(grid_crs, local_crs, always_xy=True)
    tx, ty = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True).transform(float(lon), float(lat))

    x00, y00 = raster_xy(transform, int(r0), int(c0), offset="center")
    x01, y01 = raster_xy(transform, int(r0), min(int(c0 + 1), grid["width"] - 1), offset="center")
    x10, y10 = raster_xy(transform, min(int(r0 + 1), grid["height"] - 1), int(c0), offset="center")
    p00 = np.asarray(grid_to_local.transform(x00, y00), dtype=float)
    p01 = np.asarray(grid_to_local.transform(x01, y01), dtype=float)
    p10 = np.asarray(grid_to_local.transform(x10, y10), dtype=float)
    pix = [np.linalg.norm(p01 - p00), np.linalg.norm(p10 - p00)]
    pix = [v for v in pix if np.isfinite(v) and v > 0]
    pixel_m = min(pix) if pix else max(float(radius_m), 1.0)
    pad = int(np.ceil(float(radius_m) / pixel_m)) + 2

    rr0 = max(0, int(r0) - pad)
    rr1 = min(grid["height"], int(r0) + pad + 1)
    cc0 = max(0, int(c0) - pad)
    cc1 = min(grid["width"], int(c0) + pad + 1)
    rr, cc = np.meshgrid(np.arange(rr0, rr1), np.arange(cc0, cc1), indexing="ij")
    xs, ys = raster_xy(transform, rr.ravel(), cc.ravel(), offset="center")
    mx, my = grid_to_local.transform(np.asarray(xs), np.asarray(ys))
    dist = np.hypot(np.asarray(mx) - tx, np.asarray(my) - ty).reshape(rr.shape)
    mask = dist <= float(radius_m)
    if not mask.any():
        mask[int(r0) - rr0, int(c0) - cc0] = True
    return rr0, rr1, cc0, cc1, mask


def _spatial_median_series(h5_path: Path, dataset: str, window):
    rr0, rr1, cc0, cc1, mask = window
    with h5py.File(h5_path, "r") as h5:
        arr = h5[dataset][:, rr0:rr1, cc0:cc1].astype(float)
        dates = days_to_dates(h5["date_days"][:])
    flat = arr[:, mask]
    values = np.nanmedian(flat, axis=1)
    return dates, values


def _interp_with_gap(source_dates, source_values, target_dates, max_gap_days: int = 45):
    src_d = np.asarray(source_dates, dtype="datetime64[D]")
    src_v = np.asarray(source_values, dtype=float)
    qry = np.asarray(target_dates, dtype="datetime64[D]")
    good = np.isfinite(src_v)
    src_d, src_v = src_d[good], src_v[good]
    out = np.full(len(qry), np.nan, dtype=float)
    if len(src_d) == 0:
        return out
    s = src_d.astype(np.int64)
    q = qry.astype(np.int64)
    pos = np.searchsorted(s, q, side="left")
    exact = (pos < len(s))
    exact[exact] &= s[pos[exact]] == q[exact]
    out[exact] = src_v[pos[exact]]
    bracket = (~exact) & (pos > 0) & (pos < len(s))
    if bracket.any():
        li = pos[bracket] - 1
        ri = pos[bracket]
        gap = s[ri] - s[li]
        ok = gap <= int(max_gap_days)
        qi = q[bracket]
        vals = src_v[li] + (src_v[ri] - src_v[li]) * (qi - s[li]) / np.maximum(gap, 1)
        temp = np.full(np.sum(bracket), np.nan, dtype=float)
        temp[ok] = vals[ok]
        out[bracket] = temp
    return out


def _write_overlap_products(cfg: ProjectConfig, sec: dict[str, Any], profile: pd.DataFrame, out_dir: Path):
    stack_path = cfg.outputs / "canonical" / "insar_stack.h5"
    head_path = cfg.outputs / "groundwater" / "groundwater_field.h5"

    e0, e1 = _coverage(profile["date"])
    i0 = i1 = g0 = g1 = None
    if stack_path.exists():
        with h5py.File(stack_path, "r") as h5:
            i0, i1 = _coverage(days_to_dates(h5["date_days"][:]))
    if head_path.exists():
        with h5py.File(head_path, "r") as h5:
            g0, g1 = _coverage(days_to_dates(h5["date_days"][:]))

    ei0, ei1 = _overlap(e0, e1, i0, i1)
    tri0, tri1 = _overlap(ei0, ei1, g0, g1)
    summary = {
        "extensometer_start": None if e0 is None else str(e0.date()),
        "extensometer_end": None if e1 is None else str(e1.date()),
        "insar_start": None if i0 is None else str(i0.date()),
        "insar_end": None if i1 is None else str(i1.date()),
        "groundwater_start": None if g0 is None else str(g0.date()),
        "groundwater_end": None if g1 is None else str(g1.date()),
        "insar_extensometer_overlap_start": None if ei0 is None else str(ei0.date()),
        "insar_extensometer_overlap_end": None if ei1 is None else str(ei1.date()),
        "three_source_overlap_start": None if tri0 is None else str(tri0.date()),
        "three_source_overlap_end": None if tri1 is None else str(tri1.date()),
    }
    write_json(out_dir / "extensometer_overlap_summary.json", summary)

    lon, lat = sec.get("lon"), sec.get("lat")
    if lon is None or lat is None or not stack_path.exists() or not head_path.exists():
        return summary

    grid = h5_grid_metadata(stack_path)
    radius = float(sec.get("comparison_radius_m", 500.0))
    window = _neighborhood_window(grid, float(lon), float(lat), radius)
    if window is None:
        return summary

    idates, its = _spatial_median_series(stack_path, "displacement_mm", window)
    hdates, hts = _spatial_median_series(head_path, "head_anomaly_m", window)
    common, ii, hi = np.intersect1d(idates, hdates, assume_unique=True, return_indices=True)
    pd.DataFrame(
        {
            "date": common.astype(str),
            "insar_displacement_mm": its[ii],
            "groundwater_head_anomaly_m": hts[hi],
        }
    ).to_csv(out_dir / "collocated_insar_groundwater_full.csv", index=False)

    if tri0 is None or tri1 is None:
        return summary
    p = profile.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p[(p["date"] >= tri0) & (p["date"] <= tri1)].copy()
    if p.empty:
        return summary

    td = p["date"].to_numpy(dtype="datetime64[D]")
    max_gap = int(sec.get("overlap_max_gap_days", 45))
    p["insar_displacement_mm"] = _interp_with_gap(idates, its, td, max_gap)
    p["groundwater_head_anomaly_m"] = _interp_with_gap(hdates, hts, td, max_gap)
    good = np.isfinite(p["insar_displacement_mm"]) & np.isfinite(p["groundwater_head_anomaly_m"]) & np.isfinite(p["profile_compaction_mm"])
    p = p.loc[good].copy()
    if not p.empty:
        p["insar_change_mm"] = p["insar_displacement_mm"] - float(p["insar_displacement_mm"].iloc[0])
        p["head_change_m"] = p["groundwater_head_anomaly_m"] - float(p["groundwater_head_anomaly_m"].iloc[0])
        p["profile_compaction_change_mm"] = p["profile_compaction_mm"] - float(p["profile_compaction_mm"].iloc[0])
        p["profile_equivalent_displacement_change_mm"] = -p["profile_compaction_change_mm"]
        p.to_csv(out_dir / "extensometer_overlap_timeseries.csv", index=False)
    return summary


def analyze_extensometer(cfg: ProjectConfig) -> dict[str, Any]:
    sec = cfg.section("extensometer")
    if not sec.get("enabled", False):
        return {"status": "skipped"}

    path = cfg.resolve(sec["path"])
    input_format = str(sec.get("format", "canonical_intervals")).lower()
    out_dir = ensure_dir(cfg.outputs / "extensometer")

    marker_long = None
    if input_format == "marker_table":
        marker_long, intervals, profile, markers = _read_marker_table(path, sec)
        marker_long.to_csv(out_dir / "extensometer_marker_timeseries.csv", index=False)
        intervals.to_csv(out_dir / "extensometer_interval_timeseries.csv", index=False)
        profile.to_csv(out_dir / "extensometer_profile_timeseries.csv", index=False)
        monitored_top = float(markers[0][1])
        monitored_bottom = float(markers[-1][1])
        measurement_type = "marker_displacement_relative_to_base"
    else:
        frame = _read_frame(path, sec.get("sheet_name", 0), sec.get("encoding"))
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
        measurement_type = str(sec.get("measurement_type", "interval_compaction"))
        intervals = _canonicalize_intervals(out, measurement_type)
        intervals = intervals.sort_values(["depth_top_m", "depth_bottom_m", "date"]).reset_index(drop=True)
        intervals.to_csv(out_dir / "extensometer_interval_timeseries.csv", index=False)
        profile = _profile_from_intervals(intervals)
        profile.to_csv(out_dir / "extensometer_profile_timeseries.csv", index=False)
        monitored_top = float(intervals["depth_top_m"].min())
        monitored_bottom = float(intervals["depth_bottom_m"].max())

    contribution = _interval_contributions(intervals)
    contribution.to_csv(out_dir / "depth_interval_contributions.csv", index=False)
    total_change = float(contribution["change_mm"].sum()) if len(contribution) else np.nan

    overlap = _write_overlap_products(cfg, sec, profile, out_dir)
    e0, e1 = _coverage(profile["date"])
    result = {
        "status": "ok",
        "input_format": input_format,
        "measurement_type": measurement_type,
        "first_date": None if e0 is None else str(e0.date()),
        "last_date": None if e1 is None else str(e1.date()),
        "epochs": int(profile["date"].nunique()),
        "intervals": int(contribution.shape[0]),
        "monitoring_top_m": monitored_top,
        "monitoring_bottom_m": monitored_bottom,
        "monitored_profile_change_mm": total_change,
        "note": (
            f"Profile compaction refers only to the monitored depth interval "
            f"{monitored_top:g}-{monitored_bottom:g} m; deformation above the shallowest marker is not inferred."
        ),
        "three_source_overlap_start": overlap.get("three_source_overlap_start"),
        "three_source_overlap_end": overlap.get("three_source_overlap_end"),
        "output_directory": str(out_dir),
    }
    write_json(out_dir / "extensometer_summary.json", result)
    return result
