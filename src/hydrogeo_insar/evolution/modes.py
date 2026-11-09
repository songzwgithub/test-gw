from __future__ import annotations

from collections import Counter
from pathlib import Path
import warnings
from typing import Any

import h5py
import numpy as np
import pandas as pd
import rasterio

from ..common import (
    block_slices,
    days_to_dates,
    ensure_dir,
    geotiff_profile,
    h5_grid_metadata,
    pixel_area_rows,
    write_json,
)
from ..config import ProjectConfig

YEAR_DAYS = 365.2425

DEFORMATION_CLASSES = {
    0: "complex_or_uncertain",
    1: "stable_or_weak_deformation",
    2: "persistent_subsidence",
    3: "accelerating_subsidence",
    4: "decelerating_subsidence",
    5: "subsidence_to_rebound",
}

COUPLED_CLASSES = {
    0: "complex_or_uncertain",
    1: "active_depletion_compaction",
    2: "recovery_responsive",
    3: "recovery_with_persistent_compaction",
    4: "recovery_with_accelerating_compaction",
    5: "quasi_stable",
}


def _anniversary(value: pd.Timestamp, year: int) -> pd.Timestamp:
    try:
        return value.replace(year=year)
    except ValueError:
        return value.replace(year=year, day=28)


def build_same_season_pairs(
    dates: np.ndarray,
    year: int,
    start_month: int,
    end_month: int,
    max_day_diff: int,
) -> list[dict[str, Any]]:
    """One-to-one nearest-anniversary pairs for a year-to-year interval."""
    dt = pd.to_datetime(np.asarray(dates, dtype="datetime64[D]"))
    src_idx = [
        i
        for i, d in enumerate(dt)
        if d.year == year and start_month <= d.month <= end_month
    ]
    dst_idx = [
        i
        for i, d in enumerate(dt)
        if d.year == year + 1 and start_month <= d.month <= end_month
    ]

    by_day: dict[int, int] = {
        int(dt[j].to_datetime64().astype("datetime64[D]").astype(np.int64)): j
        for j in dst_idx
    }
    used: set[int] = set()
    pairs: list[dict[str, Any]] = []

    offsets = [0]
    for d in range(1, int(max_day_diff) + 1):
        offsets.extend([-d, d])

    for i in src_idx:
        d1 = dt[i]
        target = _anniversary(d1, year + 1)
        target_day = int(
            target.to_datetime64().astype("datetime64[D]").astype(np.int64)
        )
        candidates: list[tuple[int, int]] = []
        for off in offsets:
            j = by_day.get(target_day + off)
            if j is not None and j not in used:
                candidates.append((abs(off), j))
                if off == 0:
                    break
        if not candidates:
            continue
        _, j = min(candidates, key=lambda x: (x[0], x[1]))
        used.add(j)
        d2 = dt[j]
        elapsed_days = int((d2 - d1).days)
        if elapsed_days <= 0:
            continue
        pairs.append(
            {
                "year1": int(year),
                "year2": int(year + 1),
                "i1": int(i),
                "i2": int(j),
                "date1": str(d1.date()),
                "date2": str(d2.date()),
                "anniversary_offset_days": int(
                    (d2 - target).days
                ),
                "elapsed_days": elapsed_days,
                "elapsed_years": elapsed_days / YEAR_DAYS,
            }
        )
    return pairs


def _interval_label(year1: int, year2: int) -> str:
    return f"{int(year1)}_{int(year2)}"


def _parse_interval_list(
    value: Any,
    default: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    raw = default if value is None else value
    out: list[tuple[int, int]] = []
    for item in raw:
        if len(item) != 2:
            raise ValueError("Each evolution interval must contain two years")
        a, b = int(item[0]), int(item[1])
        if b != a + 1:
            raise ValueError(
                "Evolution intervals must be adjacent year pairs, e.g. [2018, 2019]"
            )
        out.append((a, b))
    return out


def _build_pair_map(
    dates: np.ndarray,
    start_year: int,
    end_year: int,
    start_month: int,
    end_month: int,
    max_day_diff: int,
    min_pairs: int,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for year in range(int(start_year), int(end_year)):
        pairs = build_same_season_pairs(
            dates,
            year,
            start_month,
            end_month,
            max_day_diff,
        )
        if len(pairs) >= int(min_pairs):
            out[_interval_label(year, year + 1)] = pairs
    return out


def _open_float_writer(path: Path, grid: dict[str, Any]):
    profile = geotiff_profile(
        grid["height"],
        grid["width"],
        grid["crs"],
        grid["transform"],
        dtype="float32",
        nodata=np.nan,
    )
    return rasterio.open(path, "w", **profile)


def _open_uint_writer(path: Path, grid: dict[str, Any], dtype: str = "uint8"):
    profile = geotiff_profile(
        grid["height"],
        grid["width"],
        grid["crs"],
        grid["transform"],
        dtype=dtype,
        nodata=0,
    )
    return rasterio.open(path, "w", **profile)


def _pair_index_plan(pair_map: dict[str, list[dict[str, Any]]]):
    used = sorted(
        {
            p[key]
            for pairs in pair_map.values()
            for p in pairs
            for key in ("i1", "i2")
        }
    )
    local = {idx: k for k, idx in enumerate(used)}
    plan: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, pairs in pair_map.items():
        i1 = np.asarray([local[p["i1"]] for p in pairs], dtype=int)
        i2 = np.asarray([local[p["i2"]] for p in pairs], dtype=int)
        years = np.asarray([p["elapsed_years"] for p in pairs], dtype=float)
        plan[label] = (i1, i2, years)
    return np.asarray(used, dtype=int), plan


def _same_season_median(
    data: np.ndarray,
    i1: np.ndarray,
    i2: np.ndarray,
    elapsed_years: np.ndarray,
    min_valid_pairs: int,
) -> np.ndarray:
    scale = elapsed_years.reshape(-1, *([1] * (data.ndim - 1)))
    rates = (data[i2] - data[i1]) / scale
    nvalid = np.isfinite(rates).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(rates, axis=0)
    med[nvalid < int(min_valid_pairs)] = np.nan
    return med


def _write_same_season_maps(
    h5_path: Path,
    dataset_name: str,
    pair_map: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    prefix: str,
    suffix: str,
    grid: dict[str, Any],
    block_size: int,
    min_valid_pairs: int,
) -> dict[str, Path]:
    ensure_dir(output_dir)
    needed, plan = _pair_index_plan(pair_map)
    if len(needed) == 0:
        raise ValueError(f"No usable same-season pairs for {prefix}")

    paths = {
        label: output_dir / f"{prefix}_{label}_{suffix}.tif"
        for label in pair_map
    }
    writers = {
        label: _open_float_writer(path, grid)
        for label, path in paths.items()
    }

    try:
        with h5py.File(h5_path, "r") as h5:
            ds = h5[dataset_name]
            blocks = list(
                block_slices(
                    grid["height"],
                    grid["width"],
                    int(block_size),
                )
            )
            total = len(blocks)
            report_every = max(1, total // 40)
            for ib, (r0, r1, c0, c1) in enumerate(blocks, start=1):
                arr = ds[needed, r0:r1, c0:c1].astype(float)
                window = rasterio.windows.Window(
                    c0,
                    r0,
                    c1 - c0,
                    r1 - r0,
                )
                for label, (i1, i2, years) in plan.items():
                    med = _same_season_median(
                        arr,
                        i1,
                        i2,
                        years,
                        min_valid_pairs=min_valid_pairs,
                    )
                    writers[label].write(
                        med.astype("float32"),
                        1,
                        window=window,
                    )
                if ib == 1 or ib % report_every == 0 or ib == total:
                    print(
                        f"[EVOLUTION-{prefix.upper()}] {ib}/{total} "
                        f"({100.0 * ib / total:5.1f}%)",
                        flush=True,
                    )
    finally:
        for dst in writers.values():
            dst.close()
    return paths


def _nanmedian_stack(stack: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(stack, axis=axis)


def _nan_slope(
    stack: np.ndarray,
    x: np.ndarray,
    min_valid: int,
) -> np.ndarray:
    values = np.asarray(stack, dtype=float)
    x = np.asarray(x, dtype=float).reshape(-1, 1, 1)
    valid = np.isfinite(values)
    n = valid.sum(axis=0).astype(float)
    y0 = np.where(valid, values, 0.0)
    x0 = np.where(valid, x, 0.0)
    sx = x0.sum(axis=0)
    sy = y0.sum(axis=0)
    sxx = (x0 * x0).sum(axis=0)
    sxy = (x0 * y0).sum(axis=0)
    denom = n * sxx - sx * sx
    out = np.full(values.shape[1:], np.nan, dtype=float)
    ok = (n >= int(min_valid)) & np.isfinite(denom) & (np.abs(denom) > 1e-12)
    out[ok] = (n[ok] * sxy[ok] - sx[ok] * sy[ok]) / denom[ok]
    return out


def classify_deformation_arrays(
    early_rate: np.ndarray,
    late_rate: np.ndarray,
    rate_change: np.ndarray,
    stable_rate_mm_yr: float,
    meaningful_rate_change_mm_yr: float,
) -> np.ndarray:
    early = np.asarray(early_rate, dtype=float)
    late = np.asarray(late_rate, dtype=float)
    change = np.asarray(rate_change, dtype=float)
    out = np.zeros(early.shape, dtype="uint8")
    valid = np.isfinite(early) & np.isfinite(late) & np.isfinite(change)
    sr = float(stable_rate_mm_yr)
    dc = float(meaningful_rate_change_mm_yr)

    stable = valid & (np.abs(early) <= sr) & (np.abs(late) <= sr)
    rebound = valid & (early < -sr) & (late >= -sr)
    accelerating = valid & (late < -sr) & (change <= -dc)
    decelerating = valid & (early < -sr) & (late < -sr) & (change >= dc)
    persistent = valid & (late < -sr) & ~(accelerating | decelerating)

    out[stable] = 1
    out[persistent] = 2
    out[accelerating] = 3
    out[decelerating] = 4
    out[rebound] = 5
    return out


def classify_coupled_arrays(
    head_late_change: np.ndarray,
    deformation_mode: np.ndarray,
    deformation_late_rate: np.ndarray,
    stable_head_change_m_yr: float,
    meaningful_head_change_m_yr: float,
    stable_rate_mm_yr: float,
) -> np.ndarray:
    head = np.asarray(head_late_change, dtype=float)
    dmode = np.asarray(deformation_mode, dtype="uint8")
    late = np.asarray(deformation_late_rate, dtype=float)
    out = np.zeros(head.shape, dtype="uint8")
    valid = np.isfinite(head) & np.isfinite(late) & (dmode > 0)
    sh = float(stable_head_change_m_yr)
    mh = float(meaningful_head_change_m_yr)
    if mh < sh:
        raise ValueError("meaningful_head_change_m_yr must be >= stable_head_change_m_yr")
    sr = float(stable_rate_mm_yr)

    depletion = valid & (head < -mh) & (late < -sr)
    recovery_accel = valid & (head > mh) & (dmode == 3)
    recovery_responsive = valid & (head > mh) & np.isin(dmode, [4, 5])
    recovery_persistent = (
        valid
        & (head > mh)
        & (late < -sr)
        & ~recovery_accel
        & ~recovery_responsive
    )
    quasi_stable = valid & (np.abs(head) <= sh) & (np.abs(late) <= sr)

    out[depletion] = 1
    out[recovery_responsive] = 2
    out[recovery_persistent] = 3
    out[recovery_accel] = 4
    out[quasi_stable] = 5
    return out


def _support_mask_path(cfg: ProjectConfig, support: str) -> Path | None:
    support = str(support).strip().lower()
    mapping = {
        "high_confidence": cfg.outputs
        / "groundwater"
        / "groundwater_high_confidence_mask.tif",
        "inside_hull": cfg.outputs
        / "groundwater"
        / "groundwater_inside_well_hull.tif",
        "prediction": cfg.outputs
        / "groundwater"
        / "groundwater_support_mask.tif",
        "none": None,
    }
    if support not in mapping:
        raise ValueError(
            "evolution.support_mask must be one of: "
            "high_confidence, inside_hull, prediction, none"
        )
    return mapping[support]


def _read_support_block(
    src: rasterio.io.DatasetReader | None,
    window: rasterio.windows.Window,
    shape: tuple[int, int],
) -> np.ndarray:
    if src is None:
        return np.ones(shape, dtype=bool)
    return src.read(1, window=window) > 0


def _read_interval_block(
    readers: dict[str, rasterio.io.DatasetReader],
    labels: list[str],
    window: rasterio.windows.Window,
) -> np.ndarray:
    return np.stack(
        [readers[label].read(1, window=window).astype(float) for label in labels],
        axis=0,
    )


def _write_well_same_season_table(
    cfg: ProjectConfig,
    interval_pairs: list[tuple[int, int]],
    start_month: int,
    end_month: int,
    max_day_diff: int,
    min_pairs: int,
    out_path: Path,
) -> None:
    path = cfg.outputs / "canonical" / "groundwater.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path, parse_dates=["date"])
    aquifer = str(cfg.section("groundwater_field").get("aquifer", "confined"))
    if "aquifer_class" in frame.columns:
        frame = frame[frame["aquifer_class"].astype(str) == aquifer]
    rows: list[dict[str, Any]] = []
    for sid, group in frame.groupby("station_id", sort=True):
        g = group.dropna(subset=["head_m"]).copy()
        if g.empty:
            continue
        g = (
            g.groupby("date", as_index=False)
            .agg(
                head_m=("head_m", "median"),
                lon=("lon", "median"),
                lat=("lat", "median"),
            )
            .sort_values("date")
        )
        dates = g["date"].to_numpy(dtype="datetime64[D]")
        heads = g["head_m"].to_numpy(float)
        for year1, year2 in interval_pairs:
            pairs = build_same_season_pairs(
                dates,
                year1,
                start_month,
                end_month,
                max_day_diff,
            )
            if len(pairs) < int(min_pairs):
                continue
            values = np.asarray(
                [
                    (heads[p["i2"]] - heads[p["i1"]]) / p["elapsed_years"]
                    for p in pairs
                    if np.isfinite(heads[p["i1"]]) and np.isfinite(heads[p["i2"]])
                ],
                dtype=float,
            )
            if len(values) < int(min_pairs):
                continue
            rows.append(
                {
                    "station_id": str(sid),
                    "lon": float(g["lon"].iloc[0]),
                    "lat": float(g["lat"].iloc[0]),
                    "aquifer_class": aquifer,
                    "year1": int(year1),
                    "year2": int(year2),
                    "n_pairs": int(len(values)),
                    "head_change_m_yr": float(np.nanmedian(values)),
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def compute_evolution_modes(cfg: ProjectConfig) -> dict[str, Any]:
    """Compute physically interpretable deformation-groundwater evolution modes.

    Classification is intentionally observation-first. It uses only same-season
    interannual deformation and confined-head changes. Ske, storage partition,
    and hydrostratigraphy are reserved for independent interpretation.
    """
    sec = cfg.section("evolution")
    analysis = cfg.section("analysis")
    start_default = str(analysis.get("start_date", "2018-01-01"))
    end_default = str(analysis.get("end_date", "2025-12-31"))
    start_year = int(sec.get("start_year", pd.Timestamp(start_default).year))
    end_year = int(sec.get("end_year", pd.Timestamp(end_default).year))
    start_month, end_month = [
        int(v) for v in sec.get("same_season_months", [1, 8])
    ]
    max_day_diff = int(sec.get("max_pair_day_diff", 18))
    min_pairs = int(sec.get("min_pairs_per_interval", 3))
    block_size = int(sec.get("block_size", 256))

    stable_rate = float(sec.get("stable_rate_mm_yr", 5.0))
    meaningful_change = float(
        sec.get("meaningful_rate_change_mm_yr", 5.0)
    )
    stable_head = float(sec.get("stable_head_change_m_yr", 0.5))
    meaningful_head = float(sec.get("meaningful_head_change_m_yr", 1.0))
    if meaningful_head < stable_head:
        raise ValueError(
            "evolution.meaningful_head_change_m_yr must be >= "
            "evolution.stable_head_change_m_yr"
        )

    insar_path = cfg.outputs / "canonical" / "insar_stack.h5"
    head_path = cfg.outputs / "groundwater" / "groundwater_field.h5"
    if not insar_path.exists():
        raise FileNotFoundError(f"Missing canonical InSAR stack: {insar_path}")
    if not head_path.exists():
        raise FileNotFoundError(f"Missing groundwater field: {head_path}")

    grid = h5_grid_metadata(insar_path)
    if h5_grid_metadata(head_path) != grid:
        raise ValueError("Groundwater field grid does not match canonical InSAR grid")

    with h5py.File(insar_path, "r") as h5:
        insar_dates = days_to_dates(h5["date_days"][:])
    with h5py.File(head_path, "r") as h5:
        head_dates = days_to_dates(h5["date_days"][:])

    d_pairs = _build_pair_map(
        insar_dates,
        start_year,
        end_year,
        start_month,
        end_month,
        max_day_diff,
        min_pairs,
    )
    h_pairs = _build_pair_map(
        head_dates,
        start_year,
        end_year,
        start_month,
        end_month,
        max_day_diff,
        min_pairs,
    )

    common_labels = sorted(
        set(d_pairs) & set(h_pairs),
        key=lambda s: int(s.split("_")[0]),
    )
    if len(common_labels) < 2:
        raise ValueError("At least two common annual evolution intervals are required")

    n_default = min(3, max(1, len(common_labels) // 2))
    default_early = [tuple(map(int, x.split("_"))) for x in common_labels[:n_default]]
    default_late = [tuple(map(int, x.split("_"))) for x in common_labels[-n_default:]]
    early_pairs = _parse_interval_list(sec.get("early_intervals"), default_early)
    late_pairs = _parse_interval_list(sec.get("late_intervals"), default_late)
    early_labels = [_interval_label(*x) for x in early_pairs]
    late_labels = [_interval_label(*x) for x in late_pairs]

    required = set(early_labels + late_labels)
    missing_d = sorted(required - set(d_pairs))
    missing_h = sorted(required - set(h_pairs))
    if missing_d:
        raise ValueError(f"Missing required InSAR evolution intervals: {missing_d}")
    if missing_h:
        raise ValueError(f"Missing required groundwater evolution intervals: {missing_h}")

    min_valid_intervals = int(
        sec.get("min_valid_intervals", min(4, len(common_labels)))
    )
    min_valid_intervals = max(2, min(min_valid_intervals, len(common_labels)))

    out_dir = ensure_dir(cfg.outputs / "evolution")
    d_dir = ensure_dir(out_dir / "deformation")
    h_dir = ensure_dir(out_dir / "groundwater")

    d_paths = _write_same_season_maps(
        insar_path,
        "displacement_mm",
        {k: d_pairs[k] for k in common_labels},
        d_dir,
        "deformation_rate",
        "mm_yr",
        grid,
        block_size,
        min_pairs,
    )
    h_paths = _write_same_season_maps(
        head_path,
        "head_anomaly_m",
        {k: h_pairs[k] for k in common_labels},
        h_dir,
        "head_change",
        "m_yr",
        grid,
        block_size,
        min_pairs,
    )

    metric_names = [
        "deformation_early_rate_mm_yr",
        "deformation_late_rate_mm_yr",
        "deformation_rate_change_mm_yr",
        "deformation_rate_slope_mm_yr2",
        "head_early_change_m_yr",
        "head_late_change_m_yr",
        "head_change_difference_m_yr",
        "head_change_slope_m_yr2",
    ]
    float_writers = {
        name: _open_float_writer(out_dir / f"{name}.tif", grid)
        for name in metric_names
    }
    def_mode_writer = _open_uint_writer(
        out_dir / "deformation_evolution_mode.tif", grid
    )
    coupled_writer = _open_uint_writer(
        out_dir / "coupled_evolution_mode.tif", grid
    )
    d_valid_writer = _open_uint_writer(
        out_dir / "deformation_valid_intervals.tif", grid, dtype="uint8"
    )
    h_valid_writer = _open_uint_writer(
        out_dir / "head_valid_intervals.tif", grid, dtype="uint8"
    )

    support_name = str(sec.get("support_mask", "high_confidence"))
    support_path = _support_mask_path(cfg, support_name)
    if support_path is not None and not support_path.exists():
        raise FileNotFoundError(f"Evolution support mask does not exist: {support_path}")

    d_readers = {k: rasterio.open(d_paths[k]) for k in common_labels}
    h_readers = {k: rasterio.open(h_paths[k]) for k in common_labels}
    support_src = rasterio.open(support_path) if support_path is not None else None

    x = np.asarray(
        [
            (int(label.split("_")[0]) + int(label.split("_")[1])) / 2.0
            for label in common_labels
        ],
        dtype=float,
    )
    early_idx = np.asarray([common_labels.index(x) for x in early_labels], dtype=int)
    late_idx = np.asarray([common_labels.index(x) for x in late_labels], dtype=int)

    d_counts: Counter[int] = Counter()
    c_counts: Counter[int] = Counter()
    d_area = Counter()
    c_area = Counter()
    area_rows = pixel_area_rows(
        grid["height"], grid["width"], grid["crs"], grid["transform"]
    )

    sensitivity_cfg = sec.get("sensitivity", {})
    sr_options = [float(v) for v in sensitivity_cfg.get("stable_rate_mm_yr", [3.0, 5.0, 10.0])]
    dc_options = [float(v) for v in sensitivity_cfg.get("meaningful_rate_change_mm_yr", [3.0, 5.0, 10.0])]
    mh_options = [float(v) for v in sensitivity_cfg.get("meaningful_head_change_m_yr", [0.5, 1.0, 1.5])]
    mh_options = [v for v in mh_options if v >= stable_head]
    if not mh_options:
        mh_options = [meaningful_head]
    sensitivity_stride = max(1, int(sensitivity_cfg.get("sample_stride", 8)))
    sens_counts: dict[tuple[float, float, float], Counter[int]] = {
        (sr, dc, mh): Counter()
        for sr in sr_options
        for dc in dc_options
        for mh in mh_options
    }

    blocks = list(block_slices(grid["height"], grid["width"], block_size))
    total_blocks = len(blocks)
    report_every = max(1, total_blocks // 40)

    try:
        for ib, (r0, r1, c0, c1) in enumerate(blocks, start=1):
            window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
            shape = (r1 - r0, c1 - c0)
            dstack = _read_interval_block(d_readers, common_labels, window)
            hstack = _read_interval_block(h_readers, common_labels, window)

            dvalid = np.isfinite(dstack).sum(axis=0)
            hvalid = np.isfinite(hstack).sum(axis=0)
            enough = (dvalid >= min_valid_intervals) & (hvalid >= min_valid_intervals)
            support = _read_support_block(support_src, window, shape) & enough

            d_early = _nanmedian_stack(dstack[early_idx], axis=0)
            d_late = _nanmedian_stack(dstack[late_idx], axis=0)
            d_change = d_late - d_early
            d_slope = _nan_slope(dstack, x, min_valid_intervals)

            h_early = _nanmedian_stack(hstack[early_idx], axis=0)
            h_late = _nanmedian_stack(hstack[late_idx], axis=0)
            h_difference = h_late - h_early
            h_slope = _nan_slope(hstack, x, min_valid_intervals)

            for arr in [d_early, d_late, d_change, d_slope, h_early, h_late, h_difference, h_slope]:
                arr[~support] = np.nan

            dmode = classify_deformation_arrays(
                d_early,
                d_late,
                d_change,
                stable_rate,
                meaningful_change,
            )
            cmode = classify_coupled_arrays(
                h_late,
                dmode,
                d_late,
                stable_head,
                meaningful_head,
                stable_rate,
            )
            dmode[~support] = 0
            cmode[~support] = 0

            values = {
                "deformation_early_rate_mm_yr": d_early,
                "deformation_late_rate_mm_yr": d_late,
                "deformation_rate_change_mm_yr": d_change,
                "deformation_rate_slope_mm_yr2": d_slope,
                "head_early_change_m_yr": h_early,
                "head_late_change_m_yr": h_late,
                "head_change_difference_m_yr": h_difference,
                "head_change_slope_m_yr2": h_slope,
            }
            for name, arr in values.items():
                float_writers[name].write(arr.astype("float32"), 1, window=window)
            def_mode_writer.write(dmode, 1, window=window)
            coupled_writer.write(cmode, 1, window=window)
            d_valid_writer.write(np.where(support, dvalid, 0).astype("uint8"), 1, window=window)
            h_valid_writer.write(np.where(support, hvalid, 0).astype("uint8"), 1, window=window)

            area = area_rows[r0:r1, None]
            for cid in DEFORMATION_CLASSES:
                if cid == 0:
                    continue
                m = dmode == cid
                d_counts[cid] += int(m.sum())
                d_area[cid] += float(np.sum(np.where(m, area, 0.0)))
            for cid in COUPLED_CLASSES:
                if cid == 0:
                    continue
                m = cmode == cid
                c_counts[cid] += int(m.sum())
                c_area[cid] += float(np.sum(np.where(m, area, 0.0)))

            ss = (slice(None, None, sensitivity_stride), slice(None, None, sensitivity_stride))
            sup_s = support[ss]
            if sup_s.any():
                for sr in sr_options:
                    for dc in dc_options:
                        dm = classify_deformation_arrays(
                            d_early[ss], d_late[ss], d_change[ss], sr, dc
                        )
                        for mh in mh_options:
                            cm = classify_coupled_arrays(
                                h_late[ss], dm, d_late[ss], stable_head, mh, sr
                            )
                            cm[~sup_s] = 0
                            key = (sr, dc, mh)
                            vals, counts = np.unique(cm[cm > 0], return_counts=True)
                            for cid, count in zip(vals, counts):
                                sens_counts[key][int(cid)] += int(count)

            if ib == 1 or ib % report_every == 0 or ib == total_blocks:
                print(
                    f"[EVOLUTION-MODES] {ib}/{total_blocks} "
                    f"({100.0 * ib / total_blocks:5.1f}%)",
                    flush=True,
                )
    finally:
        for src in d_readers.values():
            src.close()
        for src in h_readers.values():
            src.close()
        if support_src is not None:
            support_src.close()
        for dst in float_writers.values():
            dst.close()
        def_mode_writer.close()
        coupled_writer.close()
        d_valid_writer.close()
        h_valid_writer.close()

    summary_rows: list[dict[str, Any]] = []
    for family, labels, counts, areas in [
        ("deformation", DEFORMATION_CLASSES, d_counts, d_area),
        ("coupled", COUPLED_CLASSES, c_counts, c_area),
    ]:
        total_area = float(sum(areas.values()))
        total_count = int(sum(counts.values()))
        for cid, label in labels.items():
            if cid == 0:
                continue
            area_km2 = float(areas[cid]) / 1e6
            summary_rows.append(
                {
                    "family": family,
                    "class_id": int(cid),
                    "class_label": label,
                    "pixel_count": int(counts[cid]),
                    "area_km2": area_km2,
                    "fraction_of_classified_pixels": (
                        float(counts[cid]) / total_count if total_count else np.nan
                    ),
                    "fraction_of_classified_area": (
                        float(areas[cid]) / total_area if total_area else np.nan
                    ),
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        out_dir / "evolution_mode_summary.csv", index=False
    )

    legend_rows = [
        {"family": "deformation", "class_id": k, "class_label": v}
        for k, v in DEFORMATION_CLASSES.items()
    ] + [
        {"family": "coupled", "class_id": k, "class_label": v}
        for k, v in COUPLED_CLASSES.items()
    ]
    pd.DataFrame(legend_rows).to_csv(out_dir / "evolution_mode_legend.csv", index=False)

    sens_rows: list[dict[str, Any]] = []
    for (sr, dc, mh), counts in sens_counts.items():
        total = int(sum(counts.values()))
        for cid, label in COUPLED_CLASSES.items():
            if cid == 0:
                continue
            sens_rows.append(
                {
                    "stable_rate_mm_yr": sr,
                    "meaningful_rate_change_mm_yr": dc,
                    "stable_head_change_m_yr": stable_head,
                    "meaningful_head_change_m_yr": mh,
                    "sample_stride": sensitivity_stride,
                    "class_id": cid,
                    "class_label": label,
                    "sample_pixel_count": int(counts[cid]),
                    "sample_fraction": (
                        float(counts[cid]) / total if total else np.nan
                    ),
                }
            )
    pd.DataFrame(sens_rows).to_csv(
        out_dir / "evolution_threshold_sensitivity.csv", index=False
    )

    interval_pairs = [tuple(map(int, label.split("_"))) for label in common_labels]
    _write_well_same_season_table(
        cfg,
        interval_pairs,
        start_month,
        end_month,
        max_day_diff,
        min_pairs,
        out_dir / "well_same_season_head_change.csv",
    )

    result = {
        "status": "ok",
        "method": "same_season_interannual_observation_first_classification",
        "analysis_years": [start_year, end_year],
        "same_season_months": [start_month, end_month],
        "max_pair_day_diff": max_day_diff,
        "min_pairs_per_interval": min_pairs,
        "common_intervals": common_labels,
        "early_intervals": early_labels,
        "late_intervals": late_labels,
        "thresholds": {
            "stable_rate_mm_yr": stable_rate,
            "meaningful_rate_change_mm_yr": meaningful_change,
            "stable_head_change_m_yr": stable_head,
            "meaningful_head_change_m_yr": meaningful_head,
        },
        "support_mask": support_name,
        "classification_inputs": [
            "same-season interannual InSAR deformation rate",
            "same-season interannual groundwater-head change",
        ],
        "classification_excludes": [
            "Ske",
            "storage partition",
            "hydrostratigraphy",
        ],
        "outputs": {
            "deformation_mode": str(out_dir / "deformation_evolution_mode.tif"),
            "coupled_mode": str(out_dir / "coupled_evolution_mode.tif"),
            "summary": str(out_dir / "evolution_mode_summary.csv"),
            "threshold_sensitivity": str(out_dir / "evolution_threshold_sensitivity.csv"),
            "well_same_season_head_change": str(out_dir / "well_same_season_head_change.csv"),
        },
    }
    write_json(out_dir / "evolution_summary.json", result)
    return result
