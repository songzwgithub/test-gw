from __future__ import annotations

from collections import Counter
from pathlib import Path
import warnings
from typing import Any

import h5py
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import xy as raster_xy
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

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
NODATA_CLASS = np.uint8(255)

DEFORMATION_CLASSES = {
    0: "complex_or_uncertain",
    1: "stable_or_weak_deformation",
    2: "persistent_subsidence",
    3: "accelerating_subsidence",
    4: "decelerating_subsidence",
    5: "subsidence_to_rebound",
}

GROUNDWATER_STATES = {
    0: "uncertain_or_transition",
    1: "head_decline",
    2: "quasi_stable",
    3: "head_recovery",
}

COUPLED_CLASSES = {
    0: "complex_or_uncertain",
    1: "persistent_head_decline_with_subsidence",
    2: "recovery_responsive",
    3: "post_recovery_persistent_subsidence",
    4: "post_recovery_reaccelerating_subsidence",
    5: "renewed_head_decline_with_reaccelerating_subsidence",
    6: "quasi_stable",
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
                "anniversary_offset_days": int((d2 - target).days),
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


def _open_uint_writer(
    path: Path,
    grid: dict[str, Any],
    dtype: str = "uint8",
    nodata: int = 255,
):
    profile = geotiff_profile(
        grid["height"],
        grid["width"],
        grid["crs"],
        grid["transform"],
        dtype=dtype,
        nodata=nodata,
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
    reuse_existing: bool = False,
) -> dict[str, Path]:
    ensure_dir(output_dir)
    paths = {
        label: output_dir / f"{prefix}_{label}_{suffix}.tif"
        for label in pair_map
    }
    if reuse_existing and paths and all(path.exists() for path in paths.values()):
        print(
            f"[EVOLUTION-{prefix.upper()}] reuse {len(paths)} existing interval maps",
            flush=True,
        )
        return paths

    needed, plan = _pair_index_plan(pair_map)
    if len(needed) == 0:
        raise ValueError(f"No usable same-season pairs for {prefix}")

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
                window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
                for label, (i1, i2, years) in plan.items():
                    med = _same_season_median(
                        arr,
                        i1,
                        i2,
                        years,
                        min_valid_pairs=min_valid_pairs,
                    )
                    writers[label].write(med.astype("float32"), 1, window=window)
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


def classify_groundwater_state_array(
    head_change_m_yr: np.ndarray,
    stable_head_change_m_yr: float,
    meaningful_head_change_m_yr: float,
) -> np.ndarray:
    head = np.asarray(head_change_m_yr, dtype=float)
    out = np.zeros(head.shape, dtype="uint8")
    valid = np.isfinite(head)
    sh = float(stable_head_change_m_yr)
    mh = float(meaningful_head_change_m_yr)
    if mh < sh:
        raise ValueError("meaningful_head_change_m_yr must be >= stable_head_change_m_yr")
    out[valid & (head < -mh)] = 1
    out[valid & (np.abs(head) <= sh)] = 2
    out[valid & (head > mh)] = 3
    return out


def classify_trajectory_arrays(
    prior_recovery_head_change: np.ndarray,
    recent_head_change: np.ndarray,
    recent_decline_count: np.ndarray,
    deformation_mode: np.ndarray,
    deformation_late_rate: np.ndarray,
    deformation_recent_rate_change: np.ndarray,
    stable_head_change_m_yr: float,
    meaningful_head_change_m_yr: float,
    stable_rate_mm_yr: float,
    meaningful_rate_change_mm_yr: float,
    recent_decline_min_intervals: int,
) -> np.ndarray:
    """Classify coupled trajectories without using inferred storage parameters."""
    recovery = np.asarray(prior_recovery_head_change, dtype=float)
    recent_head = np.asarray(recent_head_change, dtype=float)
    decline_count = np.asarray(recent_decline_count, dtype=float)
    dmode = np.asarray(deformation_mode, dtype="uint8")
    late = np.asarray(deformation_late_rate, dtype=float)
    recent_change = np.asarray(deformation_recent_rate_change, dtype=float)

    out = np.zeros(recovery.shape, dtype="uint8")
    valid = (
        np.isfinite(recovery)
        & np.isfinite(recent_head)
        & np.isfinite(decline_count)
        & np.isfinite(late)
        & np.isfinite(recent_change)
    )
    sh = float(stable_head_change_m_yr)
    mh = float(meaningful_head_change_m_yr)
    sr = float(stable_rate_mm_yr)
    dc = float(meaningful_rate_change_mm_yr)

    prior_recovery = valid & (recovery > mh)
    recent_decline = valid & (decline_count >= int(recent_decline_min_intervals))
    subsiding = valid & (late < -sr)
    recent_reacceleration = valid & (recent_change <= -dc)
    deformation_responsive = valid & np.isin(dmode, [4, 5])
    quasi_stable = valid & (np.abs(recent_head) <= sh) & (np.abs(late) <= sr)

    persistent_depletion = (
        valid
        & ~prior_recovery
        & recent_decline
        & subsiding
        & ~quasi_stable
    )
    renewed_decline_reaccel = (
        prior_recovery
        & recent_decline
        & subsiding
        & recent_reacceleration
        & ~quasi_stable
    )
    post_recovery_reaccel = (
        prior_recovery
        & ~recent_decline
        & subsiding
        & recent_reacceleration
        & ~quasi_stable
    )
    recovery_responsive = (
        prior_recovery
        & deformation_responsive
        & ~recent_reacceleration
        & ~quasi_stable
    )
    post_recovery_persistent = (
        prior_recovery
        & subsiding
        & ~recent_reacceleration
        & ~recovery_responsive
        & ~quasi_stable
    )

    out[persistent_depletion] = 1
    out[recovery_responsive] = 2
    out[post_recovery_persistent] = 3
    out[post_recovery_reaccel] = 4
    out[renewed_decline_reaccel] = 5
    out[quasi_stable] = 6
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


def _build_well_same_season_table(
    cfg: ProjectConfig,
    interval_pairs: list[tuple[int, int]],
    start_month: int,
    end_month: int,
    max_day_diff: int,
    min_pairs: int,
) -> pd.DataFrame:
    path = cfg.outputs / "canonical" / "groundwater.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical groundwater table: {path}")
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
    return pd.DataFrame(rows)


def _auto_utm_epsg(lon: float, lat: float) -> str:
    zone = int((lon + 180.0) // 6.0) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def _farthest_centers(points: np.ndarray, n: int) -> np.ndarray:
    n = min(max(1, int(n)), len(points))
    selected = [int(np.argmin(points[:, 0] + points[:, 1]))]
    min_dist = cdist(points, points[selected]).ravel()
    while len(selected) < n:
        j = int(np.argmax(min_dist))
        selected.append(j)
        min_dist = np.minimum(min_dist, cdist(points, points[[j]]).ravel())
    return points[selected]


def _rbf(points: np.ndarray, centers: np.ndarray, sigma_m: float) -> np.ndarray:
    return np.exp(-0.5 * (cdist(points, centers) / float(sigma_m)) ** 2)


def _fit_rbf_scalar(
    train_xy: np.ndarray,
    values: np.ndarray,
    sigma_m: float,
    ridge: float,
    max_centers: int,
) -> tuple[np.ndarray, np.ndarray]:
    centers = _farthest_centers(train_xy, min(max_centers, len(train_xy)))
    basis = np.column_stack([np.ones(len(train_xy)), _rbf(train_xy, centers, sigma_m)])
    penalty = float(ridge) * np.eye(basis.shape[1])
    penalty[0, 0] = 0.0
    lhs = basis.T @ basis + penalty
    rhs = basis.T @ np.asarray(values, dtype=float)
    try:
        coef = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return centers, coef


def _predict_rbf_scalar(
    query_xy: np.ndarray,
    centers: np.ndarray,
    sigma_m: float,
    coef: np.ndarray,
) -> np.ndarray:
    basis = np.column_stack([np.ones(len(query_xy)), _rbf(query_xy, centers, sigma_m)])
    return basis @ coef


def _spatial_fold_ids(
    xy: np.ndarray,
    n_folds: int,
    block_km: float,
    random_state: int,
) -> np.ndarray:
    n_folds = min(max(2, int(n_folds)), len(xy))
    block_m = float(block_km) * 1000.0
    bx = np.floor((xy[:, 0] - xy[:, 0].min()) / block_m).astype(int)
    by = np.floor((xy[:, 1] - xy[:, 1].min()) / block_m).astype(int)
    pairs = np.column_stack([bx, by])
    _, block_id = np.unique(pairs, axis=0, return_inverse=True)
    unique = np.unique(block_id)
    if len(unique) < 2:
        return np.arange(len(xy), dtype=int) % n_folds
    rng = np.random.default_rng(random_state)
    rng.shuffle(unique)
    mapping = {int(b): i % n_folds for i, b in enumerate(unique)}
    folds = np.asarray([mapping[int(b)] for b in block_id], dtype=int)
    if len(np.unique(folds)) < 2:
        folds = np.arange(len(xy), dtype=int) % n_folds
    return folds


def _project_well_changes(
    frame: pd.DataFrame,
    projected_crs: str | None,
) -> tuple[pd.DataFrame, str]:
    out = frame.copy()
    if projected_crs is None:
        projected_crs = _auto_utm_epsg(float(out["lon"].mean()), float(out["lat"].mean()))
    tr = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    x, y = tr.transform(out["lon"].to_numpy(float), out["lat"].to_numpy(float))
    out["x"] = np.asarray(x, dtype=float)
    out["y"] = np.asarray(y, dtype=float)
    return out, str(projected_crs)


def _groundwater_interpolation_settings(cfg: ProjectConfig) -> dict[str, Any]:
    sec = cfg.section("evolution")
    override = sec.get("groundwater_interpolation", {})
    field = cfg.section("groundwater_field")
    rbf = field.get("rbf", {})
    cv = field.get("cross_validation", {})
    return {
        "max_centers": int(override.get("max_centers", rbf.get("max_centers", 32))),
        "sigma_km_candidates": [
            float(v)
            for v in override.get(
                "sigma_km_candidates",
                rbf.get("sigma_km_candidates", [10.0, 15.0, 20.0, 25.0]),
            )
        ],
        "ridge_candidates": [
            float(v)
            for v in override.get(
                "ridge_candidates",
                rbf.get("ridge_candidates", [0.1, 0.3, 0.6, 1.0]),
            )
        ],
        "folds": int(override.get("folds", cv.get("folds", 5))),
        "block_km": float(override.get("block_km", cv.get("block_km", 30.0))),
        "random_state": int(
            override.get("random_state", cv.get("random_state", 20260919))
        ),
        "min_wells_per_interval": int(override.get("min_wells_per_interval", 8)),
        "max_nearest_well_km": override.get("max_nearest_well_km"),
    }


def _select_groundwater_change_model(
    frame: pd.DataFrame,
    labels: list[str],
    settings: dict[str, Any],
    stable_head: float,
    meaningful_head: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    min_wells = int(settings["min_wells_per_interval"])
    for sigma_km in settings["sigma_km_candidates"]:
        for ridge in settings["ridge_candidates"]:
            actual_all: list[np.ndarray] = []
            predicted_all: list[np.ndarray] = []
            n_interval = 0
            for label in labels:
                y1, y2 = map(int, label.split("_"))
                g = frame[(frame["year1"] == y1) & (frame["year2"] == y2)].copy()
                g = g[np.isfinite(g["head_change_m_yr"])].reset_index(drop=True)
                if len(g) < min_wells:
                    continue
                xy = g[["x", "y"]].to_numpy(float)
                values = g["head_change_m_yr"].to_numpy(float)
                folds = _spatial_fold_ids(
                    xy,
                    n_folds=min(int(settings["folds"]), len(g)),
                    block_km=float(settings["block_km"]),
                    random_state=int(settings["random_state"]),
                )
                used = False
                for fold in np.unique(folds):
                    test = folds == fold
                    train = ~test
                    if int(train.sum()) < 4 or int(test.sum()) == 0:
                        continue
                    centers, coef = _fit_rbf_scalar(
                        xy[train],
                        values[train],
                        float(sigma_km) * 1000.0,
                        float(ridge),
                        int(settings["max_centers"]),
                    )
                    pred = _predict_rbf_scalar(
                        xy[test], centers, float(sigma_km) * 1000.0, coef
                    )
                    actual_all.append(values[test])
                    predicted_all.append(pred)
                    used = True
                if used:
                    n_interval += 1

            if not actual_all:
                continue
            actual = np.concatenate(actual_all)
            predicted = np.concatenate(predicted_all)
            error = predicted - actual
            true_state = classify_groundwater_state_array(
                actual, stable_head, meaningful_head
            )
            pred_state = classify_groundwater_state_array(
                predicted, stable_head, meaningful_head
            )
            decisive = true_state > 0
            state_accuracy = (
                float(np.mean(pred_state[decisive] == true_state[decisive]))
                if decisive.any()
                else np.nan
            )
            rows.append(
                {
                    "sigma_km": float(sigma_km),
                    "ridge": float(ridge),
                    "n_intervals": int(n_interval),
                    "n_validation": int(len(actual)),
                    "cv_rmse_m_yr": float(np.sqrt(np.mean(error**2))),
                    "cv_mae_m_yr": float(np.mean(np.abs(error))),
                    "cv_bias_m_yr": float(np.mean(error)),
                    "cv_state_accuracy": state_accuracy,
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("Groundwater-change spatial cross-validation produced no valid candidate")
    ordered = table.sort_values(
        ["cv_rmse_m_yr", "cv_state_accuracy", "sigma_km", "ridge"],
        ascending=[True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    best = ordered.iloc[0].to_dict()
    return best, ordered



def _safe_pearson_r(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    ok = np.isfinite(actual) & np.isfinite(predicted)
    if int(ok.sum()) < 3:
        return np.nan
    a = actual[ok]
    p = predicted[ok]
    if np.nanstd(a) <= 1e-12 or np.nanstd(p) <= 1e-12:
        return np.nan
    return float(np.corrcoef(a, p)[0, 1])


def _groundwater_change_oof_audit(
    frame: pd.DataFrame,
    labels: list[str],
    settings: dict[str, Any],
    best: dict[str, Any],
    stable_head: float,
    meaningful_head: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Station-level out-of-fold predictions and interval diagnostics.
    prediction_rows: list[dict[str, Any]] = []
    min_wells = int(settings["min_wells_per_interval"])
    sigma_m = float(best["sigma_km"]) * 1000.0
    ridge = float(best["ridge"])

    for label in labels:
        y1, y2 = map(int, label.split("_"))
        g = frame[
            (frame["year1"] == y1)
            & (frame["year2"] == y2)
        ].copy()
        g = g[np.isfinite(g["head_change_m_yr"])].reset_index(drop=True)
        if len(g) < min_wells:
            continue

        xy = g[["x", "y"]].to_numpy(float)
        actual = g["head_change_m_yr"].to_numpy(float)
        folds = _spatial_fold_ids(
            xy,
            n_folds=min(int(settings["folds"]), len(g)),
            block_km=float(settings["block_km"]),
            random_state=int(settings["random_state"]),
        )

        predicted = np.full(len(g), np.nan, dtype=float)
        used_fold = np.full(len(g), -1, dtype=int)

        for fold in np.unique(folds):
            test = folds == fold
            train = ~test
            if int(train.sum()) < 4 or int(test.sum()) == 0:
                continue

            centers, coef = _fit_rbf_scalar(
                xy[train],
                actual[train],
                sigma_m,
                ridge,
                int(settings["max_centers"]),
            )
            predicted[test] = _predict_rbf_scalar(
                xy[test],
                centers,
                sigma_m,
                coef,
            )
            used_fold[test] = int(fold)

        actual_state = classify_groundwater_state_array(
            actual,
            stable_head,
            meaningful_head,
        )
        predicted_state = classify_groundwater_state_array(
            predicted,
            stable_head,
            meaningful_head,
        )

        for i, row in g.iterrows():
            if not np.isfinite(predicted[i]):
                continue
            prediction_rows.append(
                {
                    "station_id": str(row["station_id"]),
                    "year1": int(y1),
                    "year2": int(y2),
                    "lon": float(row["lon"]),
                    "lat": float(row["lat"]),
                    "fold": int(used_fold[i]),
                    "actual_head_change_m_yr": float(actual[i]),
                    "predicted_head_change_m_yr": float(predicted[i]),
                    "error_m_yr": float(predicted[i] - actual[i]),
                    "actual_state_id": int(actual_state[i]),
                    "actual_state": GROUNDWATER_STATES[int(actual_state[i])],
                    "predicted_state_id": int(predicted_state[i]),
                    "predicted_state": GROUNDWATER_STATES[int(predicted_state[i])],
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise ValueError(
            "Groundwater-change out-of-fold audit produced no valid predictions"
        )

    interval_rows: list[dict[str, Any]] = []
    for (y1, y2), g in predictions.groupby(["year1", "year2"], sort=True):
        actual = g["actual_head_change_m_yr"].to_numpy(float)
        predicted = g["predicted_head_change_m_yr"].to_numpy(float)
        error = predicted - actual
        actual_state = g["actual_state_id"].to_numpy(int)
        predicted_state = g["predicted_state_id"].to_numpy(int)
        decisive = actual_state > 0

        interval_rows.append(
            {
                "year1": int(y1),
                "year2": int(y2),
                "n_validation": int(len(g)),
                "n_decisive_state": int(decisive.sum()),
                "actual_median_m_yr": float(np.nanmedian(actual)),
                "predicted_median_m_yr": float(np.nanmedian(predicted)),
                "median_difference_m_yr": float(
                    np.nanmedian(predicted) - np.nanmedian(actual)
                ),
                "cv_rmse_m_yr": float(np.sqrt(np.mean(error**2))),
                "cv_mae_m_yr": float(np.mean(np.abs(error))),
                "cv_bias_m_yr": float(np.mean(error)),
                "pearson_r": _safe_pearson_r(actual, predicted),
                "state_accuracy": (
                    float(
                        np.mean(
                            predicted_state[decisive]
                            == actual_state[decisive]
                        )
                    )
                    if decisive.any()
                    else np.nan
                ),
            }
        )

    by_interval = pd.DataFrame(interval_rows)

    confusion_rows: list[dict[str, Any]] = []
    scopes = [("all", None, None, predictions)]
    scopes.extend(
        (
            "interval",
            int(y1),
            int(y2),
            g.copy(),
        )
        for (y1, y2), g in predictions.groupby(["year1", "year2"], sort=True)
    )

    for scope, y1, y2, g in scopes:
        for actual_id, actual_label in GROUNDWATER_STATES.items():
            ga = g[g["actual_state_id"] == actual_id]
            denom = int(len(ga))
            for predicted_id, predicted_label in GROUNDWATER_STATES.items():
                count = int((ga["predicted_state_id"] == predicted_id).sum())
                confusion_rows.append(
                    {
                        "scope": scope,
                        "year1": y1,
                        "year2": y2,
                        "actual_state_id": int(actual_id),
                        "actual_state": actual_label,
                        "predicted_state_id": int(predicted_id),
                        "predicted_state": predicted_label,
                        "count": count,
                        "actual_state_count": denom,
                        "fraction_of_actual_state": (
                            count / denom if denom else np.nan
                        ),
                    }
                )

    confusion = pd.DataFrame(confusion_rows)
    return predictions, by_interval, confusion


def _fit_interval_groundwater_models(
    frame: pd.DataFrame,
    labels: list[str],
    best: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    models: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    min_wells = int(settings["min_wells_per_interval"])
    for label in labels:
        y1, y2 = map(int, label.split("_"))
        g = frame[(frame["year1"] == y1) & (frame["year2"] == y2)].copy()
        g = g[np.isfinite(g["head_change_m_yr"])].reset_index(drop=True)
        if len(g) < min_wells:
            continue
        xy = g[["x", "y"]].to_numpy(float)
        values = g["head_change_m_yr"].to_numpy(float)
        centers, coef = _fit_rbf_scalar(
            xy,
            values,
            float(best["sigma_km"]) * 1000.0,
            float(best["ridge"]),
            int(settings["max_centers"]),
        )
        hull = Delaunay(xy) if len(xy) >= 3 else None
        models[label] = {
            "xy": xy,
            "values": values,
            "centers": centers,
            "coef": coef,
            "hull": hull,
        }
        rows.append(
            {
                "year1": y1,
                "year2": y2,
                "n_wells": int(len(g)),
                "well_median_m_yr": float(np.nanmedian(values)),
                "well_mean_m_yr": float(np.nanmean(values)),
                "well_p10_m_yr": float(np.nanpercentile(values, 10)),
                "well_p90_m_yr": float(np.nanpercentile(values, 90)),
            }
        )
    return models, pd.DataFrame(rows)


def _write_groundwater_change_maps(
    cfg: ProjectConfig,
    frame: pd.DataFrame,
    labels: list[str],
    grid: dict[str, Any],
    out_dir: Path,
    stable_head: float,
    meaningful_head: float,
    block_size: int,
    support_path: Path | None,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Any]]:
    settings = _groundwater_interpolation_settings(cfg)
    projected_cfg = cfg.section("groundwater_field").get("projected_crs")
    projected, projected_crs = _project_well_changes(frame, projected_cfg)

    best, cv_table = _select_groundwater_change_model(
        projected,
        labels,
        settings,
        stable_head,
        meaningful_head,
    )
    cv_table.to_csv(out_dir.parent / "groundwater_change_spatial_cv.csv", index=False)

    cv_predictions, cv_by_interval, cv_confusion = _groundwater_change_oof_audit(
        projected,
        labels,
        settings,
        best,
        stable_head,
        meaningful_head,
    )
    cv_predictions.to_csv(
        out_dir.parent / "groundwater_change_cv_predictions.csv",
        index=False,
    )
    cv_by_interval.to_csv(
        out_dir.parent / "groundwater_change_spatial_cv_by_interval.csv",
        index=False,
    )
    cv_confusion.to_csv(
        out_dir.parent / "groundwater_change_state_confusion.csv",
        index=False,
    )

    models, interval_summary = _fit_interval_groundwater_models(
        projected,
        labels,
        best,
        settings,
    )
    missing = sorted(set(labels) - set(models))
    if missing:
        raise ValueError(f"Too few wells for groundwater-change intervals: {missing}")
    interval_summary.to_csv(
        out_dir.parent / "groundwater_change_interval_summary.csv", index=False
    )

    paths = {
        label: out_dir / f"head_change_{label}_m_yr.tif"
        for label in labels
    }
    state_paths = {
        label: out_dir / f"groundwater_state_{label}.tif"
        for label in labels
    }
    writers = {label: _open_float_writer(path, grid) for label, path in paths.items()}
    state_writers = {
        label: _open_uint_writer(path, grid, nodata=int(NODATA_CLASS))
        for label, path in state_paths.items()
    }

    support_src = rasterio.open(support_path) if support_path is not None else None
    grid_to_model = Transformer.from_crs(grid["crs"], projected_crs, always_xy=True)
    max_nearest_km = settings.get("max_nearest_well_km")
    max_nearest_m = None if max_nearest_km is None else float(max_nearest_km) * 1000.0

    blocks = list(block_slices(grid["height"], grid["width"], int(block_size)))
    total = len(blocks)
    report_every = max(1, total // 40)

    try:
        for ib, (r0, r1, c0, c1) in enumerate(blocks, start=1):
            window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
            shape = (r1 - r0, c1 - c0)
            rr, cc = np.meshgrid(
                np.arange(r0, r1), np.arange(c0, c1), indexing="ij"
            )
            xs, ys = raster_xy(
                grid["transform"], rr.ravel(), cc.ravel(), offset="center"
            )
            mx, my = grid_to_model.transform(np.asarray(xs), np.asarray(ys))
            qxy = np.column_stack([mx, my])
            base_support = _read_support_block(support_src, window, shape).ravel()

            for label in labels:
                model = models[label]
                valid = base_support.copy()
                hull = model["hull"]
                if hull is not None:
                    valid &= hull.find_simplex(qxy) >= 0
                if max_nearest_m is not None and valid.any():
                    d = cdist(qxy[valid], model["xy"])
                    nearest = np.min(d, axis=1)
                    vv = valid.copy()
                    vv[valid] = nearest <= max_nearest_m
                    valid = vv

                pred = np.full(len(qxy), np.nan, dtype=float)
                state = np.full(len(qxy), int(NODATA_CLASS), dtype="uint8")
                if valid.any():
                    pred[valid] = _predict_rbf_scalar(
                        qxy[valid],
                        model["centers"],
                        float(best["sigma_km"]) * 1000.0,
                        model["coef"],
                    )
                    state[valid] = classify_groundwater_state_array(
                        pred[valid], stable_head, meaningful_head
                    )

                writers[label].write(pred.reshape(shape).astype("float32"), 1, window=window)
                state_writers[label].write(state.reshape(shape), 1, window=window)

            if ib == 1 or ib % report_every == 0 or ib == total:
                print(
                    f"[EVOLUTION-GROUNDWATER] {ib}/{total} "
                    f"({100.0 * ib / total:5.1f}%)",
                    flush=True,
                )
    finally:
        if support_src is not None:
            support_src.close()
        for dst in writers.values():
            dst.close()
        for dst in state_writers.values():
            dst.close()

    summary = {
        "method": "direct_well_same_season_change_rbf_spatial_cv",
        "projected_crs": projected_crs,
        "sigma_km": float(best["sigma_km"]),
        "ridge": float(best["ridge"]),
        "cv_rmse_m_yr": float(best["cv_rmse_m_yr"]),
        "cv_mae_m_yr": float(best["cv_mae_m_yr"]),
        "cv_bias_m_yr": float(best["cv_bias_m_yr"]),
        "cv_state_accuracy": float(best["cv_state_accuracy"]),
        "max_centers": int(settings["max_centers"]),
        "min_wells_per_interval": int(settings["min_wells_per_interval"]),
        "max_nearest_well_km": settings.get("max_nearest_well_km"),
        "oof_interval_count": int(len(cv_by_interval)),
        "oof_interval_cv_rmse_m_yr_min": float(
            cv_by_interval["cv_rmse_m_yr"].min()
        ),
        "oof_interval_cv_rmse_m_yr_max": float(
            cv_by_interval["cv_rmse_m_yr"].max()
        ),
        "oof_interval_state_accuracy_min": float(
            cv_by_interval["state_accuracy"].min()
        ),
        "oof_interval_state_accuracy_max": float(
            cv_by_interval["state_accuracy"].max()
        ),
        "oof_interval_pearson_r_median": float(
            cv_by_interval["pearson_r"].median()
        ),
    }
    write_json(out_dir.parent / "groundwater_change_interpolation_summary.json", summary)
    return paths, state_paths, summary


def _mode_summary_rows(
    family: str,
    labels: dict[int, str],
    counts: Counter[int],
    areas: Counter[int],
    support_count: int,
    support_area: float,
) -> list[dict[str, Any]]:
    classified_count = int(sum(counts[cid] for cid in labels if cid != 0))
    classified_area = float(sum(areas[cid] for cid in labels if cid != 0))
    rows: list[dict[str, Any]] = []
    for cid, label in labels.items():
        count = int(counts[cid])
        area = float(areas[cid])
        rows.append(
            {
                "family": family,
                "class_id": int(cid),
                "class_label": label,
                "pixel_count": count,
                "area_km2": area / 1e6,
                "support_pixel_count": int(support_count),
                "support_area_km2": float(support_area) / 1e6,
                "fraction_of_support_pixels": (
                    count / support_count if support_count else np.nan
                ),
                "fraction_of_support_area": (
                    area / support_area if support_area else np.nan
                ),
                "classified_pixel_count": classified_count,
                "classified_area_km2": classified_area / 1e6,
                "classified_fraction_of_support": (
                    classified_count / support_count if support_count else np.nan
                ),
                "fraction_of_classified_pixels": (
                    count / classified_count
                    if cid != 0 and classified_count
                    else np.nan
                ),
                "fraction_of_classified_area": (
                    area / classified_area
                    if cid != 0 and classified_area
                    else np.nan
                ),
            }
        )
    return rows


def compute_evolution_modes(cfg: ProjectConfig) -> dict[str, Any]:
    """Compute observation-first groundwater-deformation evolution trajectories."""
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
    meaningful_change = float(sec.get("meaningful_rate_change_mm_yr", 5.0))
    stable_head = float(sec.get("stable_head_change_m_yr", 0.5))
    meaningful_head = float(sec.get("meaningful_head_change_m_yr", 1.0))
    if meaningful_head < stable_head:
        raise ValueError(
            "evolution.meaningful_head_change_m_yr must be >= "
            "evolution.stable_head_change_m_yr"
        )

    insar_path = cfg.outputs / "canonical" / "insar_stack.h5"
    if not insar_path.exists():
        raise FileNotFoundError(f"Missing canonical InSAR stack: {insar_path}")
    grid = h5_grid_metadata(insar_path)
    with h5py.File(insar_path, "r") as h5:
        insar_dates = days_to_dates(h5["date_days"][:])

    d_pairs = _build_pair_map(
        insar_dates,
        start_year,
        end_year,
        start_month,
        end_month,
        max_day_diff,
        min_pairs,
    )
    if len(d_pairs) < 2:
        raise ValueError("At least two InSAR annual evolution intervals are required")

    candidate_labels = sorted(d_pairs, key=lambda s: int(s.split("_")[0]))
    interval_pairs = [tuple(map(int, label.split("_"))) for label in candidate_labels]
    well_table = _build_well_same_season_table(
        cfg,
        interval_pairs,
        start_month,
        end_month,
        max_day_diff,
        min_pairs,
    )
    if well_table.empty:
        raise ValueError("No valid well-based same-season groundwater changes")

    out_dir = ensure_dir(cfg.outputs / "evolution")
    d_dir = ensure_dir(out_dir / "deformation")
    h_dir = ensure_dir(out_dir / "groundwater")
    well_table.to_csv(out_dir / "well_same_season_head_change.csv", index=False)

    settings = _groundwater_interpolation_settings(cfg)
    counts = (
        well_table.groupby(["year1", "year2"])
        .size()
        .rename("n")
        .reset_index()
    )
    available_h = {
        _interval_label(row.year1, row.year2)
        for row in counts.itertuples()
        if int(row.n) >= int(settings["min_wells_per_interval"])
    }
    common_labels = [label for label in candidate_labels if label in available_h]
    if len(common_labels) < 2:
        raise ValueError("At least two common InSAR/well groundwater intervals are required")

    n_default = min(3, max(1, len(common_labels) // 2))
    default_early = [tuple(map(int, x.split("_"))) for x in common_labels[:n_default]]
    default_late = [tuple(map(int, x.split("_"))) for x in common_labels[-n_default:]]
    default_recent_labels = common_labels[-min(3, len(common_labels)):]
    recent_start = max(0, len(common_labels) - len(default_recent_labels) - 2)
    recovery_stop = len(common_labels) - len(default_recent_labels)
    default_recovery_labels = common_labels[recent_start:recovery_stop]
    if not default_recovery_labels:
        default_recovery_labels = common_labels[: min(2, len(common_labels))]

    early_pairs = _parse_interval_list(sec.get("early_intervals"), default_early)
    late_pairs = _parse_interval_list(sec.get("late_intervals"), default_late)
    recovery_pairs = _parse_interval_list(
        sec.get("recovery_intervals"),
        [tuple(map(int, x.split("_"))) for x in default_recovery_labels],
    )
    recent_pairs = _parse_interval_list(
        sec.get("recent_intervals"),
        [tuple(map(int, x.split("_"))) for x in default_recent_labels],
    )
    early_labels = [_interval_label(*x) for x in early_pairs]
    late_labels = [_interval_label(*x) for x in late_pairs]
    recovery_labels = [_interval_label(*x) for x in recovery_pairs]
    recent_labels = [_interval_label(*x) for x in recent_pairs]

    required = set(early_labels + late_labels + recovery_labels + recent_labels)
    missing = sorted(required - set(common_labels))
    if missing:
        raise ValueError(f"Missing required evolution intervals: {missing}")

    min_valid_intervals = int(sec.get("min_valid_intervals", min(4, len(common_labels))))
    min_valid_intervals = max(2, min(min_valid_intervals, len(common_labels)))
    min_recent_intervals = int(
        sec.get("min_recent_intervals", min(2, len(recent_labels)))
    )
    min_recent_intervals = max(1, min(min_recent_intervals, len(recent_labels)))
    min_recovery_intervals = int(sec.get("min_recovery_intervals", 1))
    min_recovery_intervals = max(1, min(min_recovery_intervals, len(recovery_labels)))
    recent_decline_min_intervals = int(
        sec.get("recent_decline_min_intervals", min(2, len(recent_labels)))
    )
    recent_decline_min_intervals = max(
        1, min(recent_decline_min_intervals, len(recent_labels))
    )

    reuse_deformation = bool(sec.get("reuse_existing_deformation_maps", True))
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
        reuse_existing=reuse_deformation,
    )

    support_name = str(sec.get("support_mask", "high_confidence"))
    support_path = _support_mask_path(cfg, support_name)
    if support_path is not None and not support_path.exists():
        raise FileNotFoundError(f"Evolution support mask does not exist: {support_path}")

    h_paths, groundwater_state_paths, groundwater_interp = _write_groundwater_change_maps(
        cfg,
        well_table,
        common_labels,
        grid,
        h_dir,
        stable_head,
        meaningful_head,
        block_size,
        support_path,
    )

    metric_names = [
        "deformation_early_rate_mm_yr",
        "deformation_late_rate_mm_yr",
        "deformation_rate_change_mm_yr",
        "deformation_rate_slope_mm_yr2",
        "deformation_recent_rate_change_mm_yr",
        "deformation_recent_rate_slope_mm_yr2",
        "head_early_change_m_yr",
        "head_late_change_m_yr",
        "head_change_difference_m_yr",
        "head_change_slope_m_yr2",
        "head_recovery_period_change_m_yr",
        "head_recent_change_m_yr",
    ]
    float_writers = {
        name: _open_float_writer(out_dir / f"{name}.tif", grid)
        for name in metric_names
    }
    def_mode_writer = _open_uint_writer(
        out_dir / "deformation_evolution_mode.tif", grid, nodata=int(NODATA_CLASS)
    )
    coupled_writer = _open_uint_writer(
        out_dir / "coupled_evolution_mode.tif", grid, nodata=int(NODATA_CLASS)
    )
    recent_gw_state_writer = _open_uint_writer(
        out_dir / "groundwater_recent_state.tif", grid, nodata=int(NODATA_CLASS)
    )
    d_valid_writer = _open_uint_writer(
        out_dir / "deformation_valid_intervals.tif", grid, dtype="uint8", nodata=0
    )
    h_valid_writer = _open_uint_writer(
        out_dir / "head_valid_intervals.tif", grid, dtype="uint8", nodata=0
    )

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
    early_idx = np.asarray([common_labels.index(v) for v in early_labels], dtype=int)
    late_idx = np.asarray([common_labels.index(v) for v in late_labels], dtype=int)
    recovery_idx = np.asarray([common_labels.index(v) for v in recovery_labels], dtype=int)
    recent_idx = np.asarray([common_labels.index(v) for v in recent_labels], dtype=int)
    recent_x = x[recent_idx]
    recent_span = float(recent_x[-1] - recent_x[0]) if len(recent_x) >= 2 else 1.0

    d_counts: Counter[int] = Counter()
    c_counts: Counter[int] = Counter()
    d_area: Counter[int] = Counter()
    c_area: Counter[int] = Counter()
    cross_counts: Counter[tuple[int, int]] = Counter()
    cross_area: Counter[tuple[int, int]] = Counter()
    support_count = 0
    support_area = 0.0
    area_rows = pixel_area_rows(
        grid["height"], grid["width"], grid["crs"], grid["transform"]
    )

    sensitivity_cfg = sec.get("sensitivity", {})
    sr_options = [
        float(v)
        for v in sensitivity_cfg.get("stable_rate_mm_yr", [3.0, 5.0, 10.0])
    ]
    dc_options = [
        float(v)
        for v in sensitivity_cfg.get(
            "meaningful_rate_change_mm_yr", [3.0, 5.0, 10.0]
        )
    ]
    mh_options = [
        float(v)
        for v in sensitivity_cfg.get(
            "meaningful_head_change_m_yr", [0.5, 1.0, 1.5]
        )
    ]
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
    sens_support_counts: Counter[tuple[float, float, float]] = Counter()

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
            recent_dvalid = np.isfinite(dstack[recent_idx]).sum(axis=0)
            recent_hvalid = np.isfinite(hstack[recent_idx]).sum(axis=0)
            recovery_hvalid = np.isfinite(hstack[recovery_idx]).sum(axis=0)
            enough = (
                (dvalid >= min_valid_intervals)
                & (hvalid >= min_valid_intervals)
                & (recent_dvalid >= min_recent_intervals)
                & (recent_hvalid >= min_recent_intervals)
                & (recovery_hvalid >= min_recovery_intervals)
            )
            support = _read_support_block(support_src, window, shape) & enough

            d_early = _nanmedian_stack(dstack[early_idx], axis=0)
            d_late = _nanmedian_stack(dstack[late_idx], axis=0)
            d_change = d_late - d_early
            d_slope = _nan_slope(dstack, x, min_valid_intervals)
            d_recent_slope = _nan_slope(
                dstack[recent_idx], recent_x, min_recent_intervals
            )
            d_recent_change = d_recent_slope * recent_span

            h_early = _nanmedian_stack(hstack[early_idx], axis=0)
            h_late = _nanmedian_stack(hstack[late_idx], axis=0)
            h_difference = h_late - h_early
            h_slope = _nan_slope(hstack, x, min_valid_intervals)
            h_recovery = _nanmedian_stack(hstack[recovery_idx], axis=0)
            h_recent = _nanmedian_stack(hstack[recent_idx], axis=0)
            recent_decline_count = np.sum(
                hstack[recent_idx] < -meaningful_head,
                axis=0,
            ).astype(float)

            for arr in [
                d_early,
                d_late,
                d_change,
                d_slope,
                d_recent_change,
                d_recent_slope,
                h_early,
                h_late,
                h_difference,
                h_slope,
                h_recovery,
                h_recent,
            ]:
                arr[~support] = np.nan

            dmode = classify_deformation_arrays(
                d_early,
                d_late,
                d_change,
                stable_rate,
                meaningful_change,
            )
            cmode = classify_trajectory_arrays(
                h_recovery,
                h_recent,
                recent_decline_count,
                dmode,
                d_late,
                d_recent_change,
                stable_head,
                meaningful_head,
                stable_rate,
                meaningful_change,
                recent_decline_min_intervals,
            )
            recent_gw_state = classify_groundwater_state_array(
                h_recent, stable_head, meaningful_head
            )

            values = {
                "deformation_early_rate_mm_yr": d_early,
                "deformation_late_rate_mm_yr": d_late,
                "deformation_rate_change_mm_yr": d_change,
                "deformation_rate_slope_mm_yr2": d_slope,
                "deformation_recent_rate_change_mm_yr": d_recent_change,
                "deformation_recent_rate_slope_mm_yr2": d_recent_slope,
                "head_early_change_m_yr": h_early,
                "head_late_change_m_yr": h_late,
                "head_change_difference_m_yr": h_difference,
                "head_change_slope_m_yr2": h_slope,
                "head_recovery_period_change_m_yr": h_recovery,
                "head_recent_change_m_yr": h_recent,
            }
            for name, arr in values.items():
                float_writers[name].write(arr.astype("float32"), 1, window=window)

            dmode_out = np.where(support, dmode, NODATA_CLASS).astype("uint8")
            cmode_out = np.where(support, cmode, NODATA_CLASS).astype("uint8")
            gw_state_out = np.where(
                support, recent_gw_state, NODATA_CLASS
            ).astype("uint8")
            def_mode_writer.write(dmode_out, 1, window=window)
            coupled_writer.write(cmode_out, 1, window=window)
            recent_gw_state_writer.write(gw_state_out, 1, window=window)
            d_valid_writer.write(
                np.where(support, dvalid, 0).astype("uint8"), 1, window=window
            )
            h_valid_writer.write(
                np.where(support, hvalid, 0).astype("uint8"), 1, window=window
            )

            area = area_rows[r0:r1, None]
            support_count += int(support.sum())
            support_area += float(np.sum(np.where(support, area, 0.0)))
            for cid in DEFORMATION_CLASSES:
                m = support & (dmode == cid)
                d_counts[cid] += int(m.sum())
                d_area[cid] += float(np.sum(np.where(m, area, 0.0)))
            for cid in COUPLED_CLASSES:
                m = support & (cmode == cid)
                c_counts[cid] += int(m.sum())
                c_area[cid] += float(np.sum(np.where(m, area, 0.0)))
            for gid in GROUNDWATER_STATES:
                for did in DEFORMATION_CLASSES:
                    m = support & (recent_gw_state == gid) & (dmode == did)
                    if m.any():
                        cross_counts[(gid, did)] += int(m.sum())
                        cross_area[(gid, did)] += float(
                            np.sum(np.where(m, area, 0.0))
                        )

            ss = (
                slice(None, None, sensitivity_stride),
                slice(None, None, sensitivity_stride),
            )
            sup_s = support[ss]
            if sup_s.any():
                for sr in sr_options:
                    for dc in dc_options:
                        dm = classify_deformation_arrays(
                            d_early[ss], d_late[ss], d_change[ss], sr, dc
                        )
                        for mh in mh_options:
                            decline_count_s = np.sum(
                                hstack[recent_idx][:, ss[0], ss[1]] < -mh,
                                axis=0,
                            ).astype(float)
                            cm = classify_trajectory_arrays(
                                h_recovery[ss],
                                h_recent[ss],
                                decline_count_s,
                                dm,
                                d_late[ss],
                                d_recent_change[ss],
                                stable_head,
                                mh,
                                sr,
                                dc,
                                recent_decline_min_intervals,
                            )
                            key = (sr, dc, mh)
                            sens_support_counts[key] += int(sup_s.sum())
                            vals, cc = np.unique(cm[sup_s], return_counts=True)
                            for cid, count in zip(vals, cc):
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
        recent_gw_state_writer.close()
        d_valid_writer.close()
        h_valid_writer.close()

    summary_rows = _mode_summary_rows(
        "deformation",
        DEFORMATION_CLASSES,
        d_counts,
        d_area,
        support_count,
        support_area,
    )
    summary_rows += _mode_summary_rows(
        "coupled",
        COUPLED_CLASSES,
        c_counts,
        c_area,
        support_count,
        support_area,
    )
    pd.DataFrame(summary_rows).to_csv(
        out_dir / "evolution_mode_summary.csv", index=False
    )

    legend_rows = [
        {"family": "deformation", "class_id": k, "class_label": v}
        for k, v in DEFORMATION_CLASSES.items()
    ] + [
        {"family": "groundwater", "class_id": k, "class_label": v}
        for k, v in GROUNDWATER_STATES.items()
    ] + [
        {"family": "coupled", "class_id": k, "class_label": v}
        for k, v in COUPLED_CLASSES.items()
    ] + [
        {
            "family": "all",
            "class_id": int(NODATA_CLASS),
            "class_label": "outside_support_or_nodata",
        }
    ]
    pd.DataFrame(legend_rows).to_csv(out_dir / "evolution_mode_legend.csv", index=False)

    sens_rows: list[dict[str, Any]] = []
    for (sr, dc, mh), counts_i in sens_counts.items():
        support_n = int(sens_support_counts[(sr, dc, mh)])
        classified_n = int(
            sum(counts_i[cid] for cid in COUPLED_CLASSES if cid != 0)
        )
        for cid, label in COUPLED_CLASSES.items():
            count = int(counts_i[cid])
            sens_rows.append(
                {
                    "stable_rate_mm_yr": sr,
                    "meaningful_rate_change_mm_yr": dc,
                    "stable_head_change_m_yr": stable_head,
                    "meaningful_head_change_m_yr": mh,
                    "sample_stride": sensitivity_stride,
                    "support_sample_pixel_count": support_n,
                    "classified_sample_pixel_count": classified_n,
                    "classified_fraction_of_support": (
                        classified_n / support_n if support_n else np.nan
                    ),
                    "class_id": cid,
                    "class_label": label,
                    "sample_pixel_count": count,
                    "sample_fraction_of_support": (
                        count / support_n if support_n else np.nan
                    ),
                    "sample_fraction_of_classified": (
                        count / classified_n
                        if cid != 0 and classified_n
                        else np.nan
                    ),
                }
            )
    pd.DataFrame(sens_rows).to_csv(
        out_dir / "evolution_threshold_sensitivity.csv", index=False
    )

    cross_rows: list[dict[str, Any]] = []
    for gid, glabel in GROUNDWATER_STATES.items():
        for did, dlabel in DEFORMATION_CLASSES.items():
            count = int(cross_counts[(gid, did)])
            area = float(cross_area[(gid, did)])
            cross_rows.append(
                {
                    "groundwater_state_id": gid,
                    "groundwater_state": glabel,
                    "deformation_class_id": did,
                    "deformation_class": dlabel,
                    "pixel_count": count,
                    "area_km2": area / 1e6,
                    "fraction_of_support_pixels": (
                        count / support_count if support_count else np.nan
                    ),
                }
            )
    pd.DataFrame(cross_rows).to_csv(
        out_dir / "groundwater_deformation_cross_tab.csv", index=False
    )

    result = {
        "status": "ok",
        "method": "same_season_interannual_observation_first_trajectory_classification",
        "analysis_years": [start_year, end_year],
        "same_season_months": [start_month, end_month],
        "max_pair_day_diff": max_day_diff,
        "min_pairs_per_interval": min_pairs,
        "common_intervals": common_labels,
        "early_intervals": early_labels,
        "late_intervals": late_labels,
        "recovery_intervals": recovery_labels,
        "recent_intervals": recent_labels,
        "recent_decline_min_intervals": recent_decline_min_intervals,
        "thresholds": {
            "stable_rate_mm_yr": stable_rate,
            "meaningful_rate_change_mm_yr": meaningful_change,
            "stable_head_change_m_yr": stable_head,
            "meaningful_head_change_m_yr": meaningful_head,
        },
        "support_mask": support_name,
        "support_pixel_count": int(support_count),
        "support_area_km2": float(support_area) / 1e6,
        "groundwater_change_interpolation": groundwater_interp,
        "classification_inputs": [
            "same-season interannual InSAR deformation rate",
            "directly interpolated same-season well groundwater-head change",
            "prior recovery and recent trajectory metrics",
        ],
        "classification_excludes": [
            "Ske",
            "storage partition",
            "hydrostratigraphy",
        ],
        "outputs": {
            "deformation_mode": str(out_dir / "deformation_evolution_mode.tif"),
            "recent_groundwater_state": str(out_dir / "groundwater_recent_state.tif"),
            "coupled_mode": str(out_dir / "coupled_evolution_mode.tif"),
            "summary": str(out_dir / "evolution_mode_summary.csv"),
            "threshold_sensitivity": str(
                out_dir / "evolution_threshold_sensitivity.csv"
            ),
            "groundwater_deformation_cross_tab": str(
                out_dir / "groundwater_deformation_cross_tab.csv"
            ),
            "groundwater_change_spatial_cv": str(
                out_dir / "groundwater_change_spatial_cv.csv"
            ),
            "groundwater_change_spatial_cv_by_interval": str(
                out_dir / "groundwater_change_spatial_cv_by_interval.csv"
            ),
            "groundwater_change_cv_predictions": str(
                out_dir / "groundwater_change_cv_predictions.csv"
            ),
            "groundwater_change_state_confusion": str(
                out_dir / "groundwater_change_state_confusion.csv"
            ),
            "well_same_season_head_change": str(
                out_dir / "well_same_season_head_change.csv"
            ),
        },
    }
    write_json(out_dir / "evolution_summary.json", result)
    return result
