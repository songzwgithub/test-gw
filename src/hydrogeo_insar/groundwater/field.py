from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial.distance import cdist

from ..common import affine_to_list, block_slices, days_to_dates, ensure_dir, h5_grid_metadata, write_json
from ..config import ProjectConfig


@dataclass
class LowRankRBFModel:
    station_ids: list[str]
    station_lonlat: np.ndarray
    station_xy: np.ndarray
    rank: int
    temporal_dates: np.ndarray
    temporal_components: np.ndarray  # rank x time
    score_coefficients: np.ndarray   # (1+n_centers) x rank
    centers_xy: np.ndarray
    sigma_m: float
    projected_crs: str
    baseline_start: str
    baseline_end: str


def _auto_utm_epsg(lon: float, lat: float) -> str:
    zone = int((lon + 180.0) // 6.0) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def _farthest_centers(points: np.ndarray, n: int) -> np.ndarray:
    n = min(max(1, n), len(points))
    selected = [int(np.argmin(points[:, 0] + points[:, 1]))]
    min_dist = cdist(points, points[selected]).ravel()
    while len(selected) < n:
        j = int(np.argmax(min_dist))
        selected.append(j)
        min_dist = np.minimum(min_dist, cdist(points, points[[j]]).ravel())
    return points[selected]


def _rbf(points: np.ndarray, centers: np.ndarray, sigma_m: float) -> np.ndarray:
    return np.exp(-0.5 * (cdist(points, centers) / float(sigma_m)) ** 2)


def _interpolate_short_gaps(series: pd.Series, max_gap_days: int) -> pd.Series:
    if max_gap_days <= 0:
        return series
    s = series.copy()
    miss = s.isna()
    if not miss.any():
        return s
    runs = miss.ne(miss.shift()).cumsum()
    interp = s.interpolate(method="time", limit_area="inside")
    for _, idx in miss[miss].groupby(runs).groups.items():
        if len(idx) <= max_gap_days:
            s.loc[idx] = interp.loc[idx]
    return s


def _iterative_svd_impute(matrix: np.ndarray, rank: int, n_iter: int = 12, tol: float = 1e-5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = np.isfinite(matrix)
    if not obs.any():
        raise ValueError("Groundwater matrix contains no finite values")
    filled = matrix.copy()
    # Initialize each well anomaly with zero at missing values; anomaly centering makes this natural.
    filled[~obs] = 0.0
    prev = filled.copy()
    for _ in range(n_iter):
        u, s, vt = np.linalg.svd(filled, full_matrices=False)
        r = min(rank, len(s))
        recon = (u[:, :r] * s[:r]) @ vt[:r]
        filled[~obs] = recon[~obs]
        diff = np.nanmean((filled - prev) ** 2)
        prev[:] = filled
        if diff < tol:
            break
    u, s, vt = np.linalg.svd(filled, full_matrices=False)
    r = min(rank, len(s))
    scores = u[:, :r] * s[:r]
    components = vt[:r]
    return filled, scores, components


def fit_groundwater_model(cfg: ProjectConfig) -> tuple[LowRankRBFModel, pd.DataFrame]:
    sec = cfg.section("groundwater_field")
    aquifer = str(sec.get("aquifer", "confined"))
    gw = pd.read_csv(cfg.outputs / "canonical" / "groundwater.csv", parse_dates=["date"])
    gw = gw[gw["aquifer_class"].astype(str).str.lower() == aquifer.lower()].copy()
    if gw.empty:
        raise ValueError(f"No groundwater observations for aquifer_class={aquifer!r}")

    analysis = cfg.section("analysis")
    start = pd.Timestamp(analysis.get("start_date", gw["date"].min()))
    end = pd.Timestamp(analysis.get("end_date", gw["date"].max()))
    dates = pd.date_range(start, end, freq="D")
    max_gap = int(sec.get("max_gap_days", 7))

    meta = gw.groupby("station_id", as_index=False).agg(lon=("lon", "first"), lat=("lat", "first"))
    matrix_rows = []
    kept_ids = []
    baseline_cfg = sec.get("baseline", {})
    bstart = pd.Timestamp(baseline_cfg.get("start", start))
    bend = pd.Timestamp(baseline_cfg.get("end", min(end, start + pd.Timedelta(days=365))))

    for sid, group in gw.groupby("station_id"):
        s = group.groupby("date")["head_m"].median().reindex(dates)
        s = _interpolate_short_gaps(s, max_gap)
        baseline = s.loc[(s.index >= bstart) & (s.index <= bend)].median()
        if not np.isfinite(baseline):
            continue
        anomaly = s - baseline
        matrix_rows.append(anomaly.to_numpy(float))
        kept_ids.append(str(sid))

    if len(matrix_rows) < 3:
        raise ValueError("At least three wells with a valid baseline are required")
    matrix = np.vstack(matrix_rows)
    min_coverage = float(sec.get("min_coverage_fraction", 0.5))
    coverage = np.isfinite(matrix).mean(axis=1)
    keep_rows = coverage >= min_coverage
    matrix = matrix[keep_rows]
    kept_ids = [sid for sid, keep in zip(kept_ids, keep_rows) if keep]
    if len(kept_ids) < 3:
        raise ValueError("Too few groundwater wells after coverage filtering")
    meta = meta[meta["station_id"].astype(str).isin(kept_ids)].copy()
    meta["station_id"] = meta["station_id"].astype(str)
    meta = meta.set_index("station_id").loc[kept_ids].reset_index()

    rank = int(sec.get("rank", min(4, max(1, len(kept_ids) - 1))))
    _filled, scores, components = _iterative_svd_impute(matrix, rank=rank, n_iter=int(sec.get("svd_iterations", 12)))

    lonlat = meta[["lon", "lat"]].to_numpy(float)
    projected_crs = sec.get("projected_crs") or _auto_utm_epsg(float(np.nanmean(lonlat[:, 0])), float(np.nanmean(lonlat[:, 1])))
    transformer = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    xx, yy = transformer.transform(lonlat[:, 0], lonlat[:, 1])
    xy = np.column_stack([xx, yy])

    rbf_cfg = sec.get("rbf", {})
    ncenters = int(rbf_cfg.get("max_centers", min(32, len(xy))))
    centers = _farthest_centers(xy, ncenters)
    sigma_km = rbf_cfg.get("sigma_km")
    if sigma_km is None:
        d = cdist(xy, xy)
        vals = d[d > 0]
        sigma_m = float(np.median(vals)) if vals.size else 10000.0
    else:
        sigma_m = float(sigma_km) * 1000.0
    basis = np.column_stack([np.ones(len(xy)), _rbf(xy, centers, sigma_m)])
    ridge = float(rbf_cfg.get("ridge", 1e-3))
    penalty = ridge * np.eye(basis.shape[1])
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(basis.T @ basis + penalty, basis.T @ scores)

    model = LowRankRBFModel(
        station_ids=kept_ids,
        station_lonlat=lonlat,
        station_xy=xy,
        rank=components.shape[0],
        temporal_dates=dates.to_numpy(dtype="datetime64[D]"),
        temporal_components=components,
        score_coefficients=coef,
        centers_xy=centers,
        sigma_m=sigma_m,
        projected_crs=str(projected_crs),
        baseline_start=str(bstart.date()),
        baseline_end=str(bend.date()),
    )
    score_table = meta.copy()
    for k in range(model.rank):
        score_table[f"score_{k+1}"] = scores[:, k]
    return model, score_table


def _predict_scores(model: LowRankRBFModel, lonlat: np.ndarray) -> np.ndarray:
    transformer = Transformer.from_crs("EPSG:4326", model.projected_crs, always_xy=True)
    x, y = transformer.transform(lonlat[:, 0], lonlat[:, 1])
    xy = np.column_stack([x, y])
    b = np.column_stack([np.ones(len(xy)), _rbf(xy, model.centers_xy, model.sigma_m)])
    return b @ model.score_coefficients


def build_groundwater_field(cfg: ProjectConfig) -> dict[str, Any]:
    model, score_table = fit_groundwater_model(cfg)
    insar_path = cfg.outputs / "canonical" / "insar_stack.h5"
    grid = h5_grid_metadata(insar_path)
    with h5py.File(insar_path, "r") as ih5:
        obs_dates = days_to_dates(ih5["date_days"][:])

    # Temporal component interpolation from daily model dates to InSAR epochs.
    src_days = model.temporal_dates.astype("datetime64[D]").astype(np.int64)
    dst_days = obs_dates.astype("datetime64[D]").astype(np.int64)
    comp = np.vstack([np.interp(dst_days, src_days, row, left=np.nan, right=np.nan) for row in model.temporal_components])

    out_dir = ensure_dir(cfg.outputs / "groundwater")
    out_path = out_dir / "groundwater_field.h5"
    block_size = int(cfg.section("groundwater_field").get("grid_block_size", 256))

    import rasterio
    from rasterio.transform import xy as raster_xy

    transform = grid["transform"]
    height, width = grid["height"], grid["width"]
    with h5py.File(out_path, "w") as h5:
        ds = h5.create_dataset(
            "head_anomaly_m",
            shape=(len(obs_dates), height, width),
            dtype="float32",
            chunks=(1, min(block_size, height), min(block_size, width)),
            compression="gzip",
            compression_opts=4,
            fillvalue=np.nan,
        )
        h5.create_dataset("date_days", data=(obs_dates - np.datetime64("1970-01-01", "D")).astype(np.int32))
        h5.attrs["height"] = height
        h5.attrs["width"] = width
        h5.attrs["crs"] = grid["crs"]
        h5.attrs["transform"] = affine_to_list(transform)
        h5.attrs["aquifer_class"] = cfg.section("groundwater_field").get("aquifer", "confined")
        h5.attrs["baseline_start"] = model.baseline_start
        h5.attrs["baseline_end"] = model.baseline_end
        h5.attrs["method"] = "lowrank_rbf"

        # We need grid cell centers in geographic coordinates for the RBF predictor.
        grid_to_geo = Transformer.from_crs(grid["crs"], "EPSG:4326", always_xy=True)
        for r0, r1, c0, c1 in block_slices(height, width, block_size):
            rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
            xs, ys = raster_xy(transform, rr.ravel(), cc.ravel(), offset="center")
            lon, lat = grid_to_geo.transform(np.asarray(xs), np.asarray(ys))
            scores = _predict_scores(model, np.column_stack([lon, lat]))  # P x rank
            values = (scores @ comp).T.reshape(len(obs_dates), r1-r0, c1-c0)
            ds[:, r0:r1, c0:c1] = values.astype("float32")

    model_path = out_dir / "groundwater_model.npz"
    np.savez_compressed(
        model_path,
        station_ids=np.asarray(model.station_ids),
        station_lonlat=model.station_lonlat,
        rank=model.rank,
        temporal_dates=model.temporal_dates.astype("datetime64[D]").astype(str),
        temporal_components=model.temporal_components,
        score_coefficients=model.score_coefficients,
        centers_xy=model.centers_xy,
        sigma_m=model.sigma_m,
        projected_crs=model.projected_crs,
        baseline_start=model.baseline_start,
        baseline_end=model.baseline_end,
    )
    score_table.to_csv(out_dir / "groundwater_spatial_scores.csv", index=False)
    summary = {
        "status": "ok",
        "output": str(out_path),
        "n_wells": len(model.station_ids),
        "rank": model.rank,
        "projected_crs": model.projected_crs,
        "sigma_m": model.sigma_m,
        "baseline_start": model.baseline_start,
        "baseline_end": model.baseline_end,
        "epochs": len(obs_dates),
    }
    write_json(out_dir / "groundwater_field_summary.json", summary)
    return summary
