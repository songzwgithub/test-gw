from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Geod
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from ..common import ensure_dir, write_json
from ..config import ProjectConfig

FEATURE_FILES = {
    "quadratic_coeff_mm_yr2": "quadratic_coeff_mm_yr2.tif",
    "linear_coeff_mm_yr": "linear_coeff_mm_yr.tif",
    "vertex_feature_year": "vertex_feature_year.tif",
    "end_rate_mm_yr": "end_rate_mm_yr.tif",
    "start_rate_mm_yr": "start_rate_mm_yr.tif",
}


def _read_feature_stack(deformation_dir: Path, feature_names: list[str]):
    arrays, profile = [], None
    for name in feature_names:
        with rasterio.open(deformation_dir / FEATURE_FILES[name]) as src:
            arr = src.read(1).astype(float)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            if profile is None:
                profile = src.profile.copy()
            arrays.append(arr)
    return np.stack(arrays, axis=-1), profile


def _pixel_spacing_m(profile: dict) -> tuple[float, float]:
    transform = profile["transform"]
    crs = CRS.from_user_input(profile["crs"])
    if not crs.is_geographic:
        factor = float(crs.axis_info[0].unit_conversion_factor or 1.0) if crs.axis_info else 1.0
        x = float(np.hypot(transform.a, transform.d) * factor)
        y = float(np.hypot(transform.b, transform.e) * factor)
        return max(x, 1e-6), max(y, 1e-6)
    h, w = int(profile["height"]), int(profile["width"])
    r, c = h // 2, w // 2
    x0, y0 = rasterio.transform.xy(transform, r, c, offset="center")
    x1, y1 = rasterio.transform.xy(transform, r, min(c + 1, w - 1), offset="center")
    x2, y2 = rasterio.transform.xy(transform, min(r + 1, h - 1), c, offset="center")
    geod = Geod(ellps="WGS84")
    _, _, dx = geod.inv(x0, y0, x1, y1)
    _, _, dy = geod.inv(x0, y0, x2, y2)
    return max(abs(dx), 1e-6), max(abs(dy), 1e-6)


def _balanced_sample_indices(valid: np.ndarray, profile: dict, spacing_km: float, max_samples: int, seed: int):
    dx, dy = _pixel_spacing_m(profile)
    cs = max(1, int(round(float(spacing_km) * 1000.0 / dx)))
    rs = max(1, int(round(float(spacing_km) * 1000.0 / dy)))
    rr = np.arange(0, valid.shape[0], rs)
    cc = np.arange(0, valid.shape[1], cs)
    rgrid, cgrid = np.meshgrid(rr, cc, indexing="ij")
    flat = np.ravel_multi_index((rgrid.ravel(), cgrid.ravel()), valid.shape)
    keep = valid.ravel()[flat]
    idx = flat[keep]
    if len(idx) < min(100, valid.sum()):
        idx = np.flatnonzero(valid)
    if len(idx) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, max_samples, replace=False)
    return np.asarray(idx, dtype=int)


def _suggest_label(start_rate: float, end_rate: float, stable_rate: float, decel_threshold: float) -> str:
    if start_rate < -stable_rate and end_rate > stable_rate:
        return "subsidence_to_rebound"
    if start_rate > stable_rate and end_rate > stable_rate:
        return "persistent_uplift"
    if abs(end_rate) <= stable_rate:
        return "stable"
    if end_rate < -stable_rate and (end_rate - start_rate) > decel_threshold:
        return "reduced_subsidence"
    return "continuous_subsidence"


def classify_deformation(cfg: ProjectConfig) -> dict[str, Any]:
    sec = cfg.section("regimes")
    deformation_dir = cfg.outputs / "deformation"
    out_dir = ensure_dir(cfg.outputs / "regimes")
    if str(sec.get("feature_scheme", "meng2026")).lower() != "meng2026":
        raise ValueError("v0.3 supports regimes.feature_scheme='meng2026'")

    feature_names = [
        "quadratic_coeff_mm_yr2",
        "linear_coeff_mm_yr",
        "vertex_feature_year",
        "end_rate_mm_yr",
    ]
    data, profile = _read_feature_stack(deformation_dir, feature_names)
    start_rate = _read_feature_stack(deformation_dir, ["start_rate_mm_yr"])[0][..., 0]
    valid = np.isfinite(data).all(axis=-1) & np.isfinite(start_rate)
    if valid.sum() < 100:
        raise ValueError("Too few valid pixels for deformation clustering")

    seed = int(sec.get("random_state", 20260919))
    train_idx = _balanced_sample_indices(
        valid,
        profile,
        spacing_km=float(sec.get("training_spacing_km", 2.0)),
        max_samples=int(sec.get("selection_sample_size", 50000)),
        seed=seed,
    )
    Xtrain = data.reshape(-1, data.shape[-1])[train_idx]
    scaler = StandardScaler().fit(Xtrain)
    Xsel = scaler.transform(Xtrain)

    candidates = [int(k) for k in sec.get("k_candidates", [2, 3, 4, 5, 6]) if 2 <= int(k) < len(Xsel)]
    if not candidates:
        raise ValueError("No valid K candidates for clustering")
    selection_rows, best_k, best_score = [], None, -np.inf
    for k in candidates:
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = km.fit_predict(Xsel)
        sil = float(silhouette_score(Xsel, labels))
        ch = float(calinski_harabasz_score(Xsel, labels))
        db = float(davies_bouldin_score(Xsel, labels))
        selection_rows.append({"k": k, "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db, "inertia": float(km.inertia_)})
        if sil > best_score:
            best_k, best_score = k, sil

    k = int(sec["k"]) if sec.get("k") is not None else int(best_k)
    model = KMeans(n_clusters=k, random_state=seed, n_init=30).fit(Xsel)

    flat = data.reshape(-1, data.shape[-1])
    valid_idx = np.flatnonzero(valid.ravel())
    labels_all = np.empty(len(valid_idx), dtype=np.int16)
    chunk = int(sec.get("predict_chunk_size", 200000))
    for i0 in range(0, len(valid_idx), chunk):
        ii = valid_idx[i0:i0 + chunk]
        labels_all[i0:i0 + chunk] = model.predict(scaler.transform(flat[ii]))

    cluster_map = np.zeros(valid.shape, dtype="uint8")
    cluster_map.ravel()[valid_idx] = labels_all.astype("uint8") + 1
    cluster_path = out_dir / "deformation_regime_id.tif"
    out_profile = profile.copy()
    out_profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0, compress="deflate", tiled=True)
    with rasterio.open(cluster_path, "w", **out_profile) as dst:
        dst.write(cluster_map, 1)

    stable_rate = float(sec.get("stable_rate_mm_yr", 5.0))
    decel_threshold = float(sec.get("deceleration_threshold_mm_yr", 5.0))
    flat_valid = flat[valid_idx]
    flat_start = start_rate.ravel()[valid_idx]
    summary_rows = []
    for cid in range(k):
        m = labels_all == cid
        med = {name: float(np.nanmedian(flat_valid[m, j])) for j, name in enumerate(feature_names)}
        srate = float(np.nanmedian(flat_start[m]))
        summary_rows.append({
            "cluster_id": cid + 1,
            "suggested_label": _suggest_label(srate, med["end_rate_mm_yr"], stable_rate, decel_threshold),
            "pixel_count": int(m.sum()),
            "start_rate_mm_yr": srate,
            **med,
        })

    pd.DataFrame(selection_rows).to_csv(out_dir / "k_selection.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "cluster_summary.csv", index=False)
    np.savez_compressed(
        out_dir / "clustering_model.npz",
        mean=scaler.mean_, scale=scaler.scale_, centers=model.cluster_centers_,
        feature_names=np.asarray(feature_names), k=k,
    )
    result = {
        "status": "ok",
        "feature_scheme": "meng2026",
        "k": k,
        "selection_metric": "silhouette",
        "training_spacing_km": float(sec.get("training_spacing_km", 2.0)),
        "training_pixels": int(len(train_idx)),
        "features": feature_names,
        "cluster_map": str(cluster_path),
        "clusters": summary_rows,
    }
    write_json(out_dir / "regime_summary.json", result)
    return result
