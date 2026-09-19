from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from ..common import ensure_dir, write_json
from ..config import ProjectConfig

FEATURE_FILES = {
    "quadratic_coeff_mm_yr2": "quadratic_coeff_mm_yr2.tif",
    "start_rate_mm_yr": "start_rate_mm_yr.tif",
    "end_rate_mm_yr": "end_rate_mm_yr.tif",
    "rate_change_mm_yr": "rate_change_mm_yr.tif",
    "vertex_time_year": "vertex_time_year.tif",
}


def _read_feature_stack(deformation_dir: Path, feature_names: list[str]):
    arrays = []
    profile = None
    for name in feature_names:
        path = deformation_dir / FEATURE_FILES[name]
        with rasterio.open(path) as src:
            arr = src.read(1).astype(float)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            if profile is None:
                profile = src.profile.copy()
            arrays.append(arr)
    stack = np.stack(arrays, axis=-1)
    return stack, profile


def _suggest_label(end_rate: float, rate_change: float, stable_rate: float, strong_deceleration: float) -> str:
    if end_rate > stable_rate:
        return "rebound"
    if abs(end_rate) <= stable_rate:
        return "stable"
    if end_rate < -stable_rate and rate_change > strong_deceleration:
        return "decelerating_subsidence"
    return "continuous_subsidence"


def classify_deformation(cfg: ProjectConfig) -> dict[str, Any]:
    sec = cfg.section("regimes")
    deformation_dir = cfg.outputs / "deformation"
    out_dir = ensure_dir(cfg.outputs / "regimes")
    feature_names = sec.get("features", [
        "quadratic_coeff_mm_yr2",
        "start_rate_mm_yr",
        "end_rate_mm_yr",
        "rate_change_mm_yr",
        "vertex_time_year",
    ])
    feature_names = [f for f in feature_names if f in FEATURE_FILES]
    data, profile = _read_feature_stack(deformation_dir, feature_names)
    valid = np.isfinite(data).all(axis=-1)
    X = data[valid]
    if len(X) < 100:
        raise ValueError("Too few valid pixels for deformation clustering")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rng = np.random.default_rng(int(sec.get("random_state", 20260919)))
    max_samples = int(sec.get("selection_sample_size", 50000))
    sample_idx = np.arange(len(Xs))
    if len(sample_idx) > max_samples:
        sample_idx = rng.choice(sample_idx, max_samples, replace=False)
    Xsel = Xs[sample_idx]

    candidates = [int(k) for k in sec.get("k_candidates", [3, 4, 5]) if int(k) >= 2]
    selection_rows = []
    best_k = None
    best_score = -np.inf
    for k in candidates:
        km = KMeans(n_clusters=k, random_state=int(sec.get("random_state", 20260919)), n_init=20)
        labels = km.fit_predict(Xsel)
        sil = float(silhouette_score(Xsel, labels))
        ch = float(calinski_harabasz_score(Xsel, labels))
        db = float(davies_bouldin_score(Xsel, labels))
        selection_rows.append({"k": k, "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db})
        if sil > best_score:
            best_score = sil
            best_k = k

    k = int(sec.get("k", best_k)) if sec.get("k") is not None else int(best_k)
    model = KMeans(n_clusters=k, random_state=int(sec.get("random_state", 20260919)), n_init=30)
    labels = model.fit_predict(Xs)

    cluster_map = np.zeros(valid.shape, dtype="uint8")
    cluster_map[valid] = labels.astype("uint8") + 1
    cluster_path = out_dir / "deformation_regime_id.tif"
    out_profile = profile.copy()
    out_profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0, compress="deflate", tiled=True)
    with rasterio.open(cluster_path, "w", **out_profile) as dst:
        dst.write(cluster_map, 1)

    stable_rate = float(sec.get("stable_rate_mm_yr", 5.0))
    strong_deceleration = float(sec.get("deceleration_threshold_mm_yr", 5.0))
    summary_rows = []
    flat_data = data[valid]
    for cid in range(k):
        mask = labels == cid
        med = {name: float(np.nanmedian(flat_data[mask, j])) for j, name in enumerate(feature_names)}
        label = _suggest_label(
            med.get("end_rate_mm_yr", np.nan),
            med.get("rate_change_mm_yr", np.nan),
            stable_rate,
            strong_deceleration,
        )
        summary_rows.append({
            "cluster_id": cid + 1,
            "suggested_label": label,
            "pixel_count": int(mask.sum()),
            **med,
        })

    pd.DataFrame(selection_rows).to_csv(out_dir / "k_selection.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "cluster_summary.csv", index=False)
    np.savez_compressed(
        out_dir / "clustering_model.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        centers=model.cluster_centers_,
        feature_names=np.asarray(feature_names),
        k=k,
    )
    result = {
        "status": "ok",
        "k": k,
        "selection_metric": "silhouette",
        "features": feature_names,
        "cluster_map": str(cluster_path),
        "clusters": summary_rows,
    }
    write_json(out_dir / "regime_summary.json", result)
    return result
