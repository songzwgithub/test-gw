from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol, xy as raster_xy
from scipy import sparse
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree

from ..common import block_slices, ensure_dir, read_tif, write_json, write_tif
from ..config import ProjectConfig
from .seasonal import rotate_coefficients


def _projected_xy(transform, grid_crs: str, projected_crs: str, rr: np.ndarray, cc: np.ndarray):
    xs, ys = raster_xy(transform, rr, cc, offset="center")
    xs, ys = np.asarray(xs), np.asarray(ys)
    if str(grid_crs) == str(projected_crs):
        return xs, ys
    tr = Transformer.from_crs(grid_crs, projected_crs, always_xy=True)
    return tr.transform(xs, ys)


def _aggregate_observations(
    valid: np.ndarray,
    q2: np.ndarray,
    qd: np.ndarray,
    d2: np.ndarray,
    quality: np.ndarray,
    transform,
    grid_crs: str,
    projected_crs: str,
    cell_km: float,
    block_size: int = 256,
):
    cell_m = float(cell_km) * 1000.0
    acc: dict[tuple[int, int], np.ndarray] = {}
    h, w = valid.shape
    for r0, r1, c0, c1 in block_slices(h, w, block_size):
        m = valid[r0:r1, c0:c1]
        if not m.any():
            continue
        rr, cc = np.nonzero(m)
        rr = rr + r0; cc = cc + c0
        xx, yy = _projected_xy(transform, grid_crs, projected_crs, rr, cc)
        bx = np.floor(np.asarray(xx) / cell_m).astype(np.int64)
        by = np.floor(np.asarray(yy) / cell_m).astype(np.int64)
        keys = np.column_stack([bx, by])
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        fq2 = q2[rr, cc]; fqd = qd[rr, cc]; fd2 = d2[rr, cc]; fw = quality[rr, cc]
        for j, (ix, iy) in enumerate(uniq):
            sel = inv == j
            val = np.array([
                np.sum(fq2[sel]), np.sum(fqd[sel]), np.sum(fd2[sel]), np.sum(fw[sel]), np.sum(sel)
            ], dtype=float)
            key = (int(ix), int(iy))
            acc[key] = acc.get(key, np.zeros(5, dtype=float)) + val
    rows = []
    for (ix, iy), v in acc.items():
        if v[0] <= 0 or v[3] <= 0:
            continue
        rows.append({
            "x": (ix + 0.5) * cell_m,
            "y": (iy + 0.5) * cell_m,
            "q2": v[0], "qd": v[1], "d2": v[2], "quality_sum": v[3], "n_pixels": int(v[4]),
            "raw_ske": v[1] / v[0],
        })
    return pd.DataFrame(rows)


def _node_grid(gw_support: np.ndarray, transform, grid_crs: str, projected_crs: str, obs_xy: np.ndarray, spacing_km: float, max_extrapolation_km: float):
    spacing = float(spacing_km) * 1000.0
    h, w = gw_support.shape
    corners_r = np.array([0, 0, h - 1, h - 1])
    corners_c = np.array([0, w - 1, 0, w - 1])
    cx, cy = _projected_xy(transform, grid_crs, projected_crs, corners_r, corners_c)
    xs = np.arange(np.floor(np.min(cx) / spacing) * spacing, np.ceil(np.max(cx) / spacing) * spacing + 0.5 * spacing, spacing)
    ys = np.arange(np.floor(np.min(cy) / spacing) * spacing, np.ceil(np.max(cy) / spacing) * spacing + 0.5 * spacing, spacing)
    gx, gy = np.meshgrid(xs, ys)
    nodes = np.column_stack([gx.ravel(), gy.ravel()])

    back = Transformer.from_crs(projected_crs, grid_crs, always_xy=True)
    rx, ry = back.transform(nodes[:, 0], nodes[:, 1])
    rr, cc = rowcol(transform, rx, ry)
    rr = np.asarray(rr); cc = np.asarray(cc)
    inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    keep = np.zeros(len(nodes), dtype=bool)
    ii = np.flatnonzero(inside)
    keep[ii] = gw_support[rr[ii], cc[ii]]
    if obs_xy.size:
        dist, _ = cKDTree(obs_xy).query(nodes, k=1)
        keep &= dist <= float(max_extrapolation_km) * 1000.0
    return nodes[keep]


def _normalized_rbf_basis(query_xy: np.ndarray, nodes_xy: np.ndarray, sigma_m: float, max_neighbors: int = 12):
    k = min(max(1, int(max_neighbors)), len(nodes_xy))
    dist, ind = cKDTree(nodes_xy).query(query_xy, k=k)
    if k == 1:
        dist = dist[:, None]; ind = ind[:, None]
    phi = np.exp(-0.5 * (dist / float(sigma_m)) ** 2)
    phi /= np.maximum(phi.sum(axis=1, keepdims=True), 1e-12)
    rows = np.repeat(np.arange(len(query_xy)), k)
    return sparse.csr_matrix((phi.ravel(), (rows, ind.ravel())), shape=(len(query_xy), len(nodes_xy)))


def _graph_incidence(nodes_xy: np.ndarray, spacing_m: float):
    pairs = cKDTree(nodes_xy).query_pairs(r=1.5 * float(spacing_m), output_type="ndarray")
    if pairs.size == 0:
        return sparse.csr_matrix((0, len(nodes_xy)))
    nedge = len(pairs)
    rows = np.repeat(np.arange(nedge), 2)
    cols = pairs.ravel()
    vals = np.tile([1.0, -1.0], nedge)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(nedge, len(nodes_xy)))


def _solve(obs: pd.DataFrame, nodes_xy: np.ndarray, spacing_m: float, sigma_factor: float, lam: float, bounds: tuple[float, float], max_neighbors: int, train: np.ndarray | None = None):
    if train is None:
        train = np.ones(len(obs), dtype=bool)
    B = _normalized_rbf_basis(obs.loc[train, ["x", "y"]].to_numpy(float), nodes_xy, spacing_m * sigma_factor, max_neighbors)
    q2 = obs.loc[train, "q2"].to_numpy(float)
    raw = obs.loc[train, "raw_ske"].to_numpy(float)
    qscale = max(float(np.median(q2[q2 > 0])), 1e-12)
    sw = np.sqrt(q2 / qscale)
    Adata = sparse.diags(sw) @ B
    bdata = sw * raw
    L = _graph_incidence(nodes_xy, spacing_m)
    if lam > 0 and L.shape[0] > 0:
        A = sparse.vstack([Adata, np.sqrt(float(lam)) * L], format="csr")
        b = np.concatenate([bdata, np.zeros(L.shape[0])])
    else:
        A, b = Adata, bdata
    res = lsq_linear(A, b, bounds=bounds, method="trf", lsq_solver="lsmr", tol=1e-8, max_iter=400)
    return res.x


def _fold_ids(xy: np.ndarray, block_km: float):
    b = float(block_km) * 1000.0
    ix = np.floor(xy[:, 0] / b).astype(int)
    iy = np.floor(xy[:, 1] / b).astype(int)
    return (ix + 2 * iy) % 5


def _cv_models(obs: pd.DataFrame, gw_support: np.ndarray, transform, grid_crs: str, projected_crs: str, sec: dict[str, Any]):
    spacing_candidates = [float(v) for v in sec.get("node_spacing_km_candidates", [10, 15, 20, 25])]
    lambda_candidates = [float(v) for v in sec.get("lambda_candidates", [0.1, 1.0, 10.0, 100.0])]
    max_extrap = float(sec.get("max_extrapolation_km", 30.0))
    sigma_factor = float(sec.get("basis_sigma_factor", 1.5))
    max_neighbors = int(sec.get("max_basis_neighbors", 12))
    folds = _fold_ids(obs[["x", "y"]].to_numpy(float), float(sec.get("cv_block_km", 30.0)))
    rows = []
    obs_xy = obs[["x", "y"]].to_numpy(float)

    for spacing_km in spacing_candidates:
        nodes = _node_grid(gw_support, transform, grid_crs, projected_crs, obs_xy, spacing_km, max_extrap)
        if len(nodes) < 3:
            continue
        B_all = _normalized_rbf_basis(obs_xy, nodes, spacing_km * 1000.0 * sigma_factor, max_neighbors)
        for lam in lambda_candidates:
            sse = 0.0; wsum = 0.0; tested = 0
            for fold in range(5):
                train = folds != fold
                hold = ~train
                if train.sum() < 3 or hold.sum() == 0:
                    continue
                beta = _solve(obs, nodes, spacing_km * 1000.0, sigma_factor, lam, (float(sec.get("min", 0.0)), float(sec.get("max", 0.05))), max_neighbors, train=train)
                pred = np.asarray(B_all[hold] @ beta).ravel()
                q2 = obs.loc[hold, "q2"].to_numpy(float)
                qd = obs.loc[hold, "qd"].to_numpy(float)
                d2 = obs.loc[hold, "d2"].to_numpy(float)
                qw = obs.loc[hold, "quality_sum"].to_numpy(float)
                sse += float(np.sum(np.maximum(d2 - 2.0 * pred * qd + pred * pred * q2, 0.0)))
                wsum += float(np.sum(qw))
                tested += int(hold.sum())
            rows.append({
                "node_spacing_km": spacing_km,
                "lambda": lam,
                "cv_deformation_rmse_m": float(np.sqrt(sse / wsum)) if wsum > 0 else np.inf,
                "nodes": int(len(nodes)),
                "tested_cells": tested,
            })
    table = pd.DataFrame(rows).sort_values(["cv_deformation_rmse_m", "node_spacing_km", "lambda"]).reset_index(drop=True)
    if table.empty or not np.isfinite(table.iloc[0]["cv_deformation_rmse_m"]):
        raise ValueError("Ske spatial cross-validation produced no valid model")
    return table


def estimate_regularized_ske(cfg: ProjectConfig) -> dict[str, Any]:
    """Continuous effective Ske inversion on physical-kilometre basis nodes."""
    sec = cfg.section("ske")
    seasonal = cfg.outputs / "seasonal"
    lag = float(json.loads((seasonal / "lag_summary.json").read_text(encoding="utf-8"))["lag_days"])
    period = float(cfg.section("seasonal_response").get("annual_period_days", 365.2425))

    ds = read_tif(seasonal / "deformation_annual_sin_mm.tif") / 1000.0
    dc = read_tif(seasonal / "deformation_annual_cos_mm.tif") / 1000.0
    hs = read_tif(seasonal / "head_annual_sin_m.tif")
    hc = read_tif(seasonal / "head_annual_cos_m.tif")
    drmse = read_tif(seasonal / "deformation_fit_rmse_mm.tif") / 1000.0
    hrmse = read_tif(seasonal / "head_fit_rmse_m.tif")
    rhs, rhc = rotate_coefficients(hs, hc, lag, period)

    hamp = np.hypot(rhs, rhc)
    damp = np.hypot(ds, dc)
    gw_summary = json.loads((cfg.outputs / "groundwater" / "groundwater_field_summary.json").read_text(encoding="utf-8"))
    gw_cv_harmonic_rmse = float(gw_summary.get("cv_harmonic_vector_rmse_m", 0.0))
    if not np.isfinite(gw_cv_harmonic_rmse) or gw_cv_harmonic_rmse < 0:
        gw_cv_harmonic_rmse = 0.0
    head_sigma = np.sqrt(hrmse * hrmse + gw_cv_harmonic_rmse * gw_cv_harmonic_rmse)

    dot = ds * rhs + dc * rhc
    vector_cosine = dot / np.maximum(damp * hamp, 1e-12)
    coupling = np.clip(vector_cosine, 0.0, 1.0)
    quality = 1.0 / (
        1.0
        + (drmse / np.maximum(damp, 1e-9)) ** 2
        + (head_sigma / np.maximum(hamp, 1e-9)) ** 2
    )
    quality *= coupling
    q2 = quality * (rhs * rhs + rhc * rhc)
    qd = quality * dot
    d2 = quality * (ds * ds + dc * dc)

    sresp = cfg.section("seasonal_response")
    min_vector_cosine = float(sec.get("min_vector_cosine", 0.0))
    data_valid = np.isfinite(q2) & np.isfinite(qd) & np.isfinite(d2) & np.isfinite(quality)
    data_valid &= np.isfinite(vector_cosine) & (vector_cosine > min_vector_cosine)
    data_valid &= hamp >= float(sresp.get("min_head_amplitude_m", 0.2))
    data_valid &= damp * 1000.0 >= float(sresp.get("min_deformation_amplitude_mm", 0.5))
    gw_support = read_tif(cfg.outputs / "groundwater" / "groundwater_support_mask.tif") > 0
    data_valid &= gw_support

    with rasterio.open(seasonal / "head_annual_sin_m.tif") as ref:
        transform = ref.transform; grid_crs = str(ref.crs); height = ref.height; width = ref.width
    projected_crs = str(gw_summary["projected_crs"])

    obs = _aggregate_observations(
        data_valid, q2, qd, d2, quality, transform, grid_crs, projected_crs,
        cell_km=float(sec.get("observation_cell_km", 2.5)),
        block_size=int(sec.get("aggregation_block_size", 256)),
    )
    if len(obs) < 10:
        raise ValueError("Too few seasonal observation cells for Ske inversion")

    cv_table = _cv_models(obs, gw_support, transform, grid_crs, projected_crs, sec)
    best = cv_table.iloc[0]
    spacing_km = float(best["node_spacing_km"])
    lam = float(best["lambda"])
    max_extrap = float(sec.get("max_extrapolation_km", 30.0))
    sigma_factor = float(sec.get("basis_sigma_factor", 1.5))
    max_neighbors = int(sec.get("max_basis_neighbors", 12))
    obs_xy = obs[["x", "y"]].to_numpy(float)
    nodes = _node_grid(gw_support, transform, grid_crs, projected_crs, obs_xy, spacing_km, max_extrap)
    bounds = (float(sec.get("min", 0.0)), float(sec.get("max", 0.05)))
    beta = _solve(obs, nodes, spacing_km * 1000.0, sigma_factor, lam, bounds, max_neighbors)

    out_dir = ensure_dir(cfg.outputs / "seasonal")
    raw = np.full_like(ds, np.nan, dtype=float)
    raw[data_valid] = qd[data_valid] / np.maximum(q2[data_valid], 1e-12)
    write_tif(out_dir / "ske_raw_ratio.tif", raw.astype("float32"), grid_crs, transform)
    write_tif(out_dir / "seasonal_vector_cosine.tif", vector_cosine.astype("float32"), grid_crs, transform)
    write_tif(out_dir / "ske_data_support_mask.tif", data_valid.astype("uint8"), grid_crs, transform, nodata=0, dtype="uint8")

    ske = np.full((height, width), np.nan, dtype="float32")
    support = np.zeros((height, width), dtype=bool)
    obs_tree = cKDTree(obs_xy)
    block_size = int(sec.get("rasterize_block_size", 256))
    for r0, r1, c0, c1 in block_slices(height, width, block_size):
        gm = gw_support[r0:r1, c0:c1]
        if not gm.any():
            continue
        rr, cc = np.nonzero(gm)
        gr = rr + r0; gc = cc + c0
        xx, yy = _projected_xy(transform, grid_crs, projected_crs, gr, gc)
        qxy = np.column_stack([xx, yy])
        dist, _ = obs_tree.query(qxy, k=1)
        keep = dist <= max_extrap * 1000.0
        if not keep.any():
            continue
        B = _normalized_rbf_basis(qxy[keep], nodes, spacing_km * 1000.0 * sigma_factor, max_neighbors)
        pred = np.asarray(B @ beta).ravel()
        br = gr[keep]; bc = gc[keep]
        ske[br, bc] = pred.astype("float32")
        support[br, bc] = True

    write_tif(out_dir / "ske_effective.tif", ske, grid_crs, transform)
    write_tif(out_dir / "ske_support_mask.tif", support.astype("uint8"), grid_crs, transform, nodata=0, dtype="uint8")
    cv_table.to_csv(out_dir / "ske_model_cv.csv", index=False)
    pd.DataFrame({"x": nodes[:, 0], "y": nodes[:, 1], "ske": beta}).to_csv(out_dir / "ske_basis_nodes.csv", index=False)

    summary = {
        "status": "ok",
        "lag_days": lag,
        "regularization": "bounded_normalized_rbf_basis_with_graph_smoothing",
        "observation_cell_km": float(sec.get("observation_cell_km", 2.5)),
        "node_spacing_km": spacing_km,
        "basis_sigma_factor": sigma_factor,
        "max_extrapolation_km": max_extrap,
        "min_vector_cosine": min_vector_cosine,
        "groundwater_cv_harmonic_rmse_m": gw_cv_harmonic_rmse,
        "lambda": lam,
        "observation_cells": int(len(obs)),
        "basis_nodes": int(len(nodes)),
        "support_pixels": int(support.sum()),
        "ske_median": float(np.nanmedian(ske)),
        "ske_p10": float(np.nanpercentile(ske, 10)),
        "ske_p90": float(np.nanpercentile(ske, 90)),
        "bounds": list(bounds),
        "cv_deformation_rmse_m": float(best["cv_deformation_rmse_m"]),
    }
    write_json(out_dir / "ske_summary.json", summary)
    return summary
