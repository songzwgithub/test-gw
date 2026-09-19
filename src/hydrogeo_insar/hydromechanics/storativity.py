from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.warp import Resampling, reproject
from scipy import ndimage, sparse
from scipy.optimize import lsq_linear

from ..common import ensure_dir, read_tif, write_json, write_tif
from ..config import ProjectConfig
from .seasonal import rotate_coefficients


def _aggregate_sum(arr: np.ndarray, stride: int, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = arr.shape
    ch = (h + stride - 1) // stride
    cw = (w + stride - 1) // stride
    out = np.zeros((ch, cw), dtype=float)
    count = np.zeros((ch, cw), dtype=int)
    for rr in range(ch):
        r0, r1 = rr*stride, min(h, (rr+1)*stride)
        for cc in range(cw):
            c0, c1 = cc*stride, min(w, (cc+1)*stride)
            m = valid[r0:r1, c0:c1]
            if m.any():
                out[rr, cc] = float(np.nansum(arr[r0:r1, c0:c1][m]))
                count[rr, cc] = int(m.sum())
    return out, count


def _coarse_support(mask: np.ndarray, stride: int, min_fraction: float = 0.25) -> np.ndarray:
    h, w = mask.shape
    ch = (h + stride - 1) // stride
    cw = (w + stride - 1) // stride
    out = np.zeros((ch, cw), dtype=bool)
    for rr in range(ch):
        r0, r1 = rr*stride, min(h, (rr+1)*stride)
        for cc in range(cw):
            c0, c1 = cc*stride, min(w, (cc+1)*stride)
            block = mask[r0:r1, c0:c1]
            out[rr, cc] = float(np.mean(block)) >= min_fraction
    return out


def _remove_unconstrained_components(support: np.ndarray, data_mask: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(support, structure=np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=int))
    keep = np.zeros_like(support, dtype=bool)
    for i in range(1, n+1):
        comp = labels == i
        if np.any(comp & data_mask):
            keep |= comp
    return keep


def _graph_matrix(support: np.ndarray):
    ids = -np.ones(support.shape, dtype=int)
    ids[support] = np.arange(int(support.sum()))
    rows, cols, vals = [], [], []
    e = 0
    h, w = support.shape
    for r in range(h):
        for c in range(w):
            if not support[r, c]:
                continue
            i = ids[r, c]
            for dr, dc in ((1,0), (0,1)):
                rr, cc = r+dr, c+dc
                if rr < h and cc < w and support[rr, cc]:
                    j = ids[rr, cc]
                    rows.extend([e,e]); cols.extend([i,j]); vals.extend([1.0,-1.0]); e += 1
    L = sparse.csr_matrix((vals, (rows, cols)), shape=(e, int(support.sum())))
    return ids, L


def _solve_regularized(raw: np.ndarray, q2: np.ndarray, support: np.ndarray, data_mask: np.ndarray, lam: float, bounds: tuple[float,float]):
    ids, L = _graph_matrix(support)
    n = int(support.sum())
    node_data = data_mask & support
    idx = ids[node_data]
    q = q2[node_data].astype(float)
    qnorm = q / max(float(np.nanmedian(q[q > 0])), 1e-12)
    Adata = sparse.csr_matrix((np.sqrt(qnorm), (np.arange(len(idx)), idx)), shape=(len(idx), n))
    bdata = np.sqrt(qnorm) * raw[node_data]
    if L.shape[0] > 0 and lam > 0:
        A = sparse.vstack([Adata, np.sqrt(float(lam)) * L], format="csr")
        b = np.concatenate([bdata, np.zeros(L.shape[0], dtype=float)])
    else:
        A, b = Adata, bdata
    res = lsq_linear(A, b, bounds=bounds, method="trf", lsq_solver="lsmr", tol=1e-8, max_iter=300)
    field = np.full(support.shape, np.nan, dtype=float)
    field[support] = res.x
    return field


def _spatial_cv_lambda(raw, q2, support, data_mask, candidates, bounds, block_cells=3):
    rr, cc = np.indices(support.shape)
    fold = ((rr // int(block_cells)) + 2 * (cc // int(block_cells))) % 5
    rows = []
    for lam in candidates:
        se_num = 0.0
        wt_sum = 0.0
        for k in range(5):
            hold = data_mask & support & (fold == k)
            train = data_mask & support & ~hold
            if hold.sum() == 0 or train.sum() < 3:
                continue
            pred = _solve_regularized(raw, q2, support, train, float(lam), bounds)
            ok = hold & np.isfinite(pred)
            if ok.any():
                w = q2[ok]
                se_num += float(np.sum(w * (pred[ok] - raw[ok])**2))
                wt_sum += float(np.sum(w))
        score = np.sqrt(se_num / wt_sum) if wt_sum > 0 else np.inf
        rows.append({"lambda": float(lam), "weighted_cv_rmse": float(score)})
    table = pd.DataFrame(rows).sort_values("weighted_cv_rmse")
    if table.empty or not np.isfinite(table.iloc[0]["weighted_cv_rmse"]):
        return float(candidates[0]), table
    return float(table.iloc[0]["lambda"]), table


def estimate_regularized_ske(cfg: ProjectConfig) -> dict[str, Any]:
    sec = cfg.section("ske")
    seasonal = cfg.outputs / "seasonal"
    lag_summary = __import__("json").loads((seasonal / "lag_summary.json").read_text(encoding="utf-8"))
    lag = float(lag_summary["lag_days"])
    period = float(cfg.section("seasonal_response").get("annual_period_days", 365.2425))
    ds = read_tif(seasonal / "deformation_annual_sin_mm.tif") / 1000.0
    dc = read_tif(seasonal / "deformation_annual_cos_mm.tif") / 1000.0
    hs = read_tif(seasonal / "head_annual_sin_m.tif")
    hc = read_tif(seasonal / "head_annual_cos_m.tif")
    rhs, rhc = rotate_coefficients(hs, hc, lag, period)
    q2_full = rhs*rhs + rhc*rhc
    qd_full = ds*rhs + dc*rhc
    hamp = np.sqrt(q2_full)
    damp = np.hypot(ds, dc) * 1000.0
    sresp = cfg.section("seasonal_response")
    data_valid = np.isfinite(q2_full) & np.isfinite(qd_full)
    data_valid &= hamp >= float(sresp.get("min_head_amplitude_m", 0.2))
    data_valid &= damp >= float(sresp.get("min_deformation_amplitude_mm", 0.5))
    gw_support = read_tif(cfg.outputs / "groundwater" / "groundwater_support_mask.tif") > 0
    data_valid &= gw_support

    raw_full = np.full_like(q2_full, np.nan, dtype=float)
    raw_full[data_valid] = qd_full[data_valid] / np.maximum(q2_full[data_valid], 1e-12)

    stride = int(sec.get("support_stride_pixels", 16))
    q2_coarse, ndata = _aggregate_sum(q2_full, stride, data_valid)
    qd_coarse, _ = _aggregate_sum(qd_full, stride, data_valid)
    raw = np.full_like(q2_coarse, np.nan, dtype=float)
    dm = (ndata >= int(sec.get("min_data_pixels_per_support_cell", 4))) & (q2_coarse > 0)
    raw[dm] = qd_coarse[dm] / q2_coarse[dm]
    support = _coarse_support(gw_support, stride, min_fraction=float(sec.get("support_min_fraction", 0.25)))
    support = _remove_unconstrained_components(support, dm)
    dm &= support
    if dm.sum() < 3:
        raise ValueError("Too few support cells for regularized Ske inversion")

    bounds = (float(sec.get("min", 0.0)), float(sec.get("max", 0.05)))
    candidates = [float(v) for v in sec.get("lambda_candidates", [0.1, 1.0, 10.0, 100.0])]
    if sec.get("lambda") is not None:
        lam = float(sec["lambda"])
        cv_table = pd.DataFrame([{"lambda": lam, "weighted_cv_rmse": np.nan}])
    else:
        lam, cv_table = _spatial_cv_lambda(raw, q2_coarse, support, dm, candidates, bounds, block_cells=int(sec.get("cv_block_cells", 3)))
    ske_coarse = _solve_regularized(raw, q2_coarse, support, dm, lam, bounds)

    with rasterio.open(seasonal / "head_annual_sin_m.tif") as ref:
        profile = ref.profile
        crs, transform = str(ref.crs), ref.transform
        height, width = ref.height, ref.width
    coarse_transform = transform * Affine.scale(stride, stride)
    ske_full = np.full((height, width), np.nan, dtype="float32")
    reproject(
        source=ske_coarse.astype("float32"), destination=ske_full,
        src_transform=coarse_transform, src_crs=crs,
        dst_transform=transform, dst_crs=crs,
        src_nodata=np.nan, dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    ske_full[~gw_support] = np.nan
    support_full = np.isfinite(ske_full) & gw_support
    out_dir = ensure_dir(cfg.outputs / "seasonal")
    write_tif(out_dir / "ske_raw_ratio.tif", raw_full.astype("float32"), crs, transform)
    write_tif(out_dir / "ske_effective.tif", ske_full, crs, transform)
    write_tif(out_dir / "ske_support_mask.tif", support_full.astype("uint8"), crs, transform, nodata=0, dtype="uint8")
    cv_table.to_csv(out_dir / "ske_lambda_cv.csv", index=False)
    summary = {
        "status": "ok",
        "lag_days": lag,
        "regularization": "bounded_coarse_grid_laplacian",
        "support_stride_pixels": stride,
        "lambda": lam,
        "data_support_cells": int(dm.sum()),
        "solution_support_cells": int(support.sum()),
        "ske_median": float(np.nanmedian(ske_full)),
        "ske_p10": float(np.nanpercentile(ske_full, 10)),
        "ske_p90": float(np.nanpercentile(ske_full, 90)),
        "bounds": list(bounds),
    }
    write_json(out_dir / "ske_summary.json", summary)
    return summary
