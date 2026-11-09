#!/usr/bin/env python3
"""
Fast exact validation for Hengshui InSAR temporal decomposition.

Optimizations relative to validate_insar_decomposition.py
---------------------------------------------------------
* Reads each InSAR block only once.
* Uses existing M1/M2 RMSE rasters for the two full-period nested models.
* Reuses one finite-pixel mask per block.
* Uses precomputed least-squares operators for complete time series.
* Computes M0/recent RSS from normal-equation sufficient statistics instead
  of materializing full residual arrays.
* Uses direct train->test prediction operators for temporal holdouts.
* Defaults to 512x512 blocks for better BLAS efficiency.

For the current Hengshui stack, valid pixels have 245/245 observations, so
the complete-series path handles essentially the whole scientific domain.
A slower fallback is retained for partially observed pixels.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from scipy.stats import f as f_dist
from scipy.stats import pearsonr, spearmanr

from hydrogeo_insar.common import block_slices, days_to_dates, h5_grid_metadata
from hydrogeo_insar.temporal.fit import fit_block
from hydrogeo_insar.temporal.model import TimeModel, design_matrix


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stack", default="outputs_hengshui/canonical/insar_stack.h5")
    p.add_argument("--deformation-dir", default="outputs_hengshui/deformation")
    p.add_argument("--outdir", default="outputs_hengshui/validation/decomposition")
    p.add_argument("--recent-start", default="2022-01-01")
    p.add_argument("--period-days", type=float, default=365.2425)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--state-threshold-mm-yr", type=float, default=5.0)
    p.add_argument("--aggregate-km", type=float, default=5.0)
    p.add_argument("--projected-crs", default="EPSG:32650")
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--min-observations", type=int, default=24)
    p.add_argument("--holdout-years", type=int, nargs="*", default=[2023, 2024, 2025])
    p.add_argument("--skip-holdout", action="store_true")
    return p.parse_args()


def make_profile(grid, dtype="float32", nodata=np.nan):
    return {
        "driver": "GTiff",
        "height": grid["height"],
        "width": grid["width"],
        "count": 1,
        "dtype": dtype,
        "crs": grid["crs"],
        "transform": grid["transform"],
        "compress": "deflate",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
        "nodata": nodata,
    }


def open_writer(path, grid, dtype="float32", nodata=np.nan):
    return rasterio.open(path, "w", **make_profile(grid, dtype, nodata))


def fit_operator(X):
    """Return matrices needed for exact complete-series LS statistics."""
    X = np.asarray(X, dtype=np.float64)
    xtx = X.T @ X
    gram = xtx + 1e-12 * np.eye(X.shape[1], dtype=np.float64)
    inv_gram = np.linalg.inv(gram)
    projector = inv_gram @ X.T
    return {
        "X": X,
        "xtx": xtx,
        "projector": projector,
    }


def complete_fit_stats(Y, op):
    """Exact LS beta/RSS for complete columns without full residual arrays."""
    Y = np.asarray(Y, dtype=np.float64)
    X = op["X"]
    xtx = op["xtx"]
    P = op["projector"]

    beta = P @ Y
    xty = X.T @ Y
    y2 = np.einsum("tp,tp->p", Y, Y, optimize=True)
    xb = xtx @ beta
    rss = (
        y2
        - 2.0 * np.einsum("kp,kp->p", beta, xty, optimize=True)
        + np.einsum("kp,kp->p", beta, xb, optimize=True)
    )
    rss = np.maximum(rss, 0.0)
    rmse = np.sqrt(rss / float(Y.shape[0]))
    return beta, rmse, rss


def direct_holdout_operator(dates, train, test, degree, period_days):
    origin = dates[0]
    model = TimeModel(
        polynomial_degree=degree,
        periods_days=(period_days,),
    )
    Xtr, _ = design_matrix(dates[train], model, origin=origin)
    Xte, _ = design_matrix(dates[test], model, origin=origin)
    op = fit_operator(Xtr)
    return Xte @ op["projector"]


def f_test_nested(rss_reduced, rss_full, n, k_reduced, k_full):
    rss_reduced = np.asarray(rss_reduced, dtype=float)
    rss_full = np.asarray(rss_full, dtype=float)
    n = np.asarray(n, dtype=float)

    df1 = float(k_full - k_reduced)
    df2 = n - float(k_full)

    F = np.full(rss_full.shape, np.nan, dtype=float)
    p = np.full(rss_full.shape, np.nan, dtype=float)
    ok = (
        np.isfinite(rss_reduced)
        & np.isfinite(rss_full)
        & np.isfinite(n)
        & (rss_full > 0)
        & (df2 > 0)
    )
    if ok.any():
        num = np.maximum((rss_reduced[ok] - rss_full[ok]) / df1, 0.0)
        den = rss_full[ok] / df2[ok]
        ff = num / den
        F[ok] = ff
        p[ok] = f_dist.sf(ff, df1, df2[ok])
    return F, p


def classify_rate(rate, threshold):
    x = np.asarray(rate, dtype=float)
    out = np.zeros(x.shape, dtype=np.uint8)
    ok = np.isfinite(x)
    out[ok & (x < -threshold)] = 1
    out[ok & (np.abs(x) <= threshold)] = 2
    out[ok & (x > threshold)] = 3
    return out


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 3:
        return {"n": int(len(x)), "pearson_r": np.nan, "spearman_r": np.nan}
    return {
        "n": int(len(x)),
        "pearson_r": float(pearsonr(x, y).statistic),
        "spearman_r": float(spearmanr(x, y).statistic),
    }


def raster_values(path):
    with rasterio.open(path) as src:
        a = src.read(1).astype(float)
        nd = src.nodata
        if nd is not None and np.isfinite(nd):
            a[a == nd] = np.nan
        return a


def aggregate_raster_mean(path, dst_crs, resolution_m):
    with rasterio.open(path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=resolution_m,
        )
        out = np.full((height, width), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
    return out


def summarize_recent_vs_end(end_path, recent_path, diff_path, threshold, projected_crs, aggregate_km):
    end = raster_values(end_path)
    recent = raster_values(recent_path)
    diff = raster_values(diff_path)

    ok = np.isfinite(end) & np.isfinite(recent)
    d = diff[ok]
    summary = safe_corr(end[ok], recent[ok])
    summary.update({
        "mean_difference_mm_yr_end_minus_recent": float(np.mean(d)),
        "median_difference_mm_yr_end_minus_recent": float(np.median(d)),
        "mae_mm_yr": float(np.mean(np.abs(d))),
        "rmse_mm_yr": float(np.sqrt(np.mean(d**2))),
        "difference_p10_mm_yr": float(np.percentile(d, 10)),
        "difference_p90_mm_yr": float(np.percentile(d, 90)),
    })

    state_end = classify_rate(end, threshold)
    state_recent = classify_rate(recent, threshold)
    valid = (state_end > 0) & (state_recent > 0)
    summary["state_agreement_fraction"] = float(
        np.mean(state_end[valid] == state_recent[valid])
    )

    er = valid & (state_end == 3)
    es = valid & (state_end == 1)
    summary["end_rebound_pixels"] = int(er.sum())
    summary["end_rebound_confirmed_by_recent_fraction"] = (
        float(np.mean(state_recent[er] == 3)) if er.any() else np.nan
    )
    summary["end_subsidence_pixels"] = int(es.sum())
    summary["end_subsidence_confirmed_by_recent_fraction"] = (
        float(np.mean(state_recent[es] == 1)) if es.any() else np.nan
    )

    resolution_m = float(aggregate_km) * 1000.0
    ea = aggregate_raster_mean(end_path, projected_crs, resolution_m)
    ra = aggregate_raster_mean(recent_path, projected_crs, resolution_m)
    oka = np.isfinite(ea) & np.isfinite(ra)
    agg = safe_corr(ea[oka], ra[oka])
    dd = ea[oka] - ra[oka]
    agg.update({
        "aggregate_km": float(aggregate_km),
        "mean_difference_mm_yr_end_minus_recent": float(np.mean(dd)),
        "median_difference_mm_yr_end_minus_recent": float(np.median(dd)),
        "mae_mm_yr": float(np.mean(np.abs(dd))),
        "rmse_mm_yr": float(np.sqrt(np.mean(dd**2))),
    })
    summary["aggregated"] = agg
    return summary


def main():
    args = parse_args()

    stack_path = Path(args.stack)
    deformation_dir = Path(args.deformation_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    grid = h5_grid_metadata(stack_path)

    with h5py.File(stack_path, "r") as h5:
        dates = np.asarray(days_to_dates(h5["date_days"][:]), dtype="datetime64[D]")

    period = float(args.period_days)
    origin = dates[0]

    M0 = TimeModel(polynomial_degree=1, periods_days=())
    M1 = TimeModel(polynomial_degree=1, periods_days=(period,))
    M2 = TimeModel(polynomial_degree=2, periods_days=(period,))

    X0, _ = design_matrix(dates, M0, origin=origin)
    X1, _ = design_matrix(dates, M1, origin=origin)
    X2, _ = design_matrix(dates, M2, origin=origin)
    k0, k1, k2 = X0.shape[1], X1.shape[1], X2.shape[1]

    op0 = fit_operator(X0)

    recent_start = np.datetime64(args.recent_start, "D")
    recent_mask = dates >= recent_start
    recent_dates = dates[recent_mask]
    X_recent, _ = design_matrix(
        recent_dates,
        M1,
        origin=recent_dates[0],
    )
    op_recent = fit_operator(X_recent)

    holdout_years = [] if args.skip_holdout else list(args.holdout_years)
    holdout_ops = {}
    for year in holdout_years:
        cutoff = np.datetime64(f"{year}-01-01", "D")
        next_cut = np.datetime64(f"{year+1}-01-01", "D")
        train = dates < cutoff
        test = (dates >= cutoff) & (dates < next_cut)
        if train.sum() < args.min_observations or test.sum() == 0:
            continue
        H1 = direct_holdout_operator(dates, train, test, 1, period)
        H2 = direct_holdout_operator(dates, train, test, 2, period)
        holdout_ops[year] = {
            "train": train,
            "test": test,
            "H": np.vstack([H1, H2]),
            "ntest": int(test.sum()),
        }

    print(
        f"[FAST-VALIDATE] epochs={len(dates)}, "
        f"recent={str(recent_dates[0])}..{str(recent_dates[-1])} "
        f"({len(recent_dates)} epochs), "
        f"grid={grid['height']}x{grid['width']}, block={args.block_size}",
        flush=True,
    )

    outputs = {
        "annual_f_stat": outdir / "annual_f_stat.tif",
        "annual_f_pvalue": outdir / "annual_f_pvalue.tif",
        "annual_significant": outdir / "annual_significant_95.tif",
        "quadratic_f_stat": outdir / "quadratic_f_stat.tif",
        "quadratic_f_pvalue": outdir / "quadratic_f_pvalue.tif",
        "quadratic_significant": outdir / "quadratic_significant_95.tif",
        "recent_velocity": outdir / "recent_velocity_2022_2025_mm_yr.tif",
        "recent_rmse": outdir / "recent_fit_rmse_2022_2025_mm.tif",
        "end_minus_recent": outdir / "endrate_minus_recent_mm_yr.tif",
        "state_agreement": outdir / "endrate_recent_state_agreement.tif",
    }

    writers = {
        "annual_f_stat": open_writer(outputs["annual_f_stat"], grid),
        "annual_f_pvalue": open_writer(outputs["annual_f_pvalue"], grid),
        "annual_significant": open_writer(outputs["annual_significant"], grid, "uint8", 0),
        "quadratic_f_stat": open_writer(outputs["quadratic_f_stat"], grid),
        "quadratic_f_pvalue": open_writer(outputs["quadratic_f_pvalue"], grid),
        "quadratic_significant": open_writer(outputs["quadratic_significant"], grid, "uint8", 0),
        "recent_velocity": open_writer(outputs["recent_velocity"], grid),
        "recent_rmse": open_writer(outputs["recent_rmse"], grid),
        "end_minus_recent": open_writer(outputs["end_minus_recent"], grid),
        "state_agreement": open_writer(outputs["state_agreement"], grid, "uint8", 0),
    }

    holdout_writers = {}
    for year in holdout_ops:
        path = outdir / f"holdout_{year}_delta_rmse_linear_minus_quadratic_mm.tif"
        outputs[f"holdout_{year}_delta_rmse"] = path
        holdout_writers[year] = open_writer(path, grid)

    src_m1_rmse = rasterio.open(deformation_dir / "linear_annual_rmse_mm.tif")
    src_m2_rmse = rasterio.open(deformation_dir / "quadratic_annual_rmse_mm.tif")
    src_nobs = rasterio.open(deformation_dir / "n_observations.tif")
    src_end = rasterio.open(deformation_dir / "end_rate_mm_yr.tif")
    src_bic = rasterio.open(deformation_dir / "delta_bic_linear_minus_quadratic.tif")

    annual_valid = annual_sig = 0
    quad_valid = quad_sig = 0
    bic_pos = bic_pos_f = 0
    partial_pixels_total = 0

    holdout_acc = {
        y: {
            "n": 0,
            "quad_better": 0,
            "linear_sse": 0.0,
            "quadratic_sse": 0.0,
        }
        for y in holdout_ops
    }

    blocks = list(block_slices(grid["height"], grid["width"], args.block_size))
    report_every = max(1, len(blocks) // 50)
    t0 = time.perf_counter()

    try:
        with h5py.File(stack_path, "r") as h5:
            ds = h5["displacement_mm"]

            for ib, (r0, r1, c0, c1) in enumerate(blocks, start=1):
                win = rasterio.windows.Window(c0, r0, c1-c0, r1-r0)

                arr = ds[:, r0:r1, c0:c1]
                T, bh, bw = arr.shape
                y = arr.reshape(T, -1)

                nobs = src_nobs.read(1, window=win).reshape(-1).astype(np.int32)
                complete = nobs == len(dates)
                partial = (nobs >= args.min_observations) & ~complete
                partial_pixels_total += int(partial.sum())

                # Allocate full block products.
                npix = y.shape[1]
                rss0 = np.full(npix, np.nan, dtype=float)
                recent_rate = np.full(npix, np.nan, dtype=float)
                recent_rmse = np.full(npix, np.nan, dtype=float)

                if complete.any():
                    yc = np.asarray(y[:, complete], dtype=np.float64)

                    # M0 full-period RSS.
                    _, _, rss0_c = complete_fit_stats(yc, op0)
                    rss0[complete] = rss0_c

                    # Recent M1 slope + RMSE.
                    br, rr, _ = complete_fit_stats(
                        yc[recent_mask],
                        op_recent,
                    )
                    recent_rate[complete] = br[1]
                    recent_rmse[complete] = rr

                # Slow fallback only if partially observed pixels actually exist.
                if partial.any():
                    yp = np.asarray(y[:, partial], dtype=np.float64)
                    _, _, _, rss0_p = fit_block(
                        yp,
                        X0,
                        min_obs=args.min_observations,
                    )
                    rss0[partial] = rss0_p

                    brp, rrp, _, _ = fit_block(
                        yp[recent_mask],
                        X_recent,
                        min_obs=args.min_observations,
                    )
                    recent_rate[partial] = brp[:, 1]
                    recent_rmse[partial] = rrp

                m1_rmse = src_m1_rmse.read(1, window=win).reshape(-1).astype(float)
                m2_rmse = src_m2_rmse.read(1, window=win).reshape(-1).astype(float)
                n_float = nobs.astype(float)

                rss1 = m1_rmse**2 * n_float
                rss2 = m2_rmse**2 * n_float

                Fann, Pann = f_test_nested(rss0, rss1, n_float, k0, k1)
                Fquad, Pquad = f_test_nested(rss1, rss2, n_float, k1, k2)
                sigann = np.isfinite(Pann) & (Pann < args.alpha)
                sigquad = np.isfinite(Pquad) & (Pquad < args.alpha)

                bic_delta = src_bic.read(1, window=win).reshape(-1).astype(float)
                bok = np.isfinite(bic_delta)

                annual_valid += int(np.isfinite(Pann).sum())
                annual_sig += int(sigann.sum())
                quad_valid += int(np.isfinite(Pquad).sum())
                quad_sig += int(sigquad.sum())
                bic_pos += int(np.sum(bok & (bic_delta > 0)))
                bic_pos_f += int(np.sum(bok & (bic_delta > 0) & sigquad))

                end_rate = src_end.read(1, window=win).reshape(-1).astype(float)
                diff_rate = end_rate - recent_rate
                se = classify_rate(end_rate, args.state_threshold_mm_yr)
                sr = classify_rate(recent_rate, args.state_threshold_mm_yr)
                valid_state = (se > 0) & (sr > 0)
                agree = np.zeros(npix, dtype=np.uint8)
                agree[valid_state & (se == sr)] = 1
                agree[valid_state & (se != sr)] = 2

                # Holdout predictions: one BLAS call per year for both models.
                for year, info in holdout_ops.items():
                    delta = np.full(npix, np.nan, dtype=float)

                    if complete.any():
                        yc = np.asarray(y[:, complete], dtype=np.float64)
                        pred_both = info["H"] @ yc[info["train"]]
                        nt = info["ntest"]
                        actual = yc[info["test"]]
                        r1 = actual - pred_both[:nt]
                        r2 = actual - pred_both[nt:]
                        rm1 = np.sqrt(np.mean(r1*r1, axis=0))
                        rm2 = np.sqrt(np.mean(r2*r2, axis=0))
                        d = rm1 - rm2
                        delta[complete] = d

                        holdout_acc[year]["n"] += int(len(d))
                        holdout_acc[year]["quad_better"] += int(np.sum(d > 0))
                        holdout_acc[year]["linear_sse"] += float(np.sum(rm1**2))
                        holdout_acc[year]["quadratic_sse"] += float(np.sum(rm2**2))

                    # Current dataset has no partial valid pixels, so no slow
                    # holdout fallback is needed for scientific-domain pixels.
                    holdout_writers[year].write(
                        delta.reshape(bh, bw).astype("float32"),
                        1,
                        window=win,
                    )

                writers["annual_f_stat"].write(Fann.reshape(bh,bw).astype("float32"), 1, window=win)
                writers["annual_f_pvalue"].write(Pann.reshape(bh,bw).astype("float32"), 1, window=win)
                writers["annual_significant"].write(sigann.reshape(bh,bw).astype("uint8"), 1, window=win)
                writers["quadratic_f_stat"].write(Fquad.reshape(bh,bw).astype("float32"), 1, window=win)
                writers["quadratic_f_pvalue"].write(Pquad.reshape(bh,bw).astype("float32"), 1, window=win)
                writers["quadratic_significant"].write(sigquad.reshape(bh,bw).astype("uint8"), 1, window=win)
                writers["recent_velocity"].write(recent_rate.reshape(bh,bw).astype("float32"), 1, window=win)
                writers["recent_rmse"].write(recent_rmse.reshape(bh,bw).astype("float32"), 1, window=win)
                writers["end_minus_recent"].write(diff_rate.reshape(bh,bw).astype("float32"), 1, window=win)
                writers["state_agreement"].write(agree.reshape(bh,bw).astype("uint8"), 1, window=win)

                if ib == 1 or ib % report_every == 0 or ib == len(blocks):
                    elapsed = time.perf_counter() - t0
                    frac = ib / len(blocks)
                    eta = elapsed * (1.0/frac - 1.0)
                    print(
                        f"[FAST-VALIDATE] {ib}/{len(blocks)} "
                        f"({100*frac:5.1f}%) "
                        f"elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m "
                        f"partial_pixels={partial_pixels_total}",
                        flush=True,
                    )
    finally:
        for dst in writers.values():
            dst.close()
        for dst in holdout_writers.values():
            dst.close()
        for src in [src_m1_rmse, src_m2_rmse, src_nobs, src_end, src_bic]:
            src.close()

    recent_summary = summarize_recent_vs_end(
        deformation_dir / "end_rate_mm_yr.tif",
        outputs["recent_velocity"],
        outputs["end_minus_recent"],
        args.state_threshold_mm_yr,
        args.projected_crs,
        args.aggregate_km,
    )

    bic = raster_values(deformation_dir / "delta_bic_linear_minus_quadratic.tif")
    bx = bic[np.isfinite(bic)]
    bic_summary = {
        "n": int(len(bx)),
        "p01": float(np.percentile(bx, 1)),
        "p10": float(np.percentile(bx, 10)),
        "p50": float(np.percentile(bx, 50)),
        "p90": float(np.percentile(bx, 90)),
        "p99": float(np.percentile(bx, 99)),
        "fraction_delta_gt_0": float(np.mean(bx > 0)),
        "fraction_delta_ge_2": float(np.mean(bx >= 2)),
        "fraction_delta_ge_6": float(np.mean(bx >= 6)),
        "fraction_delta_ge_10": float(np.mean(bx >= 10)),
    }

    holdout_summary = {}
    for year, a in holdout_acc.items():
        n = max(a["n"], 1)
        holdout_summary[str(year)] = {
            "valid_pixels": int(a["n"]),
            "rms_of_pixel_rmse_linear_mm": float(math.sqrt(a["linear_sse"] / n)),
            "rms_of_pixel_rmse_quadratic_mm": float(math.sqrt(a["quadratic_sse"] / n)),
            "quadratic_lower_rmse_fraction": float(a["quad_better"] / n),
        }

    summary = {
        "status": "ok",
        "implementation": "fast_complete_series_exact_ls",
        "block_size": int(args.block_size),
        "partial_valid_pixels_encountered": int(partial_pixels_total),
        "model_definitions": {
            "M0": "linear",
            "M1": "linear + annual harmonic",
            "M2": "quadratic + annual harmonic",
        },
        "f_test": {
            "assumption_note": (
                "Classical nested-model F test; temporal residual autocorrelation "
                "is not explicitly corrected, so p-values are model diagnostics."
            ),
            "alpha": float(args.alpha),
            "annual_component": {
                "comparison": "M0_vs_M1",
                "valid_pixels": int(annual_valid),
                "significant_pixels": int(annual_sig),
                "significant_fraction": float(annual_sig / annual_valid) if annual_valid else np.nan,
            },
            "quadratic_component": {
                "comparison": "M1_vs_M2",
                "valid_pixels": int(quad_valid),
                "significant_pixels": int(quad_sig),
                "significant_fraction": float(quad_sig / quad_valid) if quad_valid else np.nan,
                "bic_positive_and_f_significant_fraction_of_bic_positive": (
                    float(bic_pos_f / bic_pos) if bic_pos else np.nan
                ),
            },
        },
        "bic": bic_summary,
        "recent_rate_validation": {
            "recent_start": str(recent_dates[0]),
            "recent_end": str(recent_dates[-1]),
            "recent_epochs": int(len(recent_dates)),
            "state_threshold_mm_yr": float(args.state_threshold_mm_yr),
            **recent_summary,
        },
        "temporal_holdout": holdout_summary,
        "outputs": {k: str(v) for k, v in outputs.items()},
    }

    summary_path = outdir / "decomposition_validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if holdout_summary:
        pd.DataFrame.from_dict(
            holdout_summary,
            orient="index",
        ).rename_axis("year").reset_index().to_csv(
            outdir / "temporal_holdout_summary.csv",
            index=False,
        )

    print("\n================ FAST VALIDATION SUMMARY ================")
    print(
        "Annual F-test significant fraction = "
        f"{summary['f_test']['annual_component']['significant_fraction']:.4f}"
    )
    print(
        "Quadratic F-test significant fraction = "
        f"{summary['f_test']['quadratic_component']['significant_fraction']:.4f}"
    )
    print(
        "BIC-positive pixels also F-significant = "
        f"{summary['f_test']['quadratic_component']['bic_positive_and_f_significant_fraction_of_bic_positive']:.4f}"
    )
    print(
        "Recent vs terminal rate Pearson/Spearman = "
        f"{recent_summary['pearson_r']:.4f} / {recent_summary['spearman_r']:.4f}"
    )
    print(
        f"{args.aggregate_km:g}-km aggregated Pearson/Spearman = "
        f"{recent_summary['aggregated']['pearson_r']:.4f} / "
        f"{recent_summary['aggregated']['spearman_r']:.4f}"
    )
    print("State agreement =", f"{recent_summary['state_agreement_fraction']:.4f}")
    print(
        "End-rate rebound confirmed by recent rate = "
        f"{recent_summary['end_rebound_confirmed_by_recent_fraction']:.4f}"
    )
    print(
        "End-rate subsidence confirmed by recent rate = "
        f"{recent_summary['end_subsidence_confirmed_by_recent_fraction']:.4f}"
    )
    for year, row in holdout_summary.items():
        print(
            f"Holdout {year}: "
            f"linear={row['rms_of_pixel_rmse_linear_mm']:.3f} mm, "
            f"quadratic={row['rms_of_pixel_rmse_quadratic_mm']:.3f} mm, "
            f"quad-better={row['quadratic_lower_rmse_fraction']:.4f}"
        )
    print("summary:", summary_path)
    print("=========================================================")


if __name__ == "__main__":
    main()
