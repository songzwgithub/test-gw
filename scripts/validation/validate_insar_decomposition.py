#!/usr/bin/env python3
"""
Validation of Hengshui InSAR temporal decomposition.

Main diagnostics
----------------
1) Nested F tests:
   M0 = linear
   M1 = linear + annual harmonic
   M2 = quadratic + annual harmonic

   F_annual    : M0 vs M1
   F_quadratic : M1 vs M2

2) Independent recent-rate validation:
   Fit 2022-01-01 -> end with M1 (linear + annual) and compare
   its slope with the terminal derivative from the full-period M2 model.

3) Temporal holdout validation:
   Train through year-1, predict the following year for 2023, 2024, 2025.
   Compare M1 and M2 holdout RMSE.

The script is blockwise and designed for the current Hengshui grid.
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
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from scipy.stats import f as f_dist
from scipy.stats import pearsonr, spearmanr

from hydrogeo_insar.common import (
    block_slices,
    days_to_dates,
    h5_grid_metadata,
)
from hydrogeo_insar.temporal.fit import fit_block
from hydrogeo_insar.temporal.model import TimeModel, design_matrix


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate Hengshui InSAR temporal decomposition."
    )
    p.add_argument(
        "--stack",
        default="outputs_hengshui/canonical/insar_stack.h5",
    )
    p.add_argument(
        "--deformation-dir",
        default="outputs_hengshui/deformation",
    )
    p.add_argument(
        "--outdir",
        default="outputs_hengshui/validation/decomposition",
    )
    p.add_argument(
        "--recent-start",
        default="2022-01-01",
        help="Start date for independent recent linear+annual velocity.",
    )
    p.add_argument(
        "--period-days",
        type=float,
        default=365.2425,
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
    )
    p.add_argument(
        "--state-threshold-mm-yr",
        type=float,
        default=5.0,
    )
    p.add_argument(
        "--aggregate-km",
        type=float,
        default=5.0,
    )
    p.add_argument(
        "--projected-crs",
        default="EPSG:32650",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=256,
    )
    p.add_argument(
        "--min-observations",
        type=int,
        default=24,
    )
    p.add_argument(
        "--holdout-years",
        type=int,
        nargs="*",
        default=[2023, 2024, 2025],
    )
    p.add_argument(
        "--skip-holdout",
        action="store_true",
    )
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
    return rasterio.open(
        path,
        "w",
        **make_profile(grid, dtype=dtype, nodata=nodata),
    )


def f_test_nested(rss_reduced, rss_full, n, k_reduced, k_full):
    """Classical nested-model F test, vectorized."""
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
    if not ok.any():
        return F, p

    numerator = (rss_reduced[ok] - rss_full[ok]) / df1
    numerator = np.maximum(numerator, 0.0)
    denominator = rss_full[ok] / df2[ok]

    F_ok = numerator / denominator
    F[ok] = F_ok
    p[ok] = f_dist.sf(F_ok, df1, df2[ok])
    return F, p


def classify_rate(rate, threshold):
    out = np.zeros(np.asarray(rate).shape, dtype=np.uint8)
    x = np.asarray(rate, dtype=float)
    ok = np.isfinite(x)
    out[ok & (x < -threshold)] = 1   # subsidence
    out[ok & (np.abs(x) <= threshold)] = 2  # approximately stable
    out[ok & (x > threshold)] = 3    # rebound/uplift
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
        nodata = src.nodata
        if nodata is not None and np.isfinite(nodata):
            a[a == nodata] = np.nan
        return a


def aggregate_raster_mean(path, dst_crs, resolution_m):
    """Reproject one raster to a regular projected grid using mean resampling."""
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


def summarize_recent_vs_end(
    end_path,
    recent_path,
    diff_path,
    threshold,
    projected_crs,
    aggregate_km,
):
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
    valid_state = (state_end > 0) & (state_recent > 0)

    summary["state_agreement_fraction"] = float(
        np.mean(state_end[valid_state] == state_recent[valid_state])
    )

    end_rebound = valid_state & (state_end == 3)
    end_subsidence = valid_state & (state_end == 1)

    summary["end_rebound_pixels"] = int(end_rebound.sum())
    summary["end_rebound_confirmed_by_recent_fraction"] = (
        float(np.mean(state_recent[end_rebound] == 3))
        if end_rebound.any()
        else np.nan
    )
    summary["end_subsidence_pixels"] = int(end_subsidence.sum())
    summary["end_subsidence_confirmed_by_recent_fraction"] = (
        float(np.mean(state_recent[end_subsidence] == 1))
        if end_subsidence.any()
        else np.nan
    )

    resolution_m = float(aggregate_km) * 1000.0
    end_agg = aggregate_raster_mean(
        end_path,
        projected_crs,
        resolution_m,
    )
    recent_agg = aggregate_raster_mean(
        recent_path,
        projected_crs,
        resolution_m,
    )
    agg_ok = np.isfinite(end_agg) & np.isfinite(recent_agg)

    agg = safe_corr(end_agg[agg_ok], recent_agg[agg_ok])
    dd = end_agg[agg_ok] - recent_agg[agg_ok]
    agg.update({
        "aggregate_km": float(aggregate_km),
        "mean_difference_mm_yr_end_minus_recent": float(np.mean(dd)),
        "median_difference_mm_yr_end_minus_recent": float(np.median(dd)),
        "mae_mm_yr": float(np.mean(np.abs(dd))),
        "rmse_mm_yr": float(np.sqrt(np.mean(dd**2))),
    })
    summary["aggregated"] = agg
    return summary


def fit_predict_holdout(y, dates, train_mask, test_mask, degree, period_days, min_obs):
    origin = dates[0]
    model = TimeModel(
        polynomial_degree=degree,
        periods_days=(period_days,),
    )
    X_train, _ = design_matrix(
        dates[train_mask],
        model,
        origin=origin,
    )
    X_test, _ = design_matrix(
        dates[test_mask],
        model,
        origin=origin,
    )

    beta, _, _, _ = fit_block(
        y[train_mask],
        X_train,
        min_obs=min_obs,
    )
    pred = X_test @ np.nan_to_num(beta.T, nan=0.0)

    actual = y[test_mask]
    valid = np.isfinite(actual) & np.all(np.isfinite(beta), axis=1)[None, :]

    residual = np.where(valid, actual - pred, np.nan)
    rss = np.nansum(residual**2, axis=0)
    n = valid.sum(axis=0)

    rmse = np.full(y.shape[1], np.nan, dtype=float)
    good = n > 0
    rmse[good] = np.sqrt(rss[good] / n[good])
    return rmse


def main():
    args = parse_args()

    stack_path = Path(args.stack)
    deformation_dir = Path(args.deformation_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    required = [
        deformation_dir / "end_rate_mm_yr.tif",
        deformation_dir / "linear_annual_rmse_mm.tif",
        deformation_dir / "quadratic_annual_rmse_mm.tif",
        deformation_dir / "n_observations.tif",
        deformation_dir / "delta_bic_linear_minus_quadratic.tif",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required decomposition outputs are missing:\n" + "\n".join(missing)
        )

    grid = h5_grid_metadata(stack_path)

    with h5py.File(stack_path, "r") as h5:
        dates = days_to_dates(h5["date_days"][:])

    dates = np.asarray(dates, dtype="datetime64[D]")
    period_days = float(args.period_days)

    # Full-period design matrices.
    origin = dates[0]
    M0 = TimeModel(polynomial_degree=1, periods_days=())
    M1 = TimeModel(polynomial_degree=1, periods_days=(period_days,))
    M2 = TimeModel(polynomial_degree=2, periods_days=(period_days,))
    X0, _ = design_matrix(dates, M0, origin=origin)
    X1, _ = design_matrix(dates, M1, origin=origin)
    X2, _ = design_matrix(dates, M2, origin=origin)

    k0, k1, k2 = X0.shape[1], X1.shape[1], X2.shape[1]

    recent_start = np.datetime64(args.recent_start, "D")
    recent_mask = dates >= recent_start
    recent_dates = dates[recent_mask]
    if recent_mask.sum() < max(args.min_observations, 24):
        raise ValueError("Too few recent epochs for recent-rate validation")

    X_recent, _ = design_matrix(
        recent_dates,
        M1,
        origin=recent_dates[0],
    )

    print(
        f"[VALIDATE] epochs={len(dates)}, "
        f"recent={str(recent_dates[0])}..{str(recent_dates[-1])} "
        f"({len(recent_dates)} epochs), "
        f"grid={grid['height']}x{grid['width']}",
        flush=True,
    )
    print(
        f"[VALIDATE] models: M0 k={k0}, M1 k={k1}, M2 k={k2}; "
        f"alpha={args.alpha}",
        flush=True,
    )

    output_paths = {
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
        "annual_f_stat": open_writer(output_paths["annual_f_stat"], grid),
        "annual_f_pvalue": open_writer(output_paths["annual_f_pvalue"], grid),
        "annual_significant": open_writer(
            output_paths["annual_significant"], grid, dtype="uint8", nodata=0
        ),
        "quadratic_f_stat": open_writer(output_paths["quadratic_f_stat"], grid),
        "quadratic_f_pvalue": open_writer(output_paths["quadratic_f_pvalue"], grid),
        "quadratic_significant": open_writer(
            output_paths["quadratic_significant"], grid, dtype="uint8", nodata=0
        ),
        "recent_velocity": open_writer(output_paths["recent_velocity"], grid),
        "recent_rmse": open_writer(output_paths["recent_rmse"], grid),
        "end_minus_recent": open_writer(output_paths["end_minus_recent"], grid),
        "state_agreement": open_writer(
            output_paths["state_agreement"], grid, dtype="uint8", nodata=0
        ),
    }

    holdout_years = [] if args.skip_holdout else list(args.holdout_years)
    holdout_writers = {}
    for year in holdout_years:
        path = outdir / f"holdout_{year}_delta_rmse_linear_minus_quadratic_mm.tif"
        output_paths[f"holdout_{year}_delta_rmse"] = path
        holdout_writers[year] = open_writer(path, grid)

    # Existing decomposition rasters.
    src_linear_rmse = rasterio.open(
        deformation_dir / "linear_annual_rmse_mm.tif"
    )
    src_quad_rmse = rasterio.open(
        deformation_dir / "quadratic_annual_rmse_mm.tif"
    )
    src_nobs = rasterio.open(
        deformation_dir / "n_observations.tif"
    )
    src_end = rasterio.open(
        deformation_dir / "end_rate_mm_yr.tif"
    )
    src_bic = rasterio.open(
        deformation_dir / "delta_bic_linear_minus_quadratic.tif"
    )

    # Summary counters.
    annual_valid = annual_sig = 0
    quad_valid = quad_sig = 0
    quad_bic_and_f = quad_bic_positive = 0
    recent_valid = 0
    state_agree = 0

    holdout_acc = {
        year: {
            "linear_sse": 0.0,
            "quadratic_sse": 0.0,
            "n_pixels": 0,
            "quad_better_pixels": 0,
            "valid_pixel_rmse": 0,
        }
        for year in holdout_years
    }

    blocks = list(
        block_slices(
            grid["height"],
            grid["width"],
            args.block_size,
        )
    )
    report_every = max(1, len(blocks) // 50)
    t0 = time.perf_counter()

    try:
        with h5py.File(stack_path, "r") as h5:
            ds = h5["displacement_mm"]

            for block_no, (r0, r1, c0, c1) in enumerate(blocks, start=1):
                win = rasterio.windows.Window(
                    c0,
                    r0,
                    c1 - c0,
                    r1 - r0,
                )
                arr = ds[:, r0:r1, c0:c1].astype(float)
                T, bh, bw = arr.shape
                y = arr.reshape(T, -1)

                # ---- annual F test: M0 vs M1 ----
                _, _, n0, rss0 = fit_block(
                    y,
                    X0,
                    min_obs=args.min_observations,
                )

                linear_rmse = src_linear_rmse.read(1, window=win).reshape(-1).astype(float)
                n_existing = src_nobs.read(1, window=win).reshape(-1).astype(float)
                rss1 = (linear_rmse**2) * n_existing

                F_ann, p_ann = f_test_nested(
                    rss0,
                    rss1,
                    n_existing,
                    k0,
                    k1,
                )
                sig_ann = np.isfinite(p_ann) & (p_ann < args.alpha)

                # ---- quadratic F test: M1 vs M2 ----
                quad_rmse = src_quad_rmse.read(1, window=win).reshape(-1).astype(float)
                rss2 = (quad_rmse**2) * n_existing

                F_quad, p_quad = f_test_nested(
                    rss1,
                    rss2,
                    n_existing,
                    k1,
                    k2,
                )
                sig_quad = np.isfinite(p_quad) & (p_quad < args.alpha)

                bic_delta = src_bic.read(1, window=win).reshape(-1).astype(float)
                bic_ok = np.isfinite(bic_delta)

                annual_valid += int(np.isfinite(p_ann).sum())
                annual_sig += int(sig_ann.sum())
                quad_valid += int(np.isfinite(p_quad).sum())
                quad_sig += int(sig_quad.sum())
                quad_bic_positive += int(np.sum(bic_ok & (bic_delta > 0)))
                quad_bic_and_f += int(np.sum(bic_ok & (bic_delta > 0) & sig_quad))

                # ---- independent recent rate: M1 on recent period ----
                beta_recent, rmse_recent, _, _ = fit_block(
                    y[recent_mask],
                    X_recent,
                    min_obs=max(args.min_observations, X_recent.shape[1] + 2),
                )
                recent_rate = beta_recent[:, 1]

                end_rate = src_end.read(1, window=win).reshape(-1).astype(float)
                diff_rate = end_rate - recent_rate

                state_end = classify_rate(
                    end_rate,
                    args.state_threshold_mm_yr,
                )
                state_recent = classify_rate(
                    recent_rate,
                    args.state_threshold_mm_yr,
                )
                state_valid = (state_end > 0) & (state_recent > 0)
                agreement = np.zeros_like(state_end, dtype=np.uint8)
                agreement[state_valid & (state_end == state_recent)] = 1
                agreement[state_valid & (state_end != state_recent)] = 2

                recent_valid += int(state_valid.sum())
                state_agree += int(np.sum(state_valid & (state_end == state_recent)))

                # ---- temporal holdout ----
                for year in holdout_years:
                    cutoff = np.datetime64(f"{year}-01-01", "D")
                    next_cut = np.datetime64(f"{year+1}-01-01", "D")
                    train = dates < cutoff
                    test = (dates >= cutoff) & (dates < next_cut)

                    if train.sum() < args.min_observations or test.sum() == 0:
                        continue

                    rmse_l = fit_predict_holdout(
                        y,
                        dates,
                        train,
                        test,
                        degree=1,
                        period_days=period_days,
                        min_obs=args.min_observations,
                    )
                    rmse_q = fit_predict_holdout(
                        y,
                        dates,
                        train,
                        test,
                        degree=2,
                        period_days=period_days,
                        min_obs=args.min_observations,
                    )

                    delta_hold = rmse_l - rmse_q
                    okh = np.isfinite(rmse_l) & np.isfinite(rmse_q)

                    # Accumulate pixel-level RMSE comparison.
                    holdout_acc[year]["n_pixels"] += int(okh.sum())
                    holdout_acc[year]["quad_better_pixels"] += int(
                        np.sum(okh & (delta_hold > 0))
                    )
                    holdout_acc[year]["valid_pixel_rmse"] += int(okh.sum())
                    holdout_acc[year]["linear_sse"] += float(
                        np.nansum(rmse_l[okh] ** 2)
                    )
                    holdout_acc[year]["quadratic_sse"] += float(
                        np.nansum(rmse_q[okh] ** 2)
                    )

                    holdout_writers[year].write(
                        delta_hold.reshape(bh, bw).astype("float32"),
                        1,
                        window=win,
                    )

                # ---- write rasters ----
                writers["annual_f_stat"].write(
                    F_ann.reshape(bh, bw).astype("float32"), 1, window=win
                )
                writers["annual_f_pvalue"].write(
                    p_ann.reshape(bh, bw).astype("float32"), 1, window=win
                )
                writers["annual_significant"].write(
                    sig_ann.reshape(bh, bw).astype("uint8"), 1, window=win
                )
                writers["quadratic_f_stat"].write(
                    F_quad.reshape(bh, bw).astype("float32"), 1, window=win
                )
                writers["quadratic_f_pvalue"].write(
                    p_quad.reshape(bh, bw).astype("float32"), 1, window=win
                )
                writers["quadratic_significant"].write(
                    sig_quad.reshape(bh, bw).astype("uint8"), 1, window=win
                )
                writers["recent_velocity"].write(
                    recent_rate.reshape(bh, bw).astype("float32"), 1, window=win
                )
                writers["recent_rmse"].write(
                    rmse_recent.reshape(bh, bw).astype("float32"), 1, window=win
                )
                writers["end_minus_recent"].write(
                    diff_rate.reshape(bh, bw).astype("float32"), 1, window=win
                )
                writers["state_agreement"].write(
                    agreement.reshape(bh, bw).astype("uint8"), 1, window=win
                )

                if (
                    block_no == 1
                    or block_no % report_every == 0
                    or block_no == len(blocks)
                ):
                    elapsed = time.perf_counter() - t0
                    frac = block_no / len(blocks)
                    eta = elapsed * (1.0 / frac - 1.0)
                    print(
                        f"[VALIDATE] {block_no}/{len(blocks)} "
                        f"({100*frac:5.1f}%) "
                        f"elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m",
                        flush=True,
                    )
    finally:
        for dst in writers.values():
            dst.close()
        for dst in holdout_writers.values():
            dst.close()
        for src in [
            src_linear_rmse,
            src_quad_rmse,
            src_nobs,
            src_end,
            src_bic,
        ]:
            src.close()

    # ---- exact recent/end summary from output rasters ----
    recent_summary = summarize_recent_vs_end(
        deformation_dir / "end_rate_mm_yr.tif",
        output_paths["recent_velocity"],
        output_paths["end_minus_recent"],
        args.state_threshold_mm_yr,
        args.projected_crs,
        args.aggregate_km,
    )

    # ---- BIC distribution already generated by decompose step ----
    bic = raster_values(
        deformation_dir / "delta_bic_linear_minus_quadratic.tif"
    )
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
        n = max(a["valid_pixel_rmse"], 1)
        holdout_summary[str(year)] = {
            "valid_pixels": int(a["valid_pixel_rmse"]),
            "mean_pixel_rmse_linear_mm": float(
                math.sqrt(a["linear_sse"] / n)
            ),
            "mean_pixel_rmse_quadratic_mm": float(
                math.sqrt(a["quadratic_sse"] / n)
            ),
            "quadratic_lower_rmse_fraction": float(
                a["quad_better_pixels"] / n
            ),
        }

    summary = {
        "status": "ok",
        "model_definitions": {
            "M0": "linear",
            "M1": "linear + annual harmonic",
            "M2": "quadratic + annual harmonic",
        },
        "f_test": {
            "assumption_note": (
                "Classical nested-model F test; temporal residual autocorrelation "
                "is not explicitly corrected, so treat p-values as model diagnostics."
            ),
            "alpha": float(args.alpha),
            "annual_component": {
                "comparison": "M0_vs_M1",
                "valid_pixels": int(annual_valid),
                "significant_pixels": int(annual_sig),
                "significant_fraction": (
                    float(annual_sig / annual_valid)
                    if annual_valid
                    else np.nan
                ),
            },
            "quadratic_component": {
                "comparison": "M1_vs_M2",
                "valid_pixels": int(quad_valid),
                "significant_pixels": int(quad_sig),
                "significant_fraction": (
                    float(quad_sig / quad_valid)
                    if quad_valid
                    else np.nan
                ),
                "bic_positive_and_f_significant_fraction_of_bic_positive": (
                    float(quad_bic_and_f / quad_bic_positive)
                    if quad_bic_positive
                    else np.nan
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
        "outputs": {k: str(v) for k, v in output_paths.items()},
    }

    summary_path = outdir / "decomposition_validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Compact CSV for holdout.
    if holdout_summary:
        pd.DataFrame.from_dict(
            holdout_summary,
            orient="index",
        ).rename_axis("year").reset_index().to_csv(
            outdir / "temporal_holdout_summary.csv",
            index=False,
        )

    # Print the scientifically important results.
    print("\n================ VALIDATION SUMMARY ================")
    print(
        "Annual F-test significant fraction = "
        f"{summary['f_test']['annual_component']['significant_fraction']:.4f}"
    )
    print(
        "Quadratic F-test significant fraction = "
        f"{summary['f_test']['quadratic_component']['significant_fraction']:.4f}"
    )
    print(
        "Quadratic: BIC-positive pixels also F-significant = "
        f"{summary['f_test']['quadratic_component']['bic_positive_and_f_significant_fraction_of_bic_positive']:.4f}"
    )
    print(
        "Recent vs terminal rate pixel Pearson/Spearman = "
        f"{recent_summary['pearson_r']:.4f} / {recent_summary['spearman_r']:.4f}"
    )
    print(
        f"{args.aggregate_km:g}-km aggregated Pearson/Spearman = "
        f"{recent_summary['aggregated']['pearson_r']:.4f} / "
        f"{recent_summary['aggregated']['spearman_r']:.4f}"
    )
    print(
        "State agreement = "
        f"{recent_summary['state_agreement_fraction']:.4f}"
    )
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
            f"Holdout {year}: linear RMSE={row['mean_pixel_rmse_linear_mm']:.3f} mm, "
            f"quadratic RMSE={row['mean_pixel_rmse_quadratic_mm']:.3f} mm, "
            f"quad-better fraction={row['quadratic_lower_rmse_fraction']:.4f}"
        )
    print("summary:", summary_path)
    print("====================================================")


if __name__ == "__main__":
    main()
