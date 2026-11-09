#!/usr/bin/env python3
"""
Stage-2 diagnostic: is the 2024-2025 Hengshui subsidence intensification
likely to be a real deformation signal or an InSAR processing artifact?

This script does NOT try to "prove" causality. It attacks the main failure
modes that could mimic a late-time acceleration:

A) acquisition-specific anomalies
   - full-period quadratic+annual residuals on a spatial sample
   - robust residual scatter
   - long-wavelength planar ramp amplitude
   - common-mode residual
   - rank suspicious 2024-2025 acquisitions

B) sensitivity to suspicious acquisitions
   - recompute the 2024-2025 same-season rate after excluding the 5 and 10
     most anomalous 2024-2025 acquisitions
   - compare spatial correlation and median shift with the original result

C) pair-jackknife stability
   - remove one 2024-2025 same-season pair at a time on a spatial sample
   - quantify the maximum shift in regional median rate

D) spatial-pattern diagnostics
   - does the 2024-2025 acceleration resemble older subsidence patterns?
   - how much of the acceleration map is explained by a simple spatial plane?
     (large planar dominance is suspicious for residual orbit/long-wave error)

E) groundwater consistency
   - independently compute same-season annual groundwater-head changes
   - compare 2024-2025 head change with 2024-2025 deformation on a 5-km
     aggregated grid
   - this is supporting evidence only: weak correlation does NOT prove an
     InSAR artifact because delayed aquitard compaction is physically possible.

Outputs are diagnostic evidence, not a formal probability that the signal is
"real" or "artifact".
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.stats import pearsonr, spearmanr

from hydrogeo_insar.common import (
    block_slices,
    days_to_dates,
    h5_grid_metadata,
)

YEAR_DAYS = 365.2425


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stack",
        default="outputs_hengshui/canonical/insar_stack.h5",
    )
    p.add_argument(
        "--deformation-dir",
        default="outputs_hengshui/deformation",
    )
    p.add_argument(
        "--stage1-dir",
        default="outputs_hengshui/validation/acceleration_2025_stage1",
    )
    p.add_argument(
        "--groundwater",
        default="outputs_hengshui/groundwater/groundwater_field.h5",
    )
    p.add_argument(
        "--groundwater-distance",
        default="outputs_hengshui/groundwater/groundwater_nearest_well_distance_km.tif",
    )
    p.add_argument(
        "--outdir",
        default="outputs_hengshui/validation/acceleration_2025_stage2",
    )
    p.add_argument("--sample-stride", type=int, default=24)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--max-pair-day-diff", type=int, default=18)
    p.add_argument("--support-distance-km", type=float, default=15.0)
    p.add_argument("--aggregate-km", type=float, default=5.0)
    p.add_argument(
        "--projected-crs",
        default="EPSG:32650",
    )
    return p.parse_args()


def read_exact_stride(path: Path, stride: int):
    with rasterio.open(path) as src:
        a = src.read(1).astype("float32")
        nd = src.nodata
        if nd is not None and np.isfinite(nd):
            a[a == nd] = np.nan
    return a[::stride, ::stride]


def robust_z(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad <= 0:
        return np.zeros_like(x)
    return 0.6744897501960817 * (x - med) / mad


def corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return {
            "n": int(ok.sum()),
            "pearson_r": np.nan,
            "spearman_r": np.nan,
        }
    return {
        "n": int(ok.sum()),
        "pearson_r": float(pearsonr(x[ok], y[ok]).statistic),
        "spearman_r": float(spearmanr(x[ok], y[ok]).statistic),
    }


def anniversary(ts: pd.Timestamp, next_year: int):
    try:
        return ts.replace(year=next_year)
    except ValueError:
        return ts.replace(year=next_year, day=28)


def build_pairs(dates, year, max_day_diff=18, start_month=1, end_month=8):
    dt = pd.to_datetime(np.asarray(dates, dtype="datetime64[D]"))
    src = [
        i for i, d in enumerate(dt)
        if d.year == year and start_month <= d.month <= end_month
    ]
    dst = [
        i for i, d in enumerate(dt)
        if d.year == year + 1 and start_month <= d.month <= end_month
    ]
    available = set(dst)
    pairs = []
    for i in src:
        target = anniversary(dt[i], year + 1)
        cand = [
            (abs((dt[j] - target).days), j)
            for j in available
            if abs((dt[j] - target).days) <= int(max_day_diff)
        ]
        if not cand:
            continue
        _, j = min(cand)
        available.remove(j)
        elapsed_days = int((dt[j] - dt[i]).days)
        if elapsed_days <= 0:
            continue
        pairs.append({
            "i1": int(i),
            "i2": int(j),
            "date1": str(dt[i].date()),
            "date2": str(dt[j].date()),
            "elapsed_days": elapsed_days,
            "elapsed_years": elapsed_days / YEAR_DAYS,
            "anniversary_offset_days": int((dt[j] - target).days),
        })
    return pairs


def sample_xy(grid, stride, projected_crs):
    rows = np.arange(0, grid["height"], stride)
    cols = np.arange(0, grid["width"], stride)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    xs, ys = rasterio.transform.xy(
        grid["transform"],
        rr.ravel(),
        cc.ravel(),
        offset="center",
    )
    tr = Transformer.from_crs(
        grid["crs"],
        projected_crs,
        always_xy=True,
    )
    xx, yy = tr.transform(np.asarray(xs), np.asarray(ys))
    return (
        np.asarray(xx).reshape(rr.shape),
        np.asarray(yy).reshape(rr.shape),
    )


def aggregate_5km(x, y, values, valid, km):
    block = float(km) * 1000.0
    ok = valid & np.isfinite(values) & np.isfinite(x) & np.isfinite(y)
    if ok.sum() == 0:
        return pd.DataFrame(columns=["bx", "by", "value"])

    xx = x[ok]
    yy = y[ok]
    vv = values[ok]

    bx = np.floor((xx - np.nanmin(x[valid])) / block).astype(int)
    by = np.floor((yy - np.nanmin(y[valid])) / block).astype(int)

    df = pd.DataFrame({
        "bx": bx,
        "by": by,
        "value": vv,
    })
    return (
        df.groupby(["bx", "by"], as_index=False)["value"]
        .median()
    )


def aggregate_pair_corr(x, y, a, b, valid, km):
    block = float(km) * 1000.0
    ok = (
        valid
        & np.isfinite(a)
        & np.isfinite(b)
        & np.isfinite(x)
        & np.isfinite(y)
    )
    if ok.sum() == 0:
        return {
            "n_cells": 0,
            "pearson_r": np.nan,
            "spearman_r": np.nan,
        }

    xmin = np.nanmin(x[ok])
    ymin = np.nanmin(y[ok])
    bx = np.floor((x[ok] - xmin) / block).astype(int)
    by = np.floor((y[ok] - ymin) / block).astype(int)

    df = pd.DataFrame({
        "bx": bx,
        "by": by,
        "a": a[ok],
        "b": b[ok],
    })
    g = (
        df.groupby(["bx", "by"], as_index=False)
        .agg(a=("a", "median"), b=("b", "median"))
    )
    c = corr(g["a"].to_numpy(), g["b"].to_numpy())
    return {
        "n_cells": int(len(g)),
        "pearson_r": c["pearson_r"],
        "spearman_r": c["spearman_r"],
    }


def fit_plane_metrics(values, x, y, valid):
    ok = valid & np.isfinite(values) & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20:
        return {
            "plane_r2": np.nan,
            "plane_peak_to_peak_mm": np.nan,
        }

    xx = x[ok].astype(float)
    yy = y[ok].astype(float)
    zz = values[ok].astype(float)

    xn = (xx - np.mean(xx)) / max(np.std(xx), 1.0)
    yn = (yy - np.mean(yy)) / max(np.std(yy), 1.0)

    A = np.column_stack([np.ones(len(zz)), xn, yn])
    beta, *_ = np.linalg.lstsq(A, zz, rcond=None)
    pred = A @ beta

    sst = np.sum((zz - np.mean(zz)) ** 2)
    sse = np.sum((zz - pred) ** 2)
    r2 = 1.0 - sse / sst if sst > 0 else np.nan

    corners = np.array([
        [1.0, np.min(xn), np.min(yn)],
        [1.0, np.min(xn), np.max(yn)],
        [1.0, np.max(xn), np.min(yn)],
        [1.0, np.max(xn), np.max(yn)],
    ])
    plane_vals = corners @ beta
    return {
        "plane_r2": float(r2),
        "plane_peak_to_peak_mm": float(
            np.max(plane_vals) - np.min(plane_vals)
        ),
    }


def reconstruct_epoch_residual_metrics(
    stack_sample,
    dates,
    coef,
    valid,
    x,
    y,
    acceleration,
):
    t_days = (
        dates.astype("datetime64[D]")
        - dates[0].astype("datetime64[D]")
    ).astype(float)
    t_year = t_days / YEAR_DAYS

    sinv = np.sin(2.0 * np.pi * t_days / YEAR_DAYS)
    cosv = np.cos(2.0 * np.pi * t_days / YEAR_DAYS)

    rows = []

    for i in range(len(dates)):
        fit = (
            coef["intercept"]
            + coef["linear"] * t_year[i]
            + coef["quadratic"] * t_year[i] ** 2
            + coef["annual_sin"] * sinv[i]
            + coef["annual_cos"] * cosv[i]
        )
        residual = stack_sample[i] - fit
        ok = valid & np.isfinite(residual)

        r = residual[ok]
        if len(r) < 100:
            continue

        med = float(np.median(r))
        mad = float(1.4826 * np.median(np.abs(r - med)))
        p95 = float(np.percentile(np.abs(r - med), 95))

        pm = fit_plane_metrics(residual, x, y, ok)
        cr = corr(residual[ok], acceleration[ok])

        rows.append({
            "epoch_index": i,
            "date": str(dates[i]),
            "spatial_median_residual_mm": med,
            "spatial_robust_sigma_mm": mad,
            "p95_abs_centered_residual_mm": p95,
            "plane_r2": pm["plane_r2"],
            "plane_peak_to_peak_mm": pm["plane_peak_to_peak_mm"],
            "residual_vs_acceleration_pearson": cr["pearson_r"],
        })

    df = pd.DataFrame(rows)

    z_noise = np.maximum(
        robust_z(df["spatial_robust_sigma_mm"].to_numpy()),
        0.0,
    )
    z_ramp = np.maximum(
        robust_z(df["plane_peak_to_peak_mm"].to_numpy()),
        0.0,
    )

    # Deliberately exclude the common-mode median from the artifact score:
    # a true region-wide late deformation change can shift the median.
    df["scene_artifact_score"] = np.sqrt(z_noise**2 + z_ramp**2)
    df["rank_all"] = (
        df["scene_artifact_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    recent = pd.to_datetime(df["date"]) >= pd.Timestamp("2024-01-01")
    ranks = pd.Series(np.nan, index=df.index)
    ranks.loc[recent] = (
        df.loc[recent, "scene_artifact_score"]
        .rank(method="first", ascending=False)
    )
    df["rank_2024_2025"] = ranks
    return df


def pair_jackknife(
    stack_sample,
    pairs,
    valid,
):
    rate_stack = []
    labels = []

    for p in pairs:
        rate = (
            stack_sample[p["i2"]]
            - stack_sample[p["i1"]]
        ) / p["elapsed_years"]
        rate_stack.append(rate)
        labels.append(f"{p['date1']}->{p['date2']}")

    rates = np.stack(rate_stack, axis=0)
    full = np.nanmedian(rates, axis=0)
    full_regional = float(np.nanmedian(full[valid]))

    rows = []
    for k, label in enumerate(labels):
        keep = np.ones(len(labels), dtype=bool)
        keep[k] = False
        reduced = np.nanmedian(rates[keep], axis=0)
        reg = float(np.nanmedian(reduced[valid]))
        rows.append({
            "removed_pair": label,
            "regional_median_rate_mm_yr": reg,
            "shift_from_full_mm_yr": reg - full_regional,
        })

    df = pd.DataFrame(rows)
    return full_regional, df


def recompute_same_season_sensitivity(
    stack_path,
    grid,
    pairs,
    excluded_dates_by_label,
    outdir,
    block_size,
):
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    usable = {}
    for label, excluded in excluded_dates_by_label.items():
        excluded = set(excluded)
        pp = [
            p for p in pairs
            if p["date1"] not in excluded
            and p["date2"] not in excluded
        ]
        usable[label] = pp
        outputs[label] = outdir / f"same_season_rate_2024_2025_{label}_mm_yr.tif"

    profile = {
        "driver": "GTiff",
        "height": grid["height"],
        "width": grid["width"],
        "count": 1,
        "dtype": "float32",
        "crs": grid["crs"],
        "transform": grid["transform"],
        "compress": "deflate",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
        "nodata": np.nan,
    }
    writers = {
        label: rasterio.open(path, "w", **profile)
        for label, path in outputs.items()
    }

    try:
        with h5py.File(stack_path, "r") as h5:
            ds = h5["displacement_mm"]
            blocks = list(
                block_slices(
                    grid["height"],
                    grid["width"],
                    block_size,
                )
            )
            report_every = max(1, len(blocks) // 20)

            for ib, (r0, r1, c0, c1) in enumerate(blocks, start=1):
                win = rasterio.windows.Window(
                    c0, r0, c1-c0, r1-r0
                )
                for label, pp in usable.items():
                    if len(pp) == 0:
                        continue
                    indices = sorted(
                        set(
                            [p["i1"] for p in pp]
                            + [p["i2"] for p in pp]
                        )
                    )
                    arr = ds[
                        indices,
                        r0:r1,
                        c0:c1,
                    ].astype("float32")
                    pos = {idx: j for j, idx in enumerate(indices)}
                    rates = []
                    for p in pp:
                        rate = (
                            arr[pos[p["i2"]]]
                            - arr[pos[p["i1"]]]
                        ) / p["elapsed_years"]
                        rates.append(rate)
                    med = np.nanmedian(
                        np.stack(rates, axis=0),
                        axis=0,
                    )
                    writers[label].write(
                        med.astype("float32"),
                        1,
                        window=win,
                    )

                if (
                    ib == 1
                    or ib % report_every == 0
                    or ib == len(blocks)
                ):
                    print(
                        f"[SENSITIVITY] {ib}/{len(blocks)}",
                        flush=True,
                    )
    finally:
        for dst in writers.values():
            dst.close()

    return outputs, {
        label: len(pp)
        for label, pp in usable.items()
    }


def compare_rasters_sample(a_path, b_path, stride):
    a = read_exact_stride(Path(a_path), stride)
    b = read_exact_stride(Path(b_path), stride)
    ok = np.isfinite(a) & np.isfinite(b)
    c = corr(a[ok], b[ok])
    d = a[ok] - b[ok]
    return {
        **c,
        "median_difference_mm_yr": float(np.median(d)),
        "mae_difference_mm_yr": float(np.mean(np.abs(d))),
        "p90_abs_difference_mm_yr": float(
            np.percentile(np.abs(d), 90)
        ),
    }


def groundwater_same_season_sample(
    gw_path,
    stride,
    max_pair_day_diff,
):
    if not Path(gw_path).exists():
        return None, []

    with h5py.File(gw_path, "r") as h5:
        dates = np.asarray(
            days_to_dates(h5["date_days"][:]),
            dtype="datetime64[D]",
        )
        data = h5["head_anomaly_m"][:, ::stride, ::stride].astype("float32")

    summaries = []
    maps = {}

    for year in (2021, 2022, 2023, 2024):
        pairs = build_pairs(
            dates,
            year,
            max_day_diff=max_pair_day_diff,
            start_month=1,
            end_month=8,
        )
        if not pairs:
            continue
        rates = []
        for p in pairs:
            rates.append(
                (data[p["i2"]] - data[p["i1"]])
                / p["elapsed_years"]
            )
        med = np.nanmedian(np.stack(rates, axis=0), axis=0)
        maps[(year, year+1)] = med
        summaries.append({
            "year1": year,
            "year2": year + 1,
            "pair_count": len(pairs),
            "median_head_change_m_yr": float(
                np.nanmedian(med)
            ),
            "p10_head_change_m_yr": float(
                np.nanpercentile(med, 10)
            ),
            "p90_head_change_m_yr": float(
                np.nanpercentile(med, 90)
            ),
        })

    return maps, summaries


def diagnostic_assessment(
    sensitivity,
    jackknife_df,
    acceleration_plane,
    historical_corr,
    epoch_df,
):
    """Project QC heuristic, not a formal probability."""
    evidence_real = []
    evidence_artifact = []

    # Sensitivity to suspicious acquisitions.
    for label, s in sensitivity.items():
        if label == "original":
            continue
        if (
            np.isfinite(s.get("pearson_r", np.nan))
            and s["pearson_r"] >= 0.90
            and abs(s["median_difference_mm_yr"]) <= 5.0
        ):
            evidence_real.append(
                f"{label}: acceleration pattern stable after excluding suspicious acquisitions"
            )
        elif (
            np.isfinite(s.get("pearson_r", np.nan))
            and s["pearson_r"] < 0.70
        ):
            evidence_artifact.append(
                f"{label}: acceleration pattern is highly sensitive to acquisition removal"
            )

    # Pair jackknife.
    max_shift = float(
        jackknife_df["shift_from_full_mm_yr"].abs().max()
    )
    if max_shift <= 5.0:
        evidence_real.append(
            "No single same-season pair controls the regional 2024-2025 rate"
        )
    elif max_shift >= 10.0:
        evidence_artifact.append(
            "A single same-season pair strongly controls the regional result"
        )

    # Long-wavelength plane.
    if (
        np.isfinite(acceleration_plane["plane_r2"])
        and acceleration_plane["plane_r2"] < 0.25
    ):
        evidence_real.append(
            "Acceleration is not dominated by a simple planar long-wavelength pattern"
        )
    elif (
        np.isfinite(acceleration_plane["plane_r2"])
        and acceleration_plane["plane_r2"] > 0.60
    ):
        evidence_artifact.append(
            "A simple spatial plane explains most of the acceleration pattern"
        )

    # Historical spatial recurrence.
    hr = historical_corr.get("pearson_r", np.nan)
    if np.isfinite(hr) and hr >= 0.40:
        evidence_real.append(
            "Recent intensification spatially recurs in previously subsiding areas"
        )

    # Number of extreme-scene scores in 2025.
    recent = epoch_df[
        pd.to_datetime(epoch_df["date"]) >= pd.Timestamp("2024-01-01")
    ]
    top = recent.nsmallest(10, "rank_2024_2025")
    very_high = int((top["scene_artifact_score"] >= 5.0).sum())
    if very_high >= 5:
        evidence_artifact.append(
            "Several 2024-2025 acquisitions have very large residual-noise/ramp scores"
        )

    if len(evidence_artifact) == 0 and len(evidence_real) >= 3:
        assessment = (
            "Strong internal evidence favors a persistent deformation signal "
            "over an artifact caused by a few bad acquisitions or a simple residual ramp."
        )
    elif len(evidence_artifact) >= 2:
        assessment = (
            "The late-time signal is materially sensitive to InSAR quality diagnostics; "
            "treat the 2024-2025 acceleration as unresolved until external validation."
        )
    else:
        assessment = (
            "Evidence is mixed: the signal is not explained by a single simple failure mode, "
            "but the current diagnostics are insufficient for a definitive attribution."
        )

    return {
        "assessment": assessment,
        "note": (
            "This is a project QC heuristic, not a statistical probability. "
            "A 2025 GNSS/leveling record, a second SAR geometry/track, or independent "
            "reprocessing would provide stronger external validation."
        ),
        "evidence_favoring_real_deformation": evidence_real,
        "evidence_favoring_insar_artifact": evidence_artifact,
    }


def main():
    args = parse_args()

    stack_path = Path(args.stack)
    deform_dir = Path(args.deformation_dir)
    stage1_dir = Path(args.stage1_dir)
    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    sensdir = outdir / "sensitivity"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    sensdir.mkdir(parents=True, exist_ok=True)

    grid = h5_grid_metadata(stack_path)
    stride = int(args.sample_stride)

    with h5py.File(stack_path, "r") as h5:
        dates = np.asarray(
            days_to_dates(h5["date_days"][:]),
            dtype="datetime64[D]",
        )
        stack_sample = h5["displacement_mm"][
            :,
            ::stride,
            ::stride,
        ].astype("float32")

    # Exact-grid sampled coefficients.
    coef = {
        "intercept": read_exact_stride(
            deform_dir / "intercept_mm.tif", stride
        ),
        "linear": read_exact_stride(
            deform_dir / "linear_coeff_mm_yr.tif", stride
        ),
        "quadratic": read_exact_stride(
            deform_dir / "quadratic_coeff_mm_yr2.tif", stride
        ),
        "annual_sin": read_exact_stride(
            deform_dir / "annual_sin_mm.tif", stride
        ),
        "annual_cos": read_exact_stride(
            deform_dir / "annual_cos_mm.tif", stride
        ),
    }

    acceleration_path = (
        stage1_dir
        / "differences"
        / "delta_same_season_2024_2025_minus_2023_2024_mm_yr.tif"
    )
    original_2425_path = (
        stage1_dir
        / "same_season"
        / "same_season_rate_2024_2025_mm_yr.tif"
    )
    historical_path = (
        stage1_dir
        / "same_season"
        / "same_season_rate_2019_2020_mm_yr.tif"
    )
    pair_table_path = stage1_dir / "same_season_pair_table.csv"

    for p in [
        acceleration_path,
        original_2425_path,
        historical_path,
        pair_table_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Required Stage-1 output missing: {p}")

    acceleration = read_exact_stride(acceleration_path, stride)
    historical = read_exact_stride(historical_path, stride)
    rmse_sample = read_exact_stride(
        deform_dir / "fit_rmse_mm.tif",
        stride,
    )

    valid = (
        np.isfinite(acceleration)
        & np.isfinite(coef["intercept"])
    )

    x, y = sample_xy(
        grid,
        stride,
        args.projected_crs,
    )

    print(
        f"[STAGE2] sample={stack_sample.shape}, "
        f"valid_sample_pixels={int(valid.sum())}",
        flush=True,
    )

    # ------------------------------------------------------------
    # A. Epoch-level residual / ramp diagnostics
    # ------------------------------------------------------------
    epoch_df = reconstruct_epoch_residual_metrics(
        stack_sample,
        dates,
        coef,
        valid,
        x,
        y,
        acceleration,
    )
    epoch_df.to_csv(
        outdir / "epoch_quality_diagnostics.csv",
        index=False,
    )

    recent_ranked = (
        epoch_df[
            pd.to_datetime(epoch_df["date"])
            >= pd.Timestamp("2024-01-01")
        ]
        .sort_values("scene_artifact_score", ascending=False)
        .reset_index(drop=True)
    )
    recent_ranked.to_csv(
        outdir / "epoch_quality_ranked_2024_2025.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # B. 2024-2025 pair jackknife
    # ------------------------------------------------------------
    pair_df = pd.read_csv(pair_table_path)
    pair_2425 = (
        pair_df[
            (pair_df["year1"] == 2024)
            & (pair_df["year2"] == 2025)
        ]
        .copy()
    )
    pairs = pair_2425.to_dict("records")

    full_regional_rate, jackknife_df = pair_jackknife(
        stack_sample,
        pairs,
        valid,
    )
    jackknife_df.to_csv(
        outdir / "same_season_2024_2025_pair_jackknife.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # C. Full-resolution sensitivity excluding suspicious dates
    # ------------------------------------------------------------
    top5 = recent_ranked.head(5)["date"].tolist()
    top10 = recent_ranked.head(10)["date"].tolist()

    sens_outputs, pair_counts = recompute_same_season_sensitivity(
        stack_path,
        grid,
        pairs,
        {
            "exclude_top5_scene_anomalies": top5,
            "exclude_top10_scene_anomalies": top10,
        },
        sensdir,
        args.block_size,
    )

    sensitivity = {
        "original": {
            "pair_count": len(pairs),
            "regional_median_rate_mm_yr_sample": full_regional_rate,
        }
    }
    for label, path in sens_outputs.items():
        c = compare_rasters_sample(
            path,
            original_2425_path,
            stride,
        )
        c["pair_count"] = pair_counts[label]
        c["excluded_dates"] = (
            top5 if "top5" in label else top10
        )
        sensitivity[label] = c

    # ------------------------------------------------------------
    # D. Spatial pattern / ramp diagnostics
    # ------------------------------------------------------------
    acceleration_plane = fit_plane_metrics(
        acceleration,
        x,
        y,
        valid,
    )

    hist_corr = aggregate_pair_corr(
        x,
        y,
        acceleration,
        historical,
        valid,
        args.aggregate_km,
    )
    rmse_corr = aggregate_pair_corr(
        x,
        y,
        np.abs(acceleration),
        rmse_sample,
        valid,
        args.aggregate_km,
    )

    # ------------------------------------------------------------
    # E. Groundwater consistency on the exact sampled grid
    # ------------------------------------------------------------
    gw_maps, gw_rows = groundwater_same_season_sample(
        args.groundwater,
        stride,
        args.max_pair_day_diff,
    )

    gw_summary = {
        "available": gw_maps is not None,
        "same_season_intervals": gw_rows,
    }

    if gw_maps is not None and (2024, 2025) in gw_maps:
        head_2425 = gw_maps[(2024, 2025)]

        distance_path = Path(args.groundwater_distance)
        if distance_path.exists():
            dist = read_exact_stride(
                distance_path,
                stride,
            )
            support = (
                valid
                & np.isfinite(dist)
                & (dist <= float(args.support_distance_km))
            )
        else:
            support = valid.copy()

        def_rate = read_exact_stride(
            original_2425_path,
            stride,
        )

        gw_corr = aggregate_pair_corr(
            x,
            y,
            def_rate,
            head_2425,
            support,
            args.aggregate_km,
        )
        # Same sign: falling head (negative) with subsidence (negative)
        # gives positive correlation.
        gw_summary.update({
            "support_distance_km": float(args.support_distance_km),
            "support_sample_pixels": int(support.sum()),
            "deformation_vs_head_2024_2025_5km": gw_corr,
            "regional_median_head_change_2024_2025_m_yr": float(
                np.nanmedian(head_2425[support])
            ),
            "regional_median_deformation_rate_2024_2025_mm_yr": float(
                np.nanmedian(def_rate[support])
            ),
        })

    # ------------------------------------------------------------
    # Diagnostic assessment
    # ------------------------------------------------------------
    assessment = diagnostic_assessment(
        sensitivity,
        jackknife_df,
        acceleration_plane,
        hist_corr,
        epoch_df,
    )

    # ------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------
    # Epoch metrics.
    ed = epoch_df.copy()
    ed["date"] = pd.to_datetime(ed["date"])
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(
        ed["date"],
        ed["spatial_robust_sigma_mm"],
    )
    axes[0].set_ylabel("Robust residual sigma (mm)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        ed["date"],
        ed["plane_peak_to_peak_mm"],
    )
    axes[1].set_ylabel("Residual plane P-P (mm)")
    axes[1].grid(alpha=0.25)

    axes[2].plot(
        ed["date"],
        ed["scene_artifact_score"],
    )
    axes[2].set_ylabel("Scene anomaly score")
    axes[2].grid(alpha=0.25)
    axes[2].set_xlabel("Date")

    for ax in axes:
        ax.axvline(pd.Timestamp("2024-01-01"), linestyle="--", linewidth=1)

    fig.suptitle("Acquisition-level residual quality diagnostics")
    fig.tight_layout()
    fig.savefig(
        figdir / "01_epoch_quality_diagnostics.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Spatial evidence.
    original = read_exact_stride(
        original_2425_path,
        stride,
    )
    ex5 = read_exact_stride(
        sens_outputs["exclude_top5_scene_anomalies"],
        stride,
    )
    ex10 = read_exact_stride(
        sens_outputs["exclude_top10_scene_anomalies"],
        stride,
    )

    finite_vals = np.concatenate([
        np.abs(original[np.isfinite(original)]),
        np.abs(ex5[np.isfinite(ex5)]),
        np.abs(ex10[np.isfinite(ex10)]),
    ])
    lim = float(np.percentile(finite_vals, 98))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, arr, title in [
        (axes[0], original, "Original 2024-2025"),
        (axes[1], ex5, "Exclude top-5 scene anomalies"),
        (axes[2], ex10, "Exclude top-10 scene anomalies"),
    ]:
        im = ax.imshow(arr, cmap="RdBu", vmin=-lim, vmax=lim)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(
            im, ax=ax, fraction=0.046, pad=0.04, label="mm/yr"
        )
    fig.tight_layout()
    fig.savefig(
        figdir / "02_scene_removal_sensitivity.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Pair jackknife.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axhline(
        full_regional_rate,
        linewidth=1,
        label="Full 2024-2025 pair median",
    )
    ax.plot(
        np.arange(len(jackknife_df)),
        jackknife_df["regional_median_rate_mm_yr"],
        marker="o",
        linestyle="none",
    )
    ax.set_xlabel("Removed same-season pair")
    ax.set_ylabel("Regional median rate (mm/yr)")
    ax.set_title("Leave-one-pair-out sensitivity")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figdir / "03_pair_jackknife.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Groundwater time sequence if available.
    if gw_rows:
        gd = pd.DataFrame(gw_rows)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        xx = [
            (a + b) / 2
            for a, b in zip(gd["year1"], gd["year2"])
        ]
        ax.plot(
            xx,
            gd["median_head_change_m_yr"],
            marker="o",
        )
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("Interval midpoint")
        ax.set_ylabel("Median head change (m/yr)")
        ax.set_title("Same-season confined-head change")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            figdir / "04_groundwater_same_season_change.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

    summary = {
        "status": "ok",
        "purpose": (
            "Diagnose whether the 2024-2025 intensification is sensitive to "
            "scene anomalies / long-wave ramps or is spatially and temporally "
            "persistent."
        ),
        "sample_stride": stride,
        "epoch_quality": {
            "top10_2024_2025": recent_ranked.head(10).to_dict("records"),
            "note": (
                "Scene anomaly score combines residual spatial scatter and "
                "planar-ramp amplitude; common-mode median is reported but "
                "not used in the score because a true region-wide deformation "
                "change can also shift the median."
            ),
        },
        "pair_jackknife": {
            "full_regional_median_rate_mm_yr": full_regional_rate,
            "max_abs_leave_one_pair_shift_mm_yr": float(
                jackknife_df["shift_from_full_mm_yr"].abs().max()
            ),
            "median_abs_leave_one_pair_shift_mm_yr": float(
                jackknife_df["shift_from_full_mm_yr"].abs().median()
            ),
        },
        "scene_removal_sensitivity": sensitivity,
        "spatial_pattern": {
            "acceleration_plane": acceleration_plane,
            "acceleration_vs_2019_2020_rate_5km": hist_corr,
            "abs_acceleration_vs_full_model_rmse_5km": rmse_corr,
        },
        "groundwater_consistency": gw_summary,
        "diagnostic_assessment": assessment,
        "limitations": [
            (
                "This analysis can reject some common InSAR failure modes, "
                "but cannot by itself provide external 2025 geodetic validation."
            ),
            (
                "Groundwater agreement is supportive rather than decisive; "
                "persistent/delayed compaction can continue during head recovery."
            ),
            (
                "The strongest external confirmation would be 2025 GNSS/leveling, "
                "a second independent SAR track/geometry, or an independent InSAR reprocessing."
            ),
        ],
    }

    summary_path = outdir / "stage2_artifact_vs_real_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n================ STAGE-2 DIAGNOSTIC ================")
    print("\nTop 10 suspicious 2024-2025 acquisitions:")
    print(
        recent_ranked.head(10)[
            [
                "date",
                "spatial_robust_sigma_mm",
                "plane_peak_to_peak_mm",
                "plane_r2",
                "scene_artifact_score",
            ]
        ].to_string(index=False)
    )

    print("\nPair jackknife:")
    print(
        "full regional median =",
        f"{full_regional_rate:.3f} mm/yr",
    )
    print(
        "max |leave-one-pair shift| =",
        f"{summary['pair_jackknife']['max_abs_leave_one_pair_shift_mm_yr']:.3f} mm/yr",
    )

    print("\nScene-removal sensitivity:")
    for label, s in sensitivity.items():
        print(label, ":", s)

    print("\nSpatial pattern:")
    print(
        "acceleration plane R2 =",
        acceleration_plane["plane_r2"],
    )
    print(
        f"{args.aggregate_km:g}-km corr(acceleration, 2019-2020 rate) =",
        hist_corr,
    )
    print(
        f"{args.aggregate_km:g}-km corr(|acceleration|, model RMSE) =",
        rmse_corr,
    )

    print("\nGroundwater consistency:")
    print(json.dumps(gw_summary, indent=2, ensure_ascii=False))

    print("\nDiagnostic assessment:")
    print(assessment["assessment"])
    for x in assessment["evidence_favoring_real_deformation"]:
        print("  +", x)
    for x in assessment["evidence_favoring_insar_artifact"]:
        print("  -", x)

    print("\nsummary:", summary_path)
    print("====================================================")


if __name__ == "__main__":
    main()
