#!/usr/bin/env python3
"""
Stage-1 validation of the apparent 2024-2025 subsidence re-acceleration
in Hengshui.

This deliberately avoids using the full-period quadratic terminal rate as
the primary evidence.

Validation A: same-season interannual differences
-------------------------------------------------
For each adjacent year pair, acquisitions in January-August of year Y are
matched one-to-one to the nearest acquisition around the same calendar date
in Y+1 (default tolerance ±18 days). For every matched pair:

    v = [D(t2) - D(t1)] / [(t2-t1)/365.2425]

The per-pixel median across all matched pairs is used as a robust annual
same-season rate. Because both acquisitions are at nearly the same season,
the annual harmonic largely cancels without requiring a trend model.

Validation B: overlapping recent linear+annual windows
------------------------------------------------------
Independently fit:

    D(t) = c + v*t + S*sin(2*pi*t) + C*cos(2*pi*t)

over four overlapping windows:
    2020-2022, 2021-2023, 2022-2024, 2023-2025

This does not use the full-period quadratic term.

Sign convention
---------------
Canonical InSAR displacement is +uplift, so:
    negative velocity = subsidence
    negative rate difference = later interval is more subsiding
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

from hydrogeo_insar.common import (
    block_slices,
    days_to_dates,
    h5_grid_metadata,
)
from hydrogeo_insar.temporal.fit import fit_block
from hydrogeo_insar.temporal.model import TimeModel, design_matrix


YEAR_DAYS = 365.2425


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stack",
        default="outputs_hengshui/canonical/insar_stack.h5",
    )
    p.add_argument(
        "--outdir",
        default="outputs_hengshui/validation/acceleration_2025_stage1",
    )
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--max-pair-day-diff", type=int, default=18)
    p.add_argument(
        "--same-season-start-month",
        type=int,
        default=1,
    )
    p.add_argument(
        "--same-season-end-month",
        type=int,
        default=8,
    )
    p.add_argument(
        "--same-season-start-year",
        type=int,
        default=2017,
    )
    p.add_argument(
        "--same-season-end-year",
        type=int,
        default=2024,
        help="Last first-year Y; final interval is Y -> Y+1.",
    )
    p.add_argument(
        "--window",
        action="append",
        default=None,
        help=(
            "Rate-fit window START:END. Can be repeated. "
            "Default: 2020-01-01:2022-12-31, "
            "2021-01-01:2023-12-31, "
            "2022-01-01:2024-12-31, "
            "2023-01-01:2025-08-30"
        ),
    )
    p.add_argument("--min-observations", type=int, default=24)
    p.add_argument("--period-days", type=float, default=365.2425)
    return p.parse_args()


def default_windows():
    return [
        ("2020-01-01", "2022-12-31"),
        ("2021-01-01", "2023-12-31"),
        ("2022-01-01", "2024-12-31"),
        ("2023-01-01", "2025-08-30"),
    ]


def parse_windows(values):
    if not values:
        return default_windows()
    out = []
    for item in values:
        a, b = item.split(":", 1)
        out.append((a.strip(), b.strip()))
    return out


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


def open_writer(path, grid):
    return rasterio.open(
        path,
        "w",
        **make_profile(grid),
    )


def anniversary(ts: pd.Timestamp, next_year: int):
    # Handle leap day deterministically.
    try:
        return ts.replace(year=next_year)
    except ValueError:
        return ts.replace(year=next_year, day=28)


def build_pairs(
    dates,
    year,
    start_month,
    end_month,
    max_day_diff,
):
    dt = pd.to_datetime(
        np.asarray(dates, dtype="datetime64[D]")
    )
    src_idx = [
        i
        for i, d in enumerate(dt)
        if (
            d.year == year
            and start_month <= d.month <= end_month
        )
    ]
    dst_idx = [
        i
        for i, d in enumerate(dt)
        if (
            d.year == year + 1
            and start_month <= d.month <= end_month
        )
    ]

    available = set(dst_idx)
    pairs = []

    for i in src_idx:
        d1 = dt[i]
        target = anniversary(d1, year + 1)
        candidates = []
        for j in available:
            dd = abs((dt[j] - target).days)
            if dd <= int(max_day_diff):
                candidates.append((dd, j))
        if not candidates:
            continue
        _, j = min(candidates)
        available.remove(j)

        d2 = dt[j]
        dt_days = int((d2 - d1).days)
        if dt_days <= 0:
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
                "elapsed_days": dt_days,
                "elapsed_years": dt_days / YEAR_DAYS,
            }
        )
    return pairs


def prepare_window(dates, start, end, period_days):
    start64 = np.datetime64(start, "D")
    end64 = np.datetime64(end, "D")
    mask = (dates >= start64) & (dates <= end64)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        raise ValueError(f"No InSAR dates in window {start} -> {end}")

    model = TimeModel(
        polynomial_degree=1,
        periods_days=(period_days,),
    )
    window_dates = dates[idx]
    X, _ = design_matrix(
        window_dates,
        model,
        origin=window_dates[0],
    )
    return {
        "start": start,
        "end": end,
        "idx": idx,
        "dates": window_dates,
        "X": X,
        "label": f"{start[:4]}_{end[:4]}",
    }


def exact_stats(path):
    with rasterio.open(path) as src:
        a = src.read(1).astype("float32")
        nd = src.nodata
        if nd is not None and np.isfinite(nd):
            a[a == nd] = np.nan
    x = a[np.isfinite(a)]
    if len(x) == 0:
        return {}
    return {
        "n": int(len(x)),
        "p01": float(np.percentile(x, 1)),
        "p10": float(np.percentile(x, 10)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "p90": float(np.percentile(x, 90)),
        "p99": float(np.percentile(x, 99)),
        "fraction_lt_minus5": float(np.mean(x < -5.0)),
        "fraction_lt_minus20": float(np.mean(x < -20.0)),
        "fraction_gt_plus5": float(np.mean(x > 5.0)),
    }


def difference_rasters(a_path, b_path, out_path, grid):
    """Write A-B. Negative means A is more subsiding than B."""
    profile = make_profile(grid)
    with (
        rasterio.open(a_path) as a_src,
        rasterio.open(b_path) as b_src,
        rasterio.open(out_path, "w", **profile) as dst,
    ):
        blocks = list(
            block_slices(
                grid["height"],
                grid["width"],
                512,
            )
        )
        for r0, r1, c0, c1 in blocks:
            win = rasterio.windows.Window(
                c0, r0, c1 - c0, r1 - r0
            )
            a = a_src.read(1, window=win).astype(float)
            b = b_src.read(1, window=win).astype(float)
            d = a - b
            d[~(np.isfinite(a) & np.isfinite(b))] = np.nan
            dst.write(
                d.astype("float32"),
                1,
                window=win,
            )


def difference_stats(path):
    with rasterio.open(path) as src:
        a = src.read(1).astype("float32")
    x = a[np.isfinite(a)]
    return {
        "n": int(len(x)),
        "median_delta_mm_yr": float(np.median(x)),
        "p10_delta_mm_yr": float(np.percentile(x, 10)),
        "p90_delta_mm_yr": float(np.percentile(x, 90)),
        "fraction_later_more_subsiding": float(np.mean(x < 0.0)),
        "fraction_later_more_subsiding_by_5": float(np.mean(x < -5.0)),
        "fraction_later_more_subsiding_by_10": float(np.mean(x < -10.0)),
        "fraction_later_less_subsiding_by_5": float(np.mean(x > 5.0)),
    }


def shared_limit(paths, percentile=98):
    vals = []
    for path in paths:
        with rasterio.open(path) as src:
            a = src.read(
                1,
                out_shape=(
                    max(1, src.height // 8),
                    max(1, src.width // 8),
                ),
                resampling=rasterio.enums.Resampling.nearest,
            ).astype(float)
        x = np.abs(a[np.isfinite(a)])
        if len(x):
            vals.append(
                np.percentile(x, percentile)
            )
    return max(vals) if vals else 1.0


def read_display(path, max_dim=1400):
    with rasterio.open(path) as src:
        factor = max(
            1,
            int(
                math.ceil(
                    max(src.height, src.width) / max_dim
                )
            ),
        )
        a = src.read(
            1,
            out_shape=(
                max(1, src.height // factor),
                max(1, src.width // factor),
            ),
            resampling=rasterio.enums.Resampling.nearest,
        ).astype(float)
        nd = src.nodata
        if nd is not None and np.isfinite(nd):
            a[a == nd] = np.nan
    return a


def plot_maps(paths, titles, out, ncols=4, difference=False):
    n = len(paths)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 4 * nrows),
        squeeze=False,
    )
    limit = shared_limit(paths, 98)
    cmap = "RdBu" if not difference else "RdBu"
    for ax, path, title in zip(axes.ravel(), paths, titles):
        a = read_display(path)
        im = ax.imshow(
            a,
            cmap=cmap,
            vmin=-limit,
            vmax=limit,
        )
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label="mm/yr",
        )
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_summary(annual_df, window_df, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    if len(annual_df):
        x = [
            (a + b) / 2.0
            for a, b in zip(
                annual_df["year1"],
                annual_df["year2"],
            )
        ]
        ax.plot(
            x,
            annual_df["median_rate_mm_yr"],
            marker="o",
            label="Same-season annual median",
        )
    if len(window_df):
        xw = [
            (
                int(s[:4])
                + int(e[:4])
            ) / 2.0
            for s, e in zip(
                window_df["start"],
                window_df["end"],
            )
        ]
        ax.plot(
            xw,
            window_df["median_rate_mm_yr"],
            marker="o",
            label="Linear+annual window median",
        )
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Approximate interval midpoint")
    ax.set_ylabel("Median rate (mm/yr, + uplift)")
    ax.set_title("Independent checks of recent subsidence-rate evolution")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    stack_path = Path(args.stack)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    same_dir = outdir / "same_season"
    win_dir = outdir / "window_rates"
    diff_dir = outdir / "differences"
    fig_dir = outdir / "figures"
    for p in (same_dir, win_dir, diff_dir, fig_dir):
        p.mkdir(parents=True, exist_ok=True)

    grid = h5_grid_metadata(stack_path)

    with h5py.File(stack_path, "r") as h5:
        dates = np.asarray(
            days_to_dates(h5["date_days"][:]),
            dtype="datetime64[D]",
        )

    # -------------------------
    # Build same-season pairs.
    # -------------------------
    pair_groups = {}
    pair_rows = []
    for year in range(
        args.same_season_start_year,
        args.same_season_end_year + 1,
    ):
        pairs = build_pairs(
            dates,
            year,
            args.same_season_start_month,
            args.same_season_end_month,
            args.max_pair_day_diff,
        )
        if not pairs:
            continue
        pair_groups[(year, year + 1)] = pairs
        pair_rows.extend(pairs)

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(
        outdir / "same_season_pair_table.csv",
        index=False,
    )

    print("===== SAME-SEASON PAIRS =====")
    for key, pairs in pair_groups.items():
        offsets = [
            abs(p["anniversary_offset_days"])
            for p in pairs
        ]
        print(
            f"{key[0]}->{key[1]}: "
            f"{len(pairs)} pairs, "
            f"median |date offset|={np.median(offsets):.1f} d, "
            f"max={max(offsets)} d"
        )

    # -------------------------
    # Prepare independent windows.
    # -------------------------
    windows = [
        prepare_window(
            dates,
            s,
            e,
            args.period_days,
        )
        for s, e in parse_windows(args.window)
    ]

    same_paths = {
        key: same_dir / f"same_season_rate_{key[0]}_{key[1]}_mm_yr.tif"
        for key in pair_groups
    }
    window_paths = {
        w["label"]: win_dir / f"linear_annual_rate_{w['label']}_mm_yr.tif"
        for w in windows
    }
    window_rmse_paths = {
        w["label"]: win_dir / f"linear_annual_rmse_{w['label']}_mm.tif"
        for w in windows
    }

    same_writers = {
        key: open_writer(path, grid)
        for key, path in same_paths.items()
    }
    win_writers = {
        label: open_writer(path, grid)
        for label, path in window_paths.items()
    }
    rmse_writers = {
        label: open_writer(path, grid)
        for label, path in window_rmse_paths.items()
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

            for ib, (r0, r1, c0, c1) in enumerate(
                blocks,
                start=1,
            ):
                win = rasterio.windows.Window(
                    c0, r0, c1 - c0, r1 - r0
                )

                # One HDF5 read per spatial block.
                arr = ds[:, r0:r1, c0:c1].astype("float32")
                T, bh, bw = arr.shape
                y = arr.reshape(T, -1)
                npix = y.shape[1]

                # A) Same-season adjacent-year rates.
                for key, pairs in pair_groups.items():
                    pair_rates = []
                    for pair in pairs:
                        a = y[pair["i1"]]
                        b = y[pair["i2"]]
                        rate = (
                            (b - a)
                            / float(pair["elapsed_years"])
                        )
                        pair_rates.append(rate)

                    stack = np.vstack(pair_rates)
                    rate_med = np.nanmedian(
                        stack,
                        axis=0,
                    )
                    same_writers[key].write(
                        rate_med.reshape(bh, bw).astype("float32"),
                        1,
                        window=win,
                    )

                # B) Overlapping linear+annual windows.
                for w in windows:
                    idx = w["idx"]
                    yw = y[idx]
                    X = w["X"]

                    beta, rmse, _, _ = fit_block(
                        yw,
                        X,
                        min_obs=max(
                            args.min_observations,
                            X.shape[1] + 2,
                        ),
                    )
                    rate = beta[:, 1]

                    win_writers[w["label"]].write(
                        rate.reshape(bh, bw).astype("float32"),
                        1,
                        window=win,
                    )
                    rmse_writers[w["label"]].write(
                        rmse.reshape(bh, bw).astype("float32"),
                        1,
                        window=win,
                    )

                if (
                    ib == 1
                    or ib % report_every == 0
                    or ib == len(blocks)
                ):
                    elapsed = time.perf_counter() - t0
                    frac = ib / len(blocks)
                    eta = elapsed * (1.0 / frac - 1.0)
                    print(
                        f"[ACCEL-STAGE1] {ib}/{len(blocks)} "
                        f"({100*frac:5.1f}%) "
                        f"elapsed={elapsed/60:.1f}m "
                        f"ETA={eta/60:.1f}m",
                        flush=True,
                    )
    finally:
        for d in (
            same_writers,
            win_writers,
            rmse_writers,
        ):
            for dst in d.values():
                dst.close()

    # -------------------------
    # Direct comparison rasters.
    # -------------------------
    diff_specs = []

    def add_same_diff(later, earlier, name):
        if later in same_paths and earlier in same_paths:
            out = diff_dir / f"{name}.tif"
            difference_rasters(
                same_paths[later],
                same_paths[earlier],
                out,
                grid,
            )
            diff_specs.append(
                (
                    name,
                    out,
                    f"{later[0]}-{later[1]} minus "
                    f"{earlier[0]}-{earlier[1]}",
                )
            )

    add_same_diff(
        (2024, 2025),
        (2023, 2024),
        "delta_same_season_2024_2025_minus_2023_2024_mm_yr",
    )
    add_same_diff(
        (2024, 2025),
        (2022, 2023),
        "delta_same_season_2024_2025_minus_2022_2023_mm_yr",
    )
    add_same_diff(
        (2024, 2025),
        (2021, 2022),
        "delta_same_season_2024_2025_minus_2021_2022_mm_yr",
    )

    if (
        "2023_2025" in window_paths
        and "2021_2023" in window_paths
    ):
        out = diff_dir / (
            "delta_window_2023_2025_minus_2021_2023_mm_yr.tif"
        )
        difference_rasters(
            window_paths["2023_2025"],
            window_paths["2021_2023"],
            out,
            grid,
        )
        diff_specs.append(
            (
                "delta_window_2023_2025_minus_2021_2023_mm_yr",
                out,
                "2023-2025 rate minus 2021-2023 rate",
            )
        )

    # -------------------------
    # Exact summaries.
    # -------------------------
    annual_rows = []
    for key, path in same_paths.items():
        s = exact_stats(path)
        annual_rows.append({
            "year1": key[0],
            "year2": key[1],
            "pair_count": len(pair_groups[key]),
            "median_rate_mm_yr": s.get("median"),
            "mean_rate_mm_yr": s.get("mean"),
            "p10_rate_mm_yr": s.get("p10"),
            "p90_rate_mm_yr": s.get("p90"),
            "fraction_rate_lt_minus5": s.get("fraction_lt_minus5"),
            "fraction_rate_lt_minus20": s.get("fraction_lt_minus20"),
            "fraction_rate_gt_plus5": s.get("fraction_gt_plus5"),
            "path": str(path),
        })

    window_rows = []
    for w in windows:
        s = exact_stats(window_paths[w["label"]])
        window_rows.append({
            "start": w["start"],
            "end": w["end"],
            "epochs": len(w["idx"]),
            "median_rate_mm_yr": s.get("median"),
            "mean_rate_mm_yr": s.get("mean"),
            "p10_rate_mm_yr": s.get("p10"),
            "p90_rate_mm_yr": s.get("p90"),
            "fraction_rate_lt_minus5": s.get("fraction_lt_minus5"),
            "fraction_rate_lt_minus20": s.get("fraction_lt_minus20"),
            "fraction_rate_gt_plus5": s.get("fraction_gt_plus5"),
            "path": str(window_paths[w["label"]]),
            "rmse_path": str(window_rmse_paths[w["label"]]),
        })

    diff_rows = []
    for name, path, description in diff_specs:
        s = difference_stats(path)
        diff_rows.append({
            "comparison": name,
            "description": description,
            **s,
            "path": str(path),
        })

    annual_df = pd.DataFrame(annual_rows)
    window_df = pd.DataFrame(window_rows)
    diff_df = pd.DataFrame(diff_rows)

    annual_df.to_csv(
        outdir / "same_season_annual_rate_summary.csv",
        index=False,
    )
    window_df.to_csv(
        outdir / "window_rate_summary.csv",
        index=False,
    )
    diff_df.to_csv(
        outdir / "rate_difference_summary.csv",
        index=False,
    )

    # -------------------------
    # QC figures.
    # -------------------------
    if same_paths:
        keys = list(same_paths)
        plot_maps(
            [same_paths[k] for k in keys],
            [
                f"{k[0]}-{k[1]} same-season rate"
                for k in keys
            ],
            fig_dir / "01_same_season_annual_rates.png",
            ncols=4,
        )

    plot_maps(
        [window_paths[w["label"]] for w in windows],
        [
            f"{w['start'][:4]}-{w['end'][:4]} linear+annual"
            for w in windows
        ],
        fig_dir / "02_overlapping_window_rates.png",
        ncols=2,
    )

    if diff_specs:
        plot_maps(
            [x[1] for x in diff_specs],
            [x[2] for x in diff_specs],
            fig_dir / "03_rate_differences.png",
            ncols=2,
            difference=True,
        )

    plot_summary(
        annual_df,
        window_df,
        fig_dir / "04_regional_rate_summary.png",
    )

    summary = {
        "status": "ok",
        "sign_convention": (
            "positive=uplift; negative=subsidence; "
            "negative rate difference means later interval is "
            "more subsiding"
        ),
        "same_season": {
            "months": [
                int(args.same_season_start_month),
                int(args.same_season_end_month),
            ],
            "max_pair_day_diff": int(args.max_pair_day_diff),
            "intervals": annual_rows,
        },
        "window_rates": window_rows,
        "differences": diff_rows,
        "files": {
            "pair_table": str(
                outdir / "same_season_pair_table.csv"
            ),
            "annual_summary": str(
                outdir / "same_season_annual_rate_summary.csv"
            ),
            "window_summary": str(
                outdir / "window_rate_summary.csv"
            ),
            "difference_summary": str(
                outdir / "rate_difference_summary.csv"
            ),
            "figure_directory": str(fig_dir),
        },
    }

    summary_path = outdir / "stage1_validation_summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n================ STAGE-1 SUMMARY ================")
    print("\nSame-season annual rates (+uplift, -subsidence):")
    print(
        annual_df[
            [
                "year1",
                "year2",
                "pair_count",
                "median_rate_mm_yr",
                "p10_rate_mm_yr",
                "p90_rate_mm_yr",
            ]
        ].to_string(index=False)
    )

    print("\nIndependent linear+annual window rates:")
    print(
        window_df[
            [
                "start",
                "end",
                "epochs",
                "median_rate_mm_yr",
                "p10_rate_mm_yr",
                "p90_rate_mm_yr",
            ]
        ].to_string(index=False)
    )

    print("\nRate differences:")
    print(
        diff_df[
            [
                "description",
                "median_delta_mm_yr",
                "fraction_later_more_subsiding",
                "fraction_later_more_subsiding_by_5",
                "fraction_later_more_subsiding_by_10",
            ]
        ].to_string(index=False)
    )

    print("\nsummary:", summary_path)
    print("=================================================")


if __name__ == "__main__":
    main()
