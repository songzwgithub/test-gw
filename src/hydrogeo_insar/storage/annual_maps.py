"""
Export annual pixelwise storage-component maps from the existing Hengshui
pixelwise-Ske solution.

This reproduces the SAME low-frequency model already used by
storage_budget_pixelwise_fast.py:

    continuous piecewise-linear trend + annual harmonic

No new scientific model is introduced.  The purpose is to preserve the
year-by-year spatial fields that were previously only integrated into CSV.

Required existing products
---------------------------
<outputs>/canonical/insar_stack.h5
<outputs>/groundwater/groundwater_field.h5
<outputs>/seasonal/ske_pixelwise.tif
<outputs>/storage/storage_domain_mask.tif

Annual outputs
--------------
storage/annual_maps/
  YYYY_total_change_mm.tif
  YYYY_recoverable_change_mm.tif
  YYYY_irreversible_change_mm.tif
  YYYY_head_lowfreq_change_m.tif
  YYYY_recovery_with_continued_compaction.tif

The final mask is 1 where:
    recoverable > +threshold_mm
    irreversible < -threshold_mm

i.e. elastic groundwater/storage recovery coexists with continued residual
compaction during that interval.

The script also writes:
  storage/annual_maps_summary.csv

Sign convention
---------------
+ deformation/storage-equivalent thickness = uplift / recovery
- deformation/storage-equivalent thickness = subsidence / depletion
"""

from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio

from ..common import (
    block_slices,
    days_to_dates,
    ensure_dir,
    geotiff_profile,
    h5_grid_metadata,
    pixel_area_rows,
    read_tif,
)
from ..config import ProjectConfig
from ..temporal.fit import fit_block_grouped
from ..temporal.model import (
    TimeModel,
    calendar_year_knots,
    design_matrix,
    low_frequency_row,
)


def nearest_index(dates, target, default):
    if target is None:
        return default
    t = np.datetime64(str(target), "D")
    return int(
        np.argmin(
            np.abs(dates.astype("datetime64[D]") - t)
        )
    )




def open_float_writer(path, grid):
    profile = geotiff_profile(
        grid["height"],
        grid["width"],
        grid["crs"],
        grid["transform"],
        dtype="float32",
        nodata=np.nan,
    )
    profile["NUM_THREADS"] = "ALL_CPUS"
    return rasterio.open(path, "w", **profile)


def open_mask_writer(path, grid):
    profile = geotiff_profile(
        grid["height"],
        grid["width"],
        grid["crs"],
        grid["transform"],
        dtype="uint8",
        nodata=255,
    )
    profile["NUM_THREADS"] = "ALL_CPUS"
    return rasterio.open(path, "w", **profile)


def export_annual_storage_maps(cfg: ProjectConfig):
    sec = cfg.section("storage")
    block_size = int(sec.get("annual_map_block_size", 512))
    threshold_mm = float(sec.get("diagnostic_threshold_mm", 1.0))

    insar_path = cfg.outputs / "canonical" / "insar_stack.h5"
    head_path = cfg.outputs / "groundwater" / "groundwater_field.h5"
    ske_path = cfg.outputs / "seasonal" / "ske_pixelwise.tif"
    domain_path = cfg.outputs / "storage" / "storage_domain_mask.tif"

    out_dir = ensure_dir(
        cfg.outputs / "storage" / "annual_maps"
    )
    grid = h5_grid_metadata(insar_path)

    ske = read_tif(ske_path)
    domain_all = read_tif(domain_path) > 0

    if ske.shape != domain_all.shape:
        raise ValueError("Ske and storage-domain grids do not match")

    with h5py.File(insar_path, "r") as ih5:
        i_dates_all = days_to_dates(ih5["date_days"][:])

    with h5py.File(head_path, "r") as hh5:
        h_dates_all = days_to_dates(hh5["date_days"][:])

    dates, i_idx, h_idx = np.intersect1d(
        i_dates_all,
        h_dates_all,
        assume_unique=True,
        return_indices=True,
    )

    ib = nearest_index(
        dates,
        sec.get("baseline_date"),
        0,
    )
    ie = nearest_index(
        dates,
        sec.get("end_date"),
        len(dates) - 1,
    )

    final_start = dates[ib]
    final_end = dates[ie]

    period = float(
        sec.get(
            "annual_period_days",
            cfg.section("seasonal_response").get(
                "annual_period_days",
                365.2425,
            ),
        )
    )

    knots = calendar_year_knots(
        dates[0],
        dates[-1],
    )
    model = TimeModel(
        polynomial_degree=1,
        periods_days=(period,),
        polyline_knots=knots,
    )
    X, _ = design_matrix(
        dates,
        model,
        origin=dates[0],
    )
    min_obs = max(
        int(sec.get("min_observations", 24)),
        model.n_parameters + 2,
    )

    intervals = []
    for year in range(
        int(str(final_start)[:4]),
        int(str(final_end)[:4]) + 1,
    ):
        y0 = np.datetime64(f"{year}-01-01", "D")
        y1 = np.datetime64(f"{year+1}-01-01", "D")
        start = max(y0, final_start)
        end = min(y1, final_end)
        if end <= start:
            continue

        dx = (
            low_frequency_row(end, model, dates[0])
            - low_frequency_row(start, model, dates[0])
        )
        intervals.append(
            {
                "year": year,
                "start": start,
                "end": end,
                "complete": bool(
                    start == y0 and end == y1
                ),
                "dx": dx,
            }
        )

    # One set of writers for every interval.
    writers = {}
    for item in intervals:
        year = item["year"]
        writers[year] = {
            "total": open_float_writer(
                out_dir / f"{year}_total_change_mm.tif",
                grid,
            ),
            "recoverable": open_float_writer(
                out_dir / f"{year}_recoverable_change_mm.tif",
                grid,
            ),
            "irreversible": open_float_writer(
                out_dir / f"{year}_irreversible_change_mm.tif",
                grid,
            ),
            "head": open_float_writer(
                out_dir / f"{year}_head_lowfreq_change_m.tif",
                grid,
            ),
            "recovery_compaction": open_mask_writer(
                out_dir
                / f"{year}_recovery_with_continued_compaction.tif",
                grid,
            ),
        }

    # Accumulators for area/volume statistics.
    stats = {
        item["year"]: {
            "total_m3": 0.0,
            "recoverable_m3": 0.0,
            "irreversible_m3": 0.0,
            "gross_negative_irreversible_m3": 0.0,
            "domain_area_m2": 0.0,
            "recovery_compaction_area_m2": 0.0,
            "total_negative_area_m2": 0.0,
            "recoverable_positive_area_m2": 0.0,
            "irreversible_negative_area_m2": 0.0,
            # Approximate distribution summaries via deterministic sample.
            "sample_total": [],
            "sample_rec": [],
            "sample_irr": [],
            "sample_head": [],
        }
        for item in intervals
    }

    area_rows = pixel_area_rows(
        grid["height"],
        grid["width"],
        grid["crs"],
        grid["transform"],
    )

    blocks = list(
        block_slices(
            grid["height"],
            grid["width"],
            int(block_size),
        )
    )
    total_blocks = len(blocks)
    t0 = time.perf_counter()
    pattern_d = 0
    pattern_h = 0

    print("=" * 80, flush=True)
    print("ANNUAL PIXELWISE STORAGE MAPS", flush=True)
    print("=" * 80, flush=True)
    print(f"Interval      : {final_start} -> {final_end}", flush=True)
    print(f"Common epochs : {len(dates)}", flush=True)
    print(f"Model params  : {model.n_parameters}", flush=True)
    print(f"Block size    : {block_size}", flush=True)
    print(f"Blocks        : {total_blocks}", flush=True)
    print(
        f"Diagnostic threshold: {threshold_mm:g} mm",
        flush=True,
    )

    try:
        with h5py.File(insar_path, "r") as ih5, h5py.File(
            head_path,
            "r",
        ) as hh5:
            ids = ih5["displacement_mm"]
            hds = hh5["head_anomaly_m"]

            for jb, (r0, r1, c0, c1) in enumerate(
                blocks,
                start=1,
            ):
                domain_b = domain_all[
                    r0:r1,
                    c0:c1,
                ]
                if not domain_b.any():
                    # Write nodata for every annual product.
                    bh, bw = domain_b.shape
                    win = rasterio.windows.Window(
                        c0, r0, bw, bh
                    )
                    nan_block = np.full(
                        (bh, bw),
                        np.nan,
                        dtype="float32",
                    )
                    mask_block = np.full(
                        (bh, bw),
                        255,
                        dtype="uint8",
                    )
                    for item in intervals:
                        ww = writers[item["year"]]
                        ww["total"].write(
                            nan_block, 1, window=win
                        )
                        ww["recoverable"].write(
                            nan_block, 1, window=win
                        )
                        ww["irreversible"].write(
                            nan_block, 1, window=win
                        )
                        ww["head"].write(
                            nan_block, 1, window=win
                        )
                        ww["recovery_compaction"].write(
                            mask_block, 1, window=win
                        )
                    continue

                d = ids[
                    i_idx,
                    r0:r1,
                    c0:c1,
                ].astype("float32")
                h = hds[
                    h_idx,
                    r0:r1,
                    c0:c1,
                ].astype("float32")

                T, bh, bw = d.shape
                dflat = d.reshape(T, -1)
                hflat = h.reshape(T, -1)

                dbeta, _drmse, _dn, _drss, ndp = fit_block_grouped(
                    dflat,
                    X,
                    min_obs,
                )
                hbeta, _hrmse, _hn, _hrss, nhp = fit_block_grouped(
                    hflat,
                    X,
                    min_obs,
                )
                pattern_d += ndp
                pattern_h += nhp

                domain = domain_b.ravel()
                ske_b = ske[
                    r0:r1,
                    c0:c1,
                ].ravel()

                area_b = np.broadcast_to(
                    area_rows[r0:r1, None],
                    (bh, bw),
                ).ravel()

                win = rasterio.windows.Window(
                    c0,
                    r0,
                    bw,
                    bh,
                )

                for item in intervals:
                    year = item["year"]
                    dx = item["dx"]

                    total_mm = dbeta @ dx
                    head_m = hbeta @ dx
                    recoverable_mm = (
                        ske_b * head_m * 1000.0
                    )
                    irreversible_mm = (
                        total_mm - recoverable_mm
                    )

                    good = (
                        domain
                        & np.isfinite(total_mm)
                        & np.isfinite(recoverable_mm)
                        & np.isfinite(head_m)
                    )

                    total_out = np.full(
                        bh * bw,
                        np.nan,
                        dtype="float32",
                    )
                    rec_out = total_out.copy()
                    irr_out = total_out.copy()
                    head_out = total_out.copy()

                    total_out[good] = total_mm[
                        good
                    ].astype("float32")
                    rec_out[good] = recoverable_mm[
                        good
                    ].astype("float32")
                    irr_out[good] = irreversible_mm[
                        good
                    ].astype("float32")
                    head_out[good] = head_m[
                        good
                    ].astype("float32")

                    key_mask = (
                        good
                        & (
                            recoverable_mm
                            > float(threshold_mm)
                        )
                        & (
                            irreversible_mm
                            < -float(threshold_mm)
                        )
                    )

                    mask_out = np.full(
                        bh * bw,
                        255,
                        dtype="uint8",
                    )
                    mask_out[good] = 0
                    mask_out[key_mask] = 1

                    ww = writers[year]
                    ww["total"].write(
                        total_out.reshape(bh, bw),
                        1,
                        window=win,
                    )
                    ww["recoverable"].write(
                        rec_out.reshape(bh, bw),
                        1,
                        window=win,
                    )
                    ww["irreversible"].write(
                        irr_out.reshape(bh, bw),
                        1,
                        window=win,
                    )
                    ww["head"].write(
                        head_out.reshape(bh, bw),
                        1,
                        window=win,
                    )
                    ww["recovery_compaction"].write(
                        mask_out.reshape(bh, bw),
                        1,
                        window=win,
                    )

                    # Area-integrated statistics.
                    s = stats[year]
                    a = area_b[good]
                    tm = total_mm[good] / 1000.0
                    rm = recoverable_mm[good] / 1000.0
                    im = irreversible_mm[good] / 1000.0

                    s["total_m3"] += float(
                        np.sum(tm * a)
                    )
                    s["recoverable_m3"] += float(
                        np.sum(rm * a)
                    )
                    s["irreversible_m3"] += float(
                        np.sum(im * a)
                    )
                    s["gross_negative_irreversible_m3"] += float(
                        np.sum(
                            np.maximum(-im, 0.0) * a
                        )
                    )
                    s["domain_area_m2"] += float(
                        np.sum(a)
                    )
                    s["recovery_compaction_area_m2"] += float(
                        np.sum(area_b[key_mask])
                    )
                    s["total_negative_area_m2"] += float(
                        np.sum(
                            area_b[
                                good & (total_mm < 0)
                            ]
                        )
                    )
                    s["recoverable_positive_area_m2"] += float(
                        np.sum(
                            area_b[
                                good
                                & (recoverable_mm > 0)
                            ]
                        )
                    )
                    s["irreversible_negative_area_m2"] += float(
                        np.sum(
                            area_b[
                                good
                                & (irreversible_mm < 0)
                            ]
                        )
                    )

                    # Deterministic sparse sample for medians/quantiles.
                    # Keeps memory low and is only descriptive.
                    sample_idx = np.flatnonzero(good)[::200]
                    if sample_idx.size:
                        s["sample_total"].append(
                            total_mm[sample_idx]
                        )
                        s["sample_rec"].append(
                            recoverable_mm[sample_idx]
                        )
                        s["sample_irr"].append(
                            irreversible_mm[sample_idx]
                        )
                        s["sample_head"].append(
                            head_m[sample_idx]
                        )

                if (
                    jb == 1
                    or jb % max(1, total_blocks // 50) == 0
                    or jb == total_blocks
                ):
                    elapsed = time.perf_counter() - t0
                    eta = elapsed * (
                        total_blocks / jb - 1.0
                    )
                    print(
                        f"[ANNUAL] {jb:4d}/{total_blocks} "
                        f"({100*jb/total_blocks:5.1f}%) "
                        f"elapsed={elapsed/60:6.1f} min "
                        f"ETA={eta/60:6.1f} min "
                        f"patterns(d/h)={ndp}/{nhp}",
                        flush=True,
                    )
    finally:
        for per_year in writers.values():
            for dst in per_year.values():
                dst.close()

    rows = []
    for item in intervals:
        year = item["year"]
        s = stats[year]
        area = s["domain_area_m2"]

        def sampled(name):
            if not s[name]:
                return np.array([], dtype=float)
            return np.concatenate(s[name])

        t = sampled("sample_total")
        r = sampled("sample_rec")
        i = sampled("sample_irr")
        h = sampled("sample_head")

        rows.append(
            {
                "year": year,
                "start_date": str(item["start"]),
                "end_date": str(item["end"]),
                "complete_calendar_year": item["complete"],
                "total_gws_change_m3": s["total_m3"],
                "recoverable_gws_change_m3": s["recoverable_m3"],
                "irreversible_gws_change_m3": s["irreversible_m3"],
                "gross_negative_irreversible_change_m3":
                    s["gross_negative_irreversible_m3"],
                "domain_area_km2": area / 1e6,
                "fraction_area_total_negative":
                    s["total_negative_area_m2"] / area
                    if area else np.nan,
                "fraction_area_recoverable_positive":
                    s["recoverable_positive_area_m2"] / area
                    if area else np.nan,
                "fraction_area_irreversible_negative":
                    s["irreversible_negative_area_m2"] / area
                    if area else np.nan,
                "recovery_with_continued_compaction_area_km2":
                    s["recovery_compaction_area_m2"] / 1e6,
                "fraction_area_recovery_with_continued_compaction":
                    s["recovery_compaction_area_m2"] / area
                    if area else np.nan,
                "sample_median_total_mm":
                    float(np.nanmedian(t)) if t.size else np.nan,
                "sample_median_recoverable_mm":
                    float(np.nanmedian(r)) if r.size else np.nan,
                "sample_median_irreversible_mm":
                    float(np.nanmedian(i)) if i.size else np.nan,
                "sample_median_head_change_m":
                    float(np.nanmedian(h)) if h.size else np.nan,
                "diagnostic_threshold_mm":
                    float(threshold_mm),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = (
        cfg.outputs
        / "storage"
        / "annual_maps_summary.csv"
    )
    summary.to_csv(
        summary_path,
        index=False,
    )

    # Check against the already completed regional annual integration.
    old_path = (
        cfg.outputs
        / "storage"
        / "storage_annual_change.csv"
    )
    if old_path.exists():
        old = pd.read_csv(old_path)
        cols = [
            "year",
            "total_gws_change_m3",
            "recoverable_gws_change_m3",
            "irreversible_gws_change_m3",
        ]
        check = summary[cols].merge(
            old[cols],
            on="year",
            suffixes=("_maps", "_previous"),
        )
        for v in [
            "total_gws_change_m3",
            "recoverable_gws_change_m3",
            "irreversible_gws_change_m3",
        ]:
            check[f"{v}_difference"] = (
                check[f"{v}_maps"]
                - check[f"{v}_previous"]
            )
        check.to_csv(
            cfg.outputs
            / "storage"
            / "annual_maps_volume_check.csv",
            index=False,
        )

    print()
    print("=" * 80)
    print("ANNUAL MAP SUMMARY")
    print("=" * 80)
    print(
        summary[
            [
                "year",
                "start_date",
                "end_date",
                "total_gws_change_m3",
                "recoverable_gws_change_m3",
                "irreversible_gws_change_m3",
                "fraction_area_recovery_with_continued_compaction",
            ]
        ].to_string(index=False)
    )
    print()
    print("Annual maps:", out_dir)
    print("Summary    :", summary_path)
    return {
        "status": "ok",
        "output_directory": str(out_dir),
        "summary": str(summary_path),
        "years": [int(v) for v in summary["year"].tolist()],
        "diagnostic_threshold_mm": float(threshold_mm),
    }
