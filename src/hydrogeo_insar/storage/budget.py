from __future__ import annotations
import json, time
import h5py
import numpy as np
import pandas as pd
import rasterio

from ..common import (
    block_slices, days_to_dates, ensure_dir, geotiff_profile,
    h5_grid_metadata, pixel_area_rows, read_tif, write_json, write_tif
)
from ..config import ProjectConfig
from ..temporal.fit import fit_block_grouped
from ..temporal.model import (
    TimeModel, calendar_year_knots, design_matrix, low_frequency_row
)

def nearest_index(dates, target, default):
    if target is None:
        return default
    t = np.datetime64(str(target), "D")
    return int(np.argmin(np.abs(dates.astype("datetime64[D]") - t)))


def compute_storage_budget(cfg: ProjectConfig):
    sec = cfg.section("storage")
    ske_choice = str(sec.get("ske_product", "pixelwise")).lower()
    block_size = int(sec.get("block_size", 256))
    insar_path = cfg.outputs / "canonical" / "insar_stack.h5"
    head_path = cfg.outputs / "groundwater" / "groundwater_field.h5"
    seasonal = cfg.outputs / "seasonal"
    out_dir = ensure_dir(cfg.outputs / "storage")

    if ske_choice == "pixelwise":
        ske_path = seasonal / "ske_pixelwise.tif"
        mask_path = seasonal / "ske_pixelwise_support_mask.tif"
        label = "pixelwise"
    elif ske_choice in {"high-confidence", "high_confidence"}:
        ske_path = seasonal / "ske_pixelwise_high_confidence.tif"
        mask_path = seasonal / "ske_pixelwise_high_confidence_mask.tif"
        label = "pixelwise_high_confidence"
    else:
        raise ValueError("storage.ske_product must be pixelwise or high_confidence")

    grid = h5_grid_metadata(insar_path)
    ske = read_tif(ske_path)
    ske_support = read_tif(mask_path) > 0
    gw_support = read_tif(
        cfg.outputs / "groundwater" / "groundwater_support_mask.tif"
    ) > 0

    with h5py.File(insar_path, "r") as ih5:
        i_dates_all = days_to_dates(ih5["date_days"][:])
    with h5py.File(head_path, "r") as hh5:
        h_dates_all = days_to_dates(hh5["date_days"][:])

    dates, i_idx, h_idx = np.intersect1d(
        i_dates_all, h_dates_all, assume_unique=True, return_indices=True
    )
    if len(dates) < 2:
        raise ValueError("No common dates for storage budget")

    ib = nearest_index(dates, sec.get("baseline_date"), 0)
    ie = nearest_index(dates, sec.get("end_date"), len(dates)-1)
    if ie < ib:
        raise ValueError("storage.end_date precedes storage.baseline_date")

    period = float(sec.get(
        "annual_period_days",
        cfg.section("seasonal_response").get("annual_period_days", 365.2425)
    ))
    knots = calendar_year_knots(dates[0], dates[-1])
    model = TimeModel(
        polynomial_degree=1,
        periods_days=(period,),
        polyline_knots=knots
    )
    X, _ = design_matrix(dates, model, origin=dates[0])
    min_obs = max(int(sec.get("min_observations", 24)), model.n_parameters + 2)

    final_start, final_end = dates[ib], dates[ie]
    dx_final = (
        low_frequency_row(final_end, model, dates[0])
        - low_frequency_row(final_start, model, dates[0])
    )

    intervals = []
    for year in range(int(str(final_start)[:4]), int(str(final_end)[:4]) + 1):
        y0 = np.datetime64(f"{year}-01-01", "D")
        y1 = np.datetime64(f"{year+1}-01-01", "D")
        start, end = max(y0, final_start), min(y1, final_end)
        if end > start:
            dx = (
                low_frequency_row(end, model, dates[0])
                - low_frequency_row(start, model, dates[0])
            )
            intervals.append((year, start, end, bool(start == y0 and end == y1), dx))

    valid_fraction = float(sec.get("valid_fraction", 0.90))
    min_valid = int(np.ceil(valid_fraction * len(dates)))
    area_rows = pixel_area_rows(
        grid["height"], grid["width"], grid["crs"], grid["transform"]
    )

    profile = geotiff_profile(
        grid["height"], grid["width"], grid["crs"], grid["transform"],
        dtype="float32", nodata=np.nan
    )
    profile["NUM_THREADS"] = "ALL_CPUS"
    mask_profile = profile.copy()
    mask_profile.update(dtype="uint8", nodata=0)

    paths = {
        "domain": out_dir / "storage_domain_mask.tif",
        "total": out_dir / "total_gws_change_equivalent_mm.tif",
        "recoverable": out_dir / "recoverable_gws_change_equivalent_mm.tif",
        "irreversible": out_dir / "irreversible_gws_change_equivalent_mm.tif",
        "head": out_dir / "head_lowfreq_change_m.tif",
    }
    writers = {
        "domain": rasterio.open(paths["domain"], "w", **mask_profile),
        "total": rasterio.open(paths["total"], "w", **profile),
        "recoverable": rasterio.open(paths["recoverable"], "w", **profile),
        "irreversible": rasterio.open(paths["irreversible"], "w", **profile),
        "head": rasterio.open(paths["head"], "w", **profile),
    }

    blocks = list(block_slices(grid["height"], grid["width"], block_size))
    total_blocks = len(blocks)
    final_totals = np.zeros(4, float)
    annual_acc = {year: np.zeros(4, float) for year, *_ in intervals}
    obs_vt = np.zeros(ie - ib + 1, float)
    obs_vr = np.zeros(ie - ib + 1, float)
    domain_pixels = 0
    pat_d = pat_h = 0
    t0 = time.perf_counter()

    print("="*80, flush=True)
    print("PIXELWISE STORAGE BUDGET", flush=True)
    print("="*80, flush=True)
    print(f"Ske       : {ske_path}", flush=True)
    print(f"Epochs    : {len(dates)}  {dates[0]} -> {dates[-1]}", flush=True)
    print(f"Interval  : {final_start} -> {final_end}", flush=True)
    print(f"Blocks    : {total_blocks}", flush=True)

    try:
        with h5py.File(insar_path, "r") as ih5, h5py.File(head_path, "r") as hh5:
            ids = ih5["displacement_mm"]
            hds = hh5["head_anomaly_m"]

            for jb, (r0, r1, c0, c1) in enumerate(blocks, 1):
                d = ids[i_idx, r0:r1, c0:c1].astype("float32")
                h = hds[h_idx, r0:r1, c0:c1].astype("float32")
                T, bh, bw = d.shape
                dflat, hflat = d.reshape(T, -1), h.reshape(T, -1)

                dbeta, _drmse, dn, _drss, ndp = fit_block_grouped(dflat, X, min_obs)
                hbeta, _hrmse, hn, _hrss, nhp = fit_block_grouped(hflat, X, min_obs)
                pat_d += ndp
                pat_h += nhp

                ske_b = ske[r0:r1, c0:c1].ravel()
                support_b = (
                    ske_support[r0:r1, c0:c1].ravel()
                    & gw_support[r0:r1, c0:c1].ravel()
                )
                domain = (
                    support_b & np.isfinite(ske_b)
                    & (dn >= min_valid) & (hn >= min_valid)
                )
                domain_pixels += int(domain.sum())

                area_b = np.broadcast_to(
                    area_rows[r0:r1, None], (bh, bw)
                ).ravel()

                dd_mm = dbeta @ dx_final
                dh_m = hbeta @ dx_final
                dt_m = dd_mm / 1000.0
                dr_m = ske_b * dh_m
                di_m = dt_m - dr_m
                good = domain & np.isfinite(dt_m) & np.isfinite(dr_m)

                final_totals[0] += np.sum(dt_m[good] * area_b[good])
                final_totals[1] += np.sum(dr_m[good] * area_b[good])
                final_totals[2] += np.sum(di_m[good] * area_b[good])
                final_totals[3] += np.sum(
                    np.maximum(-di_m[good], 0.0) * area_b[good]
                )

                out_total = np.full(bh*bw, np.nan, dtype="float32")
                out_rec = out_total.copy()
                out_irr = out_total.copy()
                out_head = out_total.copy()
                out_total[good] = (dt_m[good]*1000.0).astype("float32")
                out_rec[good] = (dr_m[good]*1000.0).astype("float32")
                out_irr[good] = (di_m[good]*1000.0).astype("float32")
                out_head[good] = dh_m[good].astype("float32")

                win = rasterio.windows.Window(c0, r0, bw, bh)
                writers["domain"].write(domain.reshape(bh,bw).astype("uint8"), 1, window=win)
                writers["total"].write(out_total.reshape(bh,bw), 1, window=win)
                writers["recoverable"].write(out_rec.reshape(bh,bw), 1, window=win)
                writers["irreversible"].write(out_irr.reshape(bh,bw), 1, window=win)
                writers["head"].write(out_head.reshape(bh,bw), 1, window=win)

                for year, _s, _e, _complete, dx in intervals:
                    ddy_mm = dbeta @ dx
                    dhy_m = hbeta @ dx
                    dty_m = ddy_mm / 1000.0
                    dry_m = ske_b * dhy_m
                    diy_m = dty_m - dry_m
                    ok = domain & np.isfinite(dty_m) & np.isfinite(dry_m)
                    vt = float(np.sum(dty_m[ok] * area_b[ok]))
                    vr = float(np.sum(dry_m[ok] * area_b[ok]))
                    vi = vt - vr
                    gross = float(np.sum(np.maximum(-diy_m[ok], 0.0) * area_b[ok]))
                    annual_acc[year] += [vt, vr, vi, gross]

                db = (dflat[ib:ie+1] - dflat[ib][None, :]) / 1000.0
                hb = hflat[ib:ie+1] - hflat[ib][None, :]
                rec = hb * ske_b[None, :]
                good_ts = domain[None, :] & np.isfinite(db) & np.isfinite(rec)
                weights = area_b[None, :]
                obs_vt += np.sum(np.where(good_ts, db*weights, 0.0), axis=1)
                obs_vr += np.sum(np.where(good_ts, rec*weights, 0.0), axis=1)

                if jb == 1 or jb % max(1, total_blocks//50) == 0 or jb == total_blocks:
                    elapsed = time.perf_counter() - t0
                    eta = elapsed * (total_blocks/jb - 1.0)
                    print(
                        f"[STORAGE] {jb:4d}/{total_blocks} "
                        f"({100*jb/total_blocks:5.1f}%) "
                        f"elapsed={elapsed/60:6.1f} min "
                        f"ETA={eta/60:6.1f} min "
                        f"domain={domain_pixels:,} "
                        f"patterns(d/h)={ndp}/{nhp}",
                        flush=True
                    )
    finally:
        for dst in writers.values():
            dst.close()

    irr = read_tif(paths["irreversible"])
    write_tif(
        out_dir / "negative_irreversible_change_magnitude_mm.tif",
        np.where(np.isfinite(irr) & (irr < 0), -irr, np.nan).astype("float32"),
        grid["crs"], grid["transform"]
    )

    pd.DataFrame({
        "date": dates[ib:ie+1].astype(str),
        "total_gws_change_m3": obs_vt,
        "recoverable_gws_change_m3": obs_vr,
        "irreversible_gws_change_m3": obs_vt - obs_vr,
    }).to_csv(out_dir / "storage_cumulative_observed.csv", index=False)

    interval_map = {year:(s,e,c) for year,s,e,c,_ in intervals}
    rows = []
    for year, vals in annual_acc.items():
        s,e,c = interval_map[year]
        rows.append({
            "year": year,
            "start_date": str(s),
            "end_date": str(e),
            "complete_calendar_year": c,
            "total_gws_change_m3": vals[0],
            "recoverable_gws_change_m3": vals[1],
            "irreversible_gws_change_m3": vals[2],
            "net_irreversible_loss_magnitude_m3": max(0.0, -vals[2]),
            "gross_negative_irreversible_change_m3": vals[3],
        })
    pd.DataFrame(rows).to_csv(out_dir / "storage_annual_change.csv", index=False)

    summary = {
        "status": "ok",
        "ske_product": label,
        "ske_path": str(ske_path),
        "partition_model": "jiang2018",
        "annual_low_frequency_model": "continuous_piecewise_linear_plus_annual_harmonic",
        "baseline_date": str(final_start),
        "end_date": str(final_end),
        "common_epochs": int(len(dates)),
        "storage_domain_pixels": int(domain_pixels),
        "valid_fraction": valid_fraction,
        "total_gws_change_m3": float(final_totals[0]),
        "recoverable_gws_change_m3": float(final_totals[1]),
        "irreversible_gws_change_m3": float(final_totals[2]),
        "net_irreversible_loss_magnitude_m3": float(max(0.0, -final_totals[2])),
        "gross_negative_irreversible_change_m3": float(final_totals[3]),
        "identity": "V_total = V_recoverable + V_irreversible",
        "polyline_knots": [str(x) for x in knots],
        "block_size": int(block_size),
        "elapsed_seconds": float(time.perf_counter() - t0),
        "mean_deformation_mask_patterns_per_block": float(pat_d/total_blocks),
        "mean_head_mask_patterns_per_block": float(pat_h/total_blocks),
    }
    write_json(out_dir / "storage_budget_summary.json", summary)
    return summary
