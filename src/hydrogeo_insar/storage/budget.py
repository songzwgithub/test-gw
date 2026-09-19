from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import pandas as pd
import rasterio

from ..common import block_slices, days_to_dates, ensure_dir, geotiff_profile, h5_grid_metadata, pixel_area_rows, read_tif, write_json, write_tif
from ..config import ProjectConfig
from ..temporal.fit import fit_block
from ..temporal.model import TimeModel, calendar_year_knots, design_matrix, low_frequency_row


def _nearest_index(dates: np.ndarray, target: str | np.datetime64 | None, default: int) -> int:
    if target is None:
        return default
    t = np.datetime64(str(target), "D")
    return int(np.argmin(np.abs(dates.astype("datetime64[D]") - t)))


def _storage_domain(cfg: ProjectConfig, dates, i_idx, h_idx, grid):
    gw_support = read_tif(cfg.outputs / "groundwater" / "groundwater_support_mask.tif") > 0
    ske_support = read_tif(cfg.outputs / "seasonal" / "ske_support_mask.tif") > 0
    ske = read_tif(cfg.outputs / "seasonal" / "ske_effective.tif")
    valid_fraction = float(cfg.section("storage").get("valid_fraction", 0.9))
    iv = np.zeros((grid["height"], grid["width"]), dtype=np.int32)
    hv = np.zeros_like(iv)
    with h5py.File(cfg.outputs / "canonical" / "insar_stack.h5", "r") as ih5, h5py.File(cfg.outputs / "groundwater" / "groundwater_field.h5", "r") as hh5:
        for ii, hh in zip(i_idx, h_idx):
            iv += np.isfinite(ih5["displacement_mm"][ii])
            hv += np.isfinite(hh5["head_anomaly_m"][hh])
    domain = gw_support & ske_support & np.isfinite(ske)
    domain &= iv >= int(np.ceil(valid_fraction * len(dates)))
    domain &= hv >= int(np.ceil(valid_fraction * len(dates)))
    return domain, ske


def _raw_cumulative_series(cfg, dates, i_idx, h_idx, domain, ske, area2, ib, clusters):
    rows, cluster_rows = [], []
    unique_clusters = [] if clusters is None else sorted(int(v) for v in np.unique(clusters[domain]) if v > 0)
    with h5py.File(cfg.outputs / "canonical" / "insar_stack.h5", "r") as ih5, h5py.File(cfg.outputs / "groundwater" / "groundwater_field.h5", "r") as hh5:
        disp0 = ih5["displacement_mm"][i_idx[ib]].astype(float)
        head0 = hh5["head_anomaly_m"][h_idx[ib]].astype(float)
        for k, date in enumerate(dates):
            disp = ih5["displacement_mm"][i_idx[k]].astype(float)
            head = hh5["head_anomaly_m"][h_idx[k]].astype(float)
            dtot = (disp - disp0) / 1000.0
            drec = ske * (head - head0)
            m = domain & np.isfinite(dtot) & np.isfinite(drec)
            vt = float(np.sum(dtot[m] * area2[m]))
            vr = float(np.sum(drec[m] * area2[m]))
            vi = vt - vr
            rows.append({"date": str(date), "total_gws_change_m3": vt, "recoverable_gws_change_m3": vr, "irreversible_gws_change_m3": vi})
            for cid in unique_clusters:
                cm = m & (clusters == cid)
                cvt = float(np.sum(dtot[cm] * area2[cm])); cvr = float(np.sum(drec[cm] * area2[cm]))
                cluster_rows.append({"date": str(date), "cluster_id": cid, "total_gws_change_m3": cvt, "recoverable_gws_change_m3": cvr, "irreversible_gws_change_m3": cvt - cvr})
    return pd.DataFrame(rows), pd.DataFrame(cluster_rows)


def compute_storage_budget(cfg: ProjectConfig) -> dict[str, Any]:
    """Continuous-field Jiang-style TGWS/RGWS/IGWS partition.

    Cumulative observed series uses common InSAR/head epochs. Long-term maps
    and annual increments use a jointly fitted continuous piecewise-linear +
    annual-harmonic model, so annual changes are not forced by one quadratic
    trend over the full study period.
    """
    sec = cfg.section("storage")
    if str(sec.get("partition_model", "jiang2018")).lower() != "jiang2018":
        raise ValueError("v0.3 storage.partition_model supports 'jiang2018'")

    insar_path = cfg.outputs / "canonical" / "insar_stack.h5"
    head_path = cfg.outputs / "groundwater" / "groundwater_field.h5"
    out_dir = ensure_dir(cfg.outputs / "storage")
    grid = h5_grid_metadata(insar_path)

    with h5py.File(insar_path, "r") as ih5:
        if str(ih5.attrs.get("quantity", "vertical_displacement")) != "vertical_displacement":
            raise ValueError("Storage budget requires vertical_displacement")
        i_dates_all = days_to_dates(ih5["date_days"][:])
    with h5py.File(head_path, "r") as hh5:
        h_dates_all = days_to_dates(hh5["date_days"][:])
    dates, i_idx, h_idx = np.intersect1d(i_dates_all, h_dates_all, assume_unique=True, return_indices=True)
    if len(dates) < 2:
        raise ValueError("No common dates for storage budget")

    domain, ske = _storage_domain(cfg, dates, i_idx, h_idx, grid)
    if not domain.any():
        raise ValueError("Storage analysis domain is empty")
    write_tif(out_dir / "storage_domain_mask.tif", domain.astype("uint8"), grid["crs"], grid["transform"], nodata=0, dtype="uint8")

    clusters_path = cfg.outputs / "regimes" / "deformation_regime_id.tif"
    clusters = read_tif(clusters_path) if clusters_path.exists() else None
    if clusters is not None:
        clusters = np.where(np.isfinite(clusters), clusters, 0).astype(int)

    area = pixel_area_rows(grid["height"], grid["width"], grid["crs"], grid["transform"])[:, None]
    area2 = np.broadcast_to(area, domain.shape)
    ib = _nearest_index(dates, sec.get("baseline_date"), 0)
    ie = _nearest_index(dates, sec.get("end_date"), len(dates) - 1)
    if ie < ib:
        ib, ie = ie, ib

    observed_ts, observed_clusters = _raw_cumulative_series(cfg, dates, i_idx, h_idx, domain, ske, area2, ib, clusters)
    observed_ts.to_csv(out_dir / "storage_cumulative_observed.csv", index=False)
    if not observed_clusters.empty:
        observed_clusters.to_csv(out_dir / "storage_cumulative_observed_by_cluster.csv", index=False)

    period = float(sec.get("annual_period_days", cfg.section("seasonal_response").get("annual_period_days", 365.2425)))
    knots = calendar_year_knots(dates[0], dates[-1])
    model = TimeModel(polynomial_degree=1, periods_days=(period,), polyline_knots=knots)
    X, _ = design_matrix(dates, model, origin=dates[0])
    min_obs = max(int(sec.get("min_observations", 24)), model.n_parameters + 2)

    final_start = dates[ib]; final_end = dates[ie]
    x0 = low_frequency_row(final_start, model, dates[0])
    x1 = low_frequency_row(final_end, model, dates[0])
    dx_final = x1 - x0

    years = range(int(str(dates[0])[:4]), int(str(dates[-1])[:4]) + 1)
    intervals = []
    for year in years:
        y0 = np.datetime64(f"{year}-01-01", "D")
        y1 = np.datetime64(f"{year + 1}-01-01", "D")
        start = max(y0, dates[0]); end = min(y1, dates[-1])
        if end > start:
            intervals.append((year, start, end, bool(start == y0 and end == y1), low_frequency_row(end, model, dates[0]) - low_frequency_row(start, model, dates[0])))

    annual_acc = {year: np.zeros(4, dtype=float) for year, *_ in intervals}  # VT, VR, VI, gross negative VI
    cluster_ids = [] if clusters is None else sorted(int(v) for v in np.unique(clusters[domain]) if v > 0)
    cluster_acc = {(year, cid): np.zeros(3, dtype=float) for year, *_ in intervals for cid in cluster_ids}
    final_totals = np.zeros(4, dtype=float)

    profile = geotiff_profile(grid["height"], grid["width"], grid["crs"], grid["transform"], dtype="float32", nodata=np.nan)
    writers = {
        "total": rasterio.open(out_dir / "total_gws_change_equivalent_mm.tif", "w", **profile),
        "recoverable": rasterio.open(out_dir / "recoverable_gws_change_equivalent_mm.tif", "w", **profile),
        "irreversible": rasterio.open(out_dir / "irreversible_gws_change_equivalent_mm.tif", "w", **profile),
        "head": rasterio.open(out_dir / "head_lowfreq_change_m.tif", "w", **profile),
    }

    block_size = int(sec.get("block_size", 128))
    try:
        with h5py.File(insar_path, "r") as ih5, h5py.File(head_path, "r") as hh5:
            for r0, r1, c0, c1 in block_slices(grid["height"], grid["width"], block_size):
                d = ih5["displacement_mm"][i_idx, r0:r1, c0:c1].astype(float)
                h = hh5["head_anomaly_m"][h_idx, r0:r1, c0:c1].astype(float)
                T, bh, bw = d.shape
                dbeta, _drmse, _dn, _ = fit_block(d.reshape(T, -1), X, min_obs=min_obs)
                hbeta, _hrmse, _hn, _ = fit_block(h.reshape(T, -1), X, min_obs=min_obs)
                dm = domain[r0:r1, c0:c1].ravel()
                ske_b = ske[r0:r1, c0:c1].ravel()
                area_b = area2[r0:r1, c0:c1].ravel()

                dd_mm = dbeta @ dx_final
                dh_m = hbeta @ dx_final
                dt_m = dd_mm / 1000.0
                dr_m = ske_b * dh_m
                di_m = dt_m - dr_m
                good = dm & np.isfinite(dt_m) & np.isfinite(dr_m)
                final_totals[0] += np.sum(dt_m[good] * area_b[good])
                final_totals[1] += np.sum(dr_m[good] * area_b[good])
                final_totals[2] += np.sum(di_m[good] * area_b[good])
                final_totals[3] += np.sum(np.maximum(-di_m[good], 0.0) * area_b[good])

                out_total = np.full(bh * bw, np.nan); out_rec = out_total.copy(); out_irr = out_total.copy(); out_head = out_total.copy()
                out_total[good] = dt_m[good] * 1000.0
                out_rec[good] = dr_m[good] * 1000.0
                out_irr[good] = di_m[good] * 1000.0
                out_head[good] = dh_m[good]
                win = rasterio.windows.Window(c0, r0, bw, bh)
                writers["total"].write(out_total.reshape(bh, bw).astype("float32"), 1, window=win)
                writers["recoverable"].write(out_rec.reshape(bh, bw).astype("float32"), 1, window=win)
                writers["irreversible"].write(out_irr.reshape(bh, bw).astype("float32"), 1, window=win)
                writers["head"].write(out_head.reshape(bh, bw).astype("float32"), 1, window=win)

                cl_b = None if clusters is None else clusters[r0:r1, c0:c1].ravel()
                for year, _start, _end, _complete, dx in intervals:
                    ddy = dbeta @ dx
                    dhy = hbeta @ dx
                    dty = ddy / 1000.0
                    dry = ske_b * dhy
                    diy = dty - dry
                    ok = dm & np.isfinite(dty) & np.isfinite(dry)
                    vt = float(np.sum(dty[ok] * area_b[ok])); vr = float(np.sum(dry[ok] * area_b[ok])); vi = vt - vr
                    annual_acc[year] += [vt, vr, vi, float(np.sum(np.maximum(-diy[ok], 0.0) * area_b[ok]))]
                    if cl_b is not None:
                        for cid in cluster_ids:
                            cm = ok & (cl_b == cid)
                            cvt = float(np.sum(dty[cm] * area_b[cm])); cvr = float(np.sum(dry[cm] * area_b[cm]))
                            cluster_acc[(year, cid)] += [cvt, cvr, cvt - cvr]
    finally:
        for dst in writers.values():
            dst.close()

    irr_map = read_tif(out_dir / "irreversible_gws_change_equivalent_mm.tif")
    write_tif(
        out_dir / "negative_irreversible_change_magnitude_mm.tif",
        np.where(np.isfinite(irr_map) & (irr_map < 0), -irr_map, np.nan).astype("float32"),
        grid["crs"], grid["transform"],
    )

    annual_rows = []
    interval_map = {year: (start, end, complete) for year, start, end, complete, _ in intervals}
    for year, vals in annual_acc.items():
        start, end, complete = interval_map[year]
        annual_rows.append({
            "year": year, "start_date": str(start), "end_date": str(end), "complete_calendar_year": complete,
            "total_gws_change_m3": vals[0], "recoverable_gws_change_m3": vals[1], "irreversible_gws_change_m3": vals[2],
            "net_irreversible_loss_magnitude_m3": max(0.0, -vals[2]), "gross_negative_irreversible_change_m3": vals[3],
        })
    pd.DataFrame(annual_rows).to_csv(out_dir / "storage_annual_change.csv", index=False)

    if cluster_ids:
        rows = []
        for (year, cid), vals in cluster_acc.items():
            rows.append({"year": year, "cluster_id": cid, "total_gws_change_m3": vals[0], "recoverable_gws_change_m3": vals[1], "irreversible_gws_change_m3": vals[2]})
        pd.DataFrame(rows).to_csv(out_dir / "storage_annual_change_by_cluster.csv", index=False)

    result = {
        "status": "ok",
        "partition_model": "jiang2018",
        "annual_low_frequency_model": "continuous_piecewise_linear_plus_annual_harmonic",
        "baseline_date": str(final_start),
        "end_date": str(final_end),
        "storage_domain_pixels": int(domain.sum()),
        "total_gws_change_m3": float(final_totals[0]),
        "recoverable_gws_change_m3": float(final_totals[1]),
        "irreversible_gws_change_m3": float(final_totals[2]),
        "net_irreversible_loss_magnitude_m3": float(max(0.0, -final_totals[2])),
        "gross_negative_irreversible_change_m3": float(final_totals[3]),
        "identity": "V_total = V_recoverable + V_irreversible",
        "polyline_knots": list(knots),
    }
    write_json(out_dir / "storage_budget_summary.json", result)
    return result
