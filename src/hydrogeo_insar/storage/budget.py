from __future__ import annotations

import json
from typing import Any

import h5py
import numpy as np
import pandas as pd
import rasterio

from ..common import days_to_dates, ensure_dir, h5_grid_metadata, pixel_area_rows, read_tif, write_json, write_tif
from ..config import ProjectConfig


def _nearest_index(dates: np.ndarray, target: str | np.datetime64 | None, default: int) -> int:
    if target is None:
        return default
    t = np.datetime64(str(target), "D")
    return int(np.argmin(np.abs(dates.astype("datetime64[D]") - t)))


def _evaluate_trend(intercept, linear, quadratic, t_year):
    out = intercept + linear * t_year
    if quadratic is not None:
        out = out + quadratic * t_year**2
    return out


def compute_storage_budget(cfg: ProjectConfig) -> dict[str, Any]:
    """Jiang-style storage partition on a fixed spatial domain.

    Signed convention:
      positive deformation = uplift;
      positive head change = head recovery;
      positive storage change = storage gain;
      V_total = V_recoverable + V_irreversible.
    """
    sec = cfg.section("storage")
    insar_path = cfg.outputs / "canonical" / "insar_stack.h5"
    head_path = cfg.outputs / "groundwater" / "groundwater_field.h5"
    ske_path = cfg.outputs / "seasonal" / "ske_effective.tif"
    out_dir = ensure_dir(cfg.outputs / "storage")
    grid = h5_grid_metadata(insar_path)

    with h5py.File(insar_path, "r") as ih5:
        quantity = str(ih5.attrs.get("quantity", "vertical_displacement"))
        if quantity != "vertical_displacement":
            raise ValueError("Storage budget requires InSAR quantity='vertical_displacement'")
        i_dates_all = days_to_dates(ih5["date_days"][:])
    with h5py.File(head_path, "r") as hh5:
        h_dates_all = days_to_dates(hh5["date_days"][:])
    dates, i_idx, h_idx = np.intersect1d(i_dates_all, h_dates_all, assume_unique=True, return_indices=True)
    if len(dates) < 2:
        raise ValueError("No usable common dates for storage budget")

    ske = read_tif(ske_path)
    gw_support = read_tif(cfg.outputs / "groundwater" / "groundwater_support_mask.tif") > 0
    ske_support = read_tif(cfg.outputs / "seasonal" / "ske_support_mask.tif") > 0
    cluster_path = cfg.outputs / "regimes" / "deformation_regime_id.tif"
    clusters = read_tif(cluster_path) if cluster_path.exists() else None
    if clusters is not None:
        clusters = np.where(np.isfinite(clusters), clusters, 0).astype(int)

    valid_fraction = float(sec.get("valid_fraction", 0.9))
    iv = np.zeros((grid["height"], grid["width"]), dtype=np.int32)
    hv = np.zeros_like(iv)
    with h5py.File(insar_path, "r") as ih5, h5py.File(head_path, "r") as hh5:
        for ii, hh in zip(i_idx, h_idx):
            iv += np.isfinite(ih5["displacement_mm"][ii])
            hv += np.isfinite(hh5["head_anomaly_m"][hh])
    domain = gw_support & ske_support & np.isfinite(ske)
    domain &= iv >= int(np.ceil(valid_fraction * len(dates)))
    domain &= hv >= int(np.ceil(valid_fraction * len(dates)))
    if domain.sum() == 0:
        raise ValueError("Storage analysis domain is empty")

    area = pixel_area_rows(grid["height"], grid["width"], grid["crs"], grid["transform"])[:, None]
    ib = _nearest_index(dates, sec.get("baseline_date"), 0)
    ie = _nearest_index(dates, sec.get("end_date"), len(dates)-1)
    rows = []
    cluster_rows = []
    unique_clusters = [] if clusters is None else sorted(int(v) for v in np.unique(clusters[domain]) if v > 0)

    with h5py.File(insar_path, "r") as ih5, h5py.File(head_path, "r") as hh5:
        disp0 = ih5["displacement_mm"][i_idx[ib]].astype(float)
        head0 = hh5["head_anomaly_m"][h_idx[ib]].astype(float)
        for k, date in enumerate(dates):
            disp = ih5["displacement_mm"][i_idx[k]].astype(float)
            head = hh5["head_anomaly_m"][h_idx[k]].astype(float)
            dtot = (disp - disp0) / 1000.0
            dhead = head - head0
            drec = ske * dhead
            dirr = dtot - drec
            m = domain & np.isfinite(dtot) & np.isfinite(drec)
            vt = float(np.sum(dtot[m] * np.broadcast_to(area, dtot.shape)[m]))
            vr = float(np.sum(drec[m] * np.broadcast_to(area, drec.shape)[m]))
            vi = float(vt - vr)
            rows.append({
                "date": str(date),
                "total_gws_change_m3": vt,
                "recoverable_gws_change_m3": vr,
                "irreversible_gws_change_m3": vi,
                "irreversible_storage_loss_magnitude_m3": max(0.0, -vi),
            })
            for cid in unique_clusters:
                cm = m & (clusters == cid)
                cvt = float(np.sum(dtot[cm] * np.broadcast_to(area, dtot.shape)[cm]))
                cvr = float(np.sum(drec[cm] * np.broadcast_to(area, drec.shape)[cm]))
                cvi = cvt - cvr
                cluster_rows.append({
                    "date": str(date), "cluster_id": cid,
                    "total_gws_change_m3": cvt,
                    "recoverable_gws_change_m3": cvr,
                    "irreversible_gws_change_m3": cvi,
                    "irreversible_storage_loss_magnitude_m3": max(0.0, -cvi),
                })

        disp_end = ih5["displacement_mm"][i_idx[ie]].astype(float)
        head_end = hh5["head_anomaly_m"][h_idx[ie]].astype(float)
    total_mm = disp_end - disp0
    recoverable_mm = ske * (head_end - head0) * 1000.0
    irreversible_mm = total_mm - recoverable_mm
    for arr in (total_mm, recoverable_mm, irreversible_mm):
        arr[~domain] = np.nan
    write_tif(out_dir / "total_gws_change_equivalent_mm.tif", total_mm.astype("float32"), grid["crs"], grid["transform"])
    write_tif(out_dir / "recoverable_gws_change_equivalent_mm.tif", recoverable_mm.astype("float32"), grid["crs"], grid["transform"])
    write_tif(out_dir / "irreversible_gws_change_equivalent_mm.tif", irreversible_mm.astype("float32"), grid["crs"], grid["transform"])
    write_tif(out_dir / "negative_irreversible_change_magnitude_mm.tif", np.where(domain & (irreversible_mm < 0), -irreversible_mm, np.nan).astype("float32"), grid["crs"], grid["transform"])
    write_tif(out_dir / "storage_domain_mask.tif", domain.astype("uint8"), grid["crs"], grid["transform"], nodata=0, dtype="uint8")

    ts = pd.DataFrame(rows)
    ts.to_csv(out_dir / "storage_cumulative_timeseries.csv", index=False)
    if cluster_rows:
        pd.DataFrame(cluster_rows).to_csv(out_dir / "storage_cumulative_by_cluster.csv", index=False)

    # Annual changes from fitted low-frequency components, evaluated at fixed calendar boundaries.
    dsum = json.loads((cfg.outputs / "deformation" / "deformation_summary.json").read_text(encoding="utf-8"))
    hsum = json.loads((cfg.outputs / "seasonal" / "joint_harmonics_summary.json").read_text(encoding="utf-8"))
    d_origin = np.datetime64(dsum["first_date"], "D")
    h_origin = np.datetime64(hsum["first_date"], "D")
    d_intercept = read_tif(cfg.outputs / "deformation" / "intercept_mm.tif")
    d_linear = read_tif(cfg.outputs / "deformation" / "linear_coeff_mm_yr.tif")
    d_quad_path = cfg.outputs / "deformation" / "quadratic_coeff_mm_yr2.tif"
    d_quad = read_tif(d_quad_path) if d_quad_path.exists() else None
    h_intercept = read_tif(cfg.outputs / "seasonal" / "head_intercept_m.tif")
    h_linear = read_tif(cfg.outputs / "seasonal" / "head_linear_m_yr.tif")
    h_quad_path = cfg.outputs / "seasonal" / "head_quadratic_m_yr2.tif"
    h_quad = read_tif(h_quad_path) if h_quad_path.exists() else None

    years = range(pd.Timestamp(str(dates[0])).year, pd.Timestamp(str(dates[-1])).year + 1)
    annual_rows = []
    area2 = np.broadcast_to(area, domain.shape)
    for year in years:
        y0 = np.datetime64(f"{year}-01-01", "D")
        y1 = np.datetime64(f"{year}-12-31", "D")
        start = max(y0, dates[0]); end = min(y1, dates[-1])
        if end <= start:
            continue
        td0 = float((start - d_origin).astype("timedelta64[D]").astype(int)) / 365.2425
        td1 = float((end - d_origin).astype("timedelta64[D]").astype(int)) / 365.2425
        th0 = float((start - h_origin).astype("timedelta64[D]").astype(int)) / 365.2425
        th1 = float((end - h_origin).astype("timedelta64[D]").astype(int)) / 365.2425
        dd_mm = _evaluate_trend(d_intercept, d_linear, d_quad, td1) - _evaluate_trend(d_intercept, d_linear, d_quad, td0)
        dh_m = _evaluate_trend(h_intercept, h_linear, h_quad, th1) - _evaluate_trend(h_intercept, h_linear, h_quad, th0)
        dt_m = dd_mm / 1000.0
        dr_m = ske * dh_m
        m = domain & np.isfinite(dt_m) & np.isfinite(dr_m)
        vt = float(np.sum(dt_m[m] * area2[m])); vr = float(np.sum(dr_m[m] * area2[m])); vi = vt - vr
        annual_rows.append({
            "year": year, "start_date": str(start), "end_date": str(end),
            "complete_calendar_year": bool(start == y0 and end == y1),
            "total_gws_change_m3": vt,
            "recoverable_gws_change_m3": vr,
            "irreversible_gws_change_m3": vi,
            "irreversible_storage_loss_magnitude_m3": max(0.0, -vi),
        })
    pd.DataFrame(annual_rows).to_csv(out_dir / "storage_annual_change.csv", index=False)

    final = ts.iloc[ie]
    result = {
        "status": "ok",
        "partition_model": str(sec.get("partition_model", "jiang2018")),
        "baseline_date": str(dates[ib]),
        "end_date": str(dates[ie]),
        "storage_domain_pixels": int(domain.sum()),
        "total_gws_change_m3": float(final["total_gws_change_m3"]),
        "recoverable_gws_change_m3": float(final["recoverable_gws_change_m3"]),
        "irreversible_gws_change_m3": float(final["irreversible_gws_change_m3"]),
        "irreversible_storage_loss_magnitude_m3": float(final["irreversible_storage_loss_magnitude_m3"]),
        "identity": "V_total = V_recoverable + V_irreversible",
    }
    write_json(out_dir / "storage_budget_summary.json", result)
    return result
