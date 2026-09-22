"""
Fast, streaming replacement for hydrogeo_insar.hydromechanics.seasonal.compute_joint_harmonics.

Scientific model/output semantics are intentionally kept the same as the current test-gw
implementation. The speed-up comes from:
  1) solving once per UNIQUE temporal-availability mask instead of building X'X for every pixel;
  2) writing each block directly to GeoTIFF instead of holding all output rasters in RAM;
  3) explicit block progress / ETA reporting.

This is especially effective for Hengshui because most pixels share the same acquisition
availability pattern.

Run from repository root:
  PYTHONPATH=src python joint_harmonics_fast.py configs/example_project.yaml --block-size 512
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
    write_json,
)
from ..config import ProjectConfig
from .seasonal import _sample_head_series
from ..temporal.fit import fit_block_grouped
from ..temporal.model import (
    TimeModel,
    coefficient_indices,
    design_matrix,
)
from ..temporal.select import (
    choose_global_polynomial_degree,
    choose_linear_or_quadratic_f_test,
)




def open_writers(out_dir: Path, products: list[str], grid: dict):
    profile = geotiff_profile(
        grid["height"],
        grid["width"],
        grid["crs"],
        grid["transform"],
        dtype="float32",
        nodata=np.nan,
    )
    # Parallelize compression if supported by the local GDAL.
    profile["NUM_THREADS"] = "ALL_CPUS"
    return {
        name: rasterio.open(out_dir / f"{name}.tif", "w", **profile)
        for name in products
    }


def compute_joint_harmonics(cfg: ProjectConfig):
    sec = cfg.section("seasonal_response")
    period = float(sec.get("annual_period_days", 365.2425))
    min_obs = int(sec.get("min_observations", 24))
    block_size = int(sec.get("block_size", 512))
    report_every = int(sec.get("report_every_blocks", 5))

    insar_path = cfg.outputs / "canonical" / "insar_stack.h5"
    head_path = cfg.outputs / "groundwater" / "groundwater_field.h5"
    out_dir = ensure_dir(cfg.outputs / "seasonal")
    grid = h5_grid_metadata(insar_path)

    with h5py.File(insar_path, "r") as ih5, h5py.File(head_path, "r") as hh5:
        insar_dates = days_to_dates(ih5["date_days"][:])
        head_dates = days_to_dates(hh5["date_days"][:])
        dates, i_idx, h_idx = np.intersect1d(
            insar_dates,
            head_dates,
            assume_unique=True,
            return_indices=True,
        )

    if len(dates) < min_obs:
        raise ValueError(
            f"Too few common InSAR-groundwater epochs: {len(dates)} < {min_obs}"
        )

    d_degree = int(sec.get("deformation_polynomial_degree", 2))
    g_degree_cfg = sec.get("groundwater_polynomial_degree", "auto")
    model_selection_rows = []

    print("=" * 80, flush=True)
    print("FAST JOINT HARMONICS", flush=True)
    print("=" * 80, flush=True)
    print(f"InSAR       : {insar_path}", flush=True)
    print(f"Groundwater : {head_path}", flush=True)
    print(f"Grid        : {grid['height']} x {grid['width']}", flush=True)
    print(
        f"Common dates: {len(dates)}  {dates[0]} -> {dates[-1]}",
        flush=True,
    )
    print(f"Block size  : {block_size}", flush=True)

    if str(g_degree_cfg).lower() == "auto":
        print("[1/3] Selecting groundwater temporal polynomial degree...", flush=True)
        sample = _sample_head_series(
            head_path,
            h_idx,
            int(sec.get("model_selection_sample_size", 3000)),
            min(block_size, 512),
            int(sec.get("random_state", 20260919)),
        )
        candidates = [
            int(v)
            for v in sec.get("groundwater_polynomial_candidates", [1, 2])
        ]
        selection_method = str(
            sec.get("groundwater_model_selection", "aicc")
        ).lower()
        if selection_method == "f_test" and set(candidates) >= {1, 2}:
            g_degree, model_selection_rows = choose_linear_or_quadratic_f_test(
                dates,
                sample,
                periods_days=(period,),
                min_obs=min_obs,
                alpha=float(sec.get("groundwater_f_test_alpha", 0.05)),
            )
        else:
            g_degree, model_selection_rows = choose_global_polynomial_degree(
                dates,
                sample,
                candidates=candidates,
                periods_days=(period,),
                min_obs=min_obs,
            )
    else:
        g_degree = int(g_degree_cfg)

    print(f"Deformation degree : {d_degree}", flush=True)
    print(f"Groundwater degree : {g_degree}", flush=True)

    d_model = TimeModel(polynomial_degree=d_degree, periods_days=(period,))
    g_model = TimeModel(polynomial_degree=g_degree, periods_days=(period,))
    Xd, _ = design_matrix(dates, d_model)
    Xg, _ = design_matrix(dates, g_model)
    didx = coefficient_indices(d_model)
    gidx = coefficient_indices(g_model)
    dp = didx["periodic"][0]
    gp = gidx["periodic"][0]

    products = [
        "deformation_annual_sin_mm",
        "deformation_annual_cos_mm",
        "deformation_annual_amplitude_mm",
        "deformation_annual_phase_day",
        "deformation_fit_rmse_mm",
        "head_intercept_m",
        "head_linear_m_yr",
        "head_annual_sin_m",
        "head_annual_cos_m",
        "head_annual_amplitude_m",
        "head_annual_phase_day",
        "head_fit_rmse_m",
    ]
    if g_degree >= 2:
        products.append("head_quadratic_m_yr2")

    # Remove only the products this stage owns. This avoids mixing partial/stale files.
    for name in products:
        p = out_dir / f"{name}.tif"
        if p.exists():
            p.unlink()

    writers = open_writers(out_dir, products, grid)

    blocks = list(
        block_slices(grid["height"], grid["width"], block_size)
    )
    total = len(blocks)
    t0 = time.perf_counter()
    pat_d_total = 0
    pat_h_total = 0

    print(f"[2/3] Fitting {total} spatial blocks...", flush=True)

    try:
        with h5py.File(insar_path, "r") as ih5, h5py.File(head_path, "r") as hh5:
            ids = ih5["displacement_mm"]
            hds = hh5["head_anomaly_m"]

            for ib, (r0, r1, c0, c1) in enumerate(blocks, start=1):
                # Read only common epochs.
                darr = ids[i_idx, r0:r1, c0:c1]
                harr = hds[h_idx, r0:r1, c0:c1]

                T, bh, bw = darr.shape
                dbeta, drmse, _dn, _drss, npat_d = fit_block_grouped(
                    darr.reshape(T, -1),
                    Xd,
                    min_obs,
                )
                hbeta, hrmse, _hn, _hrss, npat_h = fit_block_grouped(
                    harr.reshape(T, -1),
                    Xg,
                    min_obs,
                )
                pat_d_total += npat_d
                pat_h_total += npat_h

                ds = dbeta[:, dp["sin"]]
                dc = dbeta[:, dp["cos"]]
                hs = hbeta[:, gp["sin"]]
                hc = hbeta[:, gp["cos"]]

                vals = {
                    "deformation_annual_sin_mm": ds,
                    "deformation_annual_cos_mm": dc,
                    "deformation_annual_amplitude_mm": np.hypot(ds, dc),
                    "deformation_annual_phase_day":
                        (np.arctan2(ds, dc) * period / (2.0 * np.pi)) % period,
                    "deformation_fit_rmse_mm": drmse,
                    "head_intercept_m": hbeta[:, 0],
                    "head_linear_m_yr": hbeta[:, 1],
                    "head_annual_sin_m": hs,
                    "head_annual_cos_m": hc,
                    "head_annual_amplitude_m": np.hypot(hs, hc),
                    "head_annual_phase_day":
                        (np.arctan2(hs, hc) * period / (2.0 * np.pi)) % period,
                    "head_fit_rmse_m": hrmse,
                }
                if g_degree >= 2:
                    vals["head_quadratic_m_yr2"] = hbeta[:, 2]

                window = rasterio.windows.Window(
                    c0, r0, c1 - c0, r1 - r0
                )
                for key, x in vals.items():
                    writers[key].write(
                        x.reshape(bh, bw).astype("float32"),
                        1,
                        window=window,
                    )

                if (
                    ib == 1
                    or ib % max(1, report_every) == 0
                    or ib == total
                ):
                    elapsed = time.perf_counter() - t0
                    rate = ib / max(elapsed, 1e-9)
                    eta = (total - ib) / max(rate, 1e-9)
                    print(
                        f"[JOINT] {ib:4d}/{total} "
                        f"({100.0*ib/total:5.1f}%) "
                        f"elapsed={elapsed/60:6.1f} min "
                        f"ETA={eta/60:6.1f} min "
                        f"patterns(d/h)={npat_d}/{npat_h}",
                        flush=True,
                    )
    finally:
        for dst in writers.values():
            dst.close()

    print("[3/3] Writing summaries...", flush=True)
    if model_selection_rows:
        pd.DataFrame(model_selection_rows).to_csv(
            out_dir / "groundwater_temporal_model_selection.csv",
            index=False,
        )

    result = {
        "status": "ok",
        "implementation": "grouped_availability_masks_streaming",
        "output_directory": str(out_dir),
        "common_epochs": int(len(dates)),
        "first_date": str(dates[0]),
        "last_date": str(dates[-1]),
        "deformation_polynomial_degree": d_degree,
        "groundwater_polynomial_degree": g_degree,
        "period_days": period,
        "block_size": block_size,
        "spatial_blocks": total,
        "elapsed_seconds": float(time.perf_counter() - t0),
        "mean_deformation_mask_patterns_per_block": float(pat_d_total / total),
        "mean_groundwater_mask_patterns_per_block": float(pat_h_total / total),
    }
    write_json(out_dir / "joint_harmonics_summary.json", result)

    return result
