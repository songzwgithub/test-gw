from __future__ import annotations

import time
from pathlib import Path
from typing import Any

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


def rotate_coefficients(
    sin_coef: np.ndarray,
    cos_coef: np.ndarray,
    lag_days: float,
    period_days: float,
):
    """Coefficients of y(t-lag) for y=s*sin(wt)+c*cos(wt).

    Positive lag delays the response.
    """
    angle = 2.0 * np.pi * float(lag_days) / float(period_days)
    ca, sa = np.cos(angle), np.sin(angle)
    return (
        sin_coef * ca + cos_coef * sa,
        cos_coef * ca - sin_coef * sa,
    )


def _common_dates(
    insar_dates: np.ndarray,
    head_dates: np.ndarray,
):
    return np.intersect1d(
        insar_dates,
        head_dates,
        assume_unique=True,
        return_indices=True,
    )


def _apply_analysis_window(
    cfg: ProjectConfig,
    dates: np.ndarray,
    i_idx: np.ndarray,
    h_idx: np.ndarray,
):
    """Apply the same global analysis interval used by deformation fitting."""
    analysis = cfg.section("analysis")
    start_raw = analysis.get("start_date")
    end_raw = analysis.get("end_date")
    start = (
        dates[0]
        if start_raw is None
        else np.datetime64(str(start_raw), "D")
    )
    end = (
        dates[-1]
        if end_raw is None
        else np.datetime64(str(end_raw), "D")
    )
    keep = (dates >= start) & (dates <= end)
    return dates[keep], i_idx[keep], h_idx[keep]


def _sample_head_series(
    field_path: Path,
    h_idx: np.ndarray,
    max_series: int,
    block_size: int,
    random_state: int = 20260919,
):
    grid = h5_grid_metadata(field_path)
    blocks = list(
        block_slices(
            grid["height"],
            grid["width"],
            block_size,
        )
    )
    np.random.default_rng(random_state).shuffle(blocks)

    collected = []
    n_collected = 0

    with h5py.File(field_path, "r") as h5:
        for r0, r1, c0, c1 in blocks:
            arr = h5["head_anomaly_m"][
                h_idx,
                r0:r1,
                c0:c1,
            ].astype(float)
            T = arr.shape[0]
            flat = arr.reshape(T, -1)

            good = (
                np.isfinite(flat).sum(axis=0)
                >= max(12, T // 2)
            )
            if good.any():
                take = flat[:, good]
                if take.shape[1] > 200:
                    ii = np.linspace(
                        0,
                        take.shape[1] - 1,
                        200,
                    ).astype(int)
                    take = take[:, ii]
                collected.append(take)
                n_collected += take.shape[1]

            if n_collected >= max_series:
                break

    if not collected:
        raise ValueError(
            "No groundwater field pixels are available "
            "for temporal model selection"
        )

    return np.concatenate(
        collected,
        axis=1,
    )[:, :max_series]


def _open_writers(
    out_dir: Path,
    products: list[str],
    grid: dict[str, Any],
):
    profile = geotiff_profile(
        grid["height"],
        grid["width"],
        grid["crs"],
        grid["transform"],
        dtype="float32",
        nodata=np.nan,
    )
    profile["NUM_THREADS"] = "ALL_CPUS"

    return {
        name: rasterio.open(
            out_dir / f"{name}.tif",
            "w",
            **profile,
        )
        for name in products
    }


def compute_joint_harmonics(
    cfg: ProjectConfig,
) -> dict[str, Any]:
    """Fit common-epoch deformation/head annual harmonics.

    The implementation is streaming and groups pixels by identical temporal
    validity masks. This is the publication implementation; no separate
    fast implementation module is required.
    """
    sec = cfg.section("seasonal_response")
    period = float(
        sec.get("annual_period_days", 365.2425)
    )
    min_obs = int(sec.get("min_observations", 24))
    block_size = int(sec.get("block_size", 512))
    report_every = int(
        sec.get("report_every_blocks", 5)
    )

    insar_path = (
        cfg.outputs / "canonical" / "insar_stack.h5"
    )
    head_path = (
        cfg.outputs
        / "groundwater"
        / "groundwater_field.h5"
    )
    out_dir = ensure_dir(cfg.outputs / "seasonal")
    grid = h5_grid_metadata(insar_path)

    with h5py.File(insar_path, "r") as ih5, h5py.File(
        head_path,
        "r",
    ) as hh5:
        insar_dates = days_to_dates(ih5["date_days"][:])
        head_dates = days_to_dates(hh5["date_days"][:])

    dates, i_idx, h_idx = _common_dates(
        insar_dates,
        head_dates,
    )
    if len(dates) == 0:
        raise ValueError(
            "No common InSAR-groundwater epochs"
        )
    dates, i_idx, h_idx = _apply_analysis_window(
        cfg,
        dates,
        i_idx,
        h_idx,
    )

    if len(dates) < min_obs:
        raise ValueError(
            "Too few common InSAR-groundwater epochs "
            "inside analysis interval"
        )

    d_degree = int(
        sec.get("deformation_polynomial_degree", 2)
    )
    g_degree_cfg = sec.get(
        "groundwater_polynomial_degree",
        "auto",
    )
    model_selection_rows = []

    if str(g_degree_cfg).lower() == "auto":
        sample = _sample_head_series(
            head_path,
            h_idx,
            int(
                sec.get(
                    "model_selection_sample_size",
                    3000,
                )
            ),
            min(block_size, 512),
            int(sec.get("random_state", 20260919)),
        )
        candidates = [
            int(v)
            for v in sec.get(
                "groundwater_polynomial_candidates",
                [1, 2],
            )
        ]
        selection_method = str(
            sec.get(
                "groundwater_model_selection",
                "aicc",
            )
        ).lower()

        if (
            selection_method == "f_test"
            and set(candidates) >= {1, 2}
        ):
            (
                g_degree,
                model_selection_rows,
            ) = choose_linear_or_quadratic_f_test(
                dates,
                sample,
                periods_days=(period,),
                min_obs=min_obs,
                alpha=float(
                    sec.get(
                        "groundwater_f_test_alpha",
                        0.05,
                    )
                ),
            )
        else:
            (
                g_degree,
                model_selection_rows,
            ) = choose_global_polynomial_degree(
                dates,
                sample,
                candidates=candidates,
                periods_days=(period,),
                min_obs=min_obs,
            )
    else:
        g_degree = int(g_degree_cfg)

    d_model = TimeModel(
        polynomial_degree=d_degree,
        periods_days=(period,),
    )
    g_model = TimeModel(
        polynomial_degree=g_degree,
        periods_days=(period,),
    )
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

    for name in products:
        p = out_dir / f"{name}.tif"
        if p.exists():
            p.unlink()

    writers = _open_writers(
        out_dir,
        products,
        grid,
    )
    blocks = list(
        block_slices(
            grid["height"],
            grid["width"],
            block_size,
        )
    )

    total = len(blocks)
    t0 = time.perf_counter()
    pat_d_total = 0
    pat_h_total = 0

    print("=" * 80, flush=True)
    print("JOINT HARMONICS", flush=True)
    print("=" * 80, flush=True)
    print(
        f"Common dates: {len(dates)} "
        f"{dates[0]} -> {dates[-1]}",
        flush=True,
    )
    print(
        f"Deformation degree: {d_degree}",
        flush=True,
    )
    print(
        f"Groundwater degree: {g_degree}",
        flush=True,
    )
    print(f"Blocks: {total}", flush=True)

    try:
        with h5py.File(insar_path, "r") as ih5, h5py.File(
            head_path,
            "r",
        ) as hh5:
            ids = ih5["displacement_mm"]
            hds = hh5["head_anomaly_m"]

            for ib, (
                r0,
                r1,
                c0,
                c1,
            ) in enumerate(
                blocks,
                start=1,
            ):
                darr = ids[
                    i_idx,
                    r0:r1,
                    c0:c1,
                ]
                harr = hds[
                    h_idx,
                    r0:r1,
                    c0:c1,
                ]

                T, bh, bw = darr.shape

                (
                    dbeta,
                    drmse,
                    _dn,
                    _drss,
                    npat_d,
                ) = fit_block_grouped(
                    darr.reshape(T, -1),
                    Xd,
                    min_obs,
                )
                (
                    hbeta,
                    hrmse,
                    _hn,
                    _hrss,
                    npat_h,
                ) = fit_block_grouped(
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

                values = {
                    "deformation_annual_sin_mm": ds,
                    "deformation_annual_cos_mm": dc,
                    "deformation_annual_amplitude_mm":
                        np.hypot(ds, dc),
                    "deformation_annual_phase_day":
                        (
                            np.arctan2(ds, dc)
                            * period
                            / (2.0 * np.pi)
                        )
                        % period,
                    "deformation_fit_rmse_mm": drmse,
                    "head_intercept_m": hbeta[:, 0],
                    "head_linear_m_yr": hbeta[:, 1],
                    "head_annual_sin_m": hs,
                    "head_annual_cos_m": hc,
                    "head_annual_amplitude_m":
                        np.hypot(hs, hc),
                    "head_annual_phase_day":
                        (
                            np.arctan2(hs, hc)
                            * period
                            / (2.0 * np.pi)
                        )
                        % period,
                    "head_fit_rmse_m": hrmse,
                }
                if g_degree >= 2:
                    values["head_quadratic_m_yr2"] = (
                        hbeta[:, 2]
                    )

                window = rasterio.windows.Window(
                    c0,
                    r0,
                    bw,
                    bh,
                )
                for name, value in values.items():
                    writers[name].write(
                        value.reshape(
                            bh,
                            bw,
                        ).astype("float32"),
                        1,
                        window=window,
                    )

                if (
                    ib == 1
                    or ib % max(report_every, 1) == 0
                    or ib == total
                ):
                    elapsed = time.perf_counter() - t0
                    eta = (
                        elapsed
                        * (total / ib - 1.0)
                    )
                    print(
                        f"[JOINT] {ib:4d}/{total} "
                        f"({100*ib/total:5.1f}%) "
                        f"elapsed={elapsed/60:6.1f} min "
                        f"ETA={eta/60:6.1f} min "
                        f"patterns(d/h)="
                        f"{npat_d}/{npat_h}",
                        flush=True,
                    )
    finally:
        for dst in writers.values():
            dst.close()

    if model_selection_rows:
        pd.DataFrame(
            model_selection_rows
        ).to_csv(
            out_dir
            / "groundwater_temporal_model_selection.csv",
            index=False,
        )

    result = {
        "status": "ok",
        "implementation":
            "grouped_availability_masks_streaming",
        "output_directory": str(out_dir),
        "common_epochs": int(len(dates)),
        "first_date": str(dates[0]),
        "last_date": str(dates[-1]),
        "analysis_window_applied": True,
        "deformation_polynomial_degree":
            d_degree,
        "groundwater_polynomial_degree":
            g_degree,
        "period_days": period,
        "block_size": block_size,
        "spatial_blocks": total,
        "elapsed_seconds":
            float(time.perf_counter() - t0),
        "mean_deformation_mask_patterns_per_block":
            float(pat_d_total / total),
        "mean_groundwater_mask_patterns_per_block":
            float(pat_h_total / total),
    }
    write_json(
        out_dir / "joint_harmonics_summary.json",
        result,
    )
    return result
