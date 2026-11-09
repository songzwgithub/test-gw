from __future__ import annotations

import time
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
    pixel_area_rows,
    read_tif,
    write_json,
    write_tif,
)
from ..config import ProjectConfig
from ..temporal.fit import fit_block_grouped
from ..temporal.model import (
    TimeModel,
    calendar_year_knots,
    design_matrix,
    low_frequency_row,
)


def _nearest_index(
    dates: np.ndarray,
    target,
    default: int,
) -> int:
    if target is None:
        return default
    t = np.datetime64(str(target), "D")
    return int(
        np.argmin(
            np.abs(
                dates.astype("datetime64[D]") - t
            )
        )
    )


def _float_writer(path, grid: dict[str, Any]):
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


def _mask_writer(path, grid: dict[str, Any]):
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


def _sample_median(parts: list[np.ndarray]):
    if not parts:
        return np.nan
    return float(np.nanmedian(np.concatenate(parts)))


def compute_storage_budget(
    cfg: ProjectConfig,
) -> dict[str, Any]:
    """Pixelwise Jiang-style storage partition.

    One fitting pass produces whole-interval maps/volumes, annual regional
    volumes, annual pixelwise maps, recovery-with-continued-compaction masks,
    and cosine-threshold storage sensitivity. The following
    ``annual-storage-maps`` stage only verifies saved rasters and does not refit.
    """
    sec = cfg.section("storage")

    partition_model = str(
        sec.get("partition_model", "jiang2018")
    ).lower()
    if partition_model != "jiang2018":
        raise ValueError(
            "storage.partition_model must be jiang2018"
        )

    ske_choice = str(
        sec.get("ske_product", "pixelwise")
    ).lower()
    block_size = int(sec.get("block_size", 256))
    report_every = int(
        sec.get("report_every_blocks", 10)
    )
    export_annual_maps = bool(
        sec.get("export_annual_maps", True)
    )
    diagnostic_threshold_mm = float(
        sec.get("diagnostic_threshold_mm", 1.0)
    )
    sensitivity_thresholds = sorted(
        {
            float(x)
            for x in sec.get(
                "ske_cosine_sensitivity",
                [0.0, 0.5, 0.7],
            )
        }
    )

    insar_path = (
        cfg.outputs / "canonical" / "insar_stack.h5"
    )
    head_path = (
        cfg.outputs
        / "groundwater"
        / "groundwater_field.h5"
    )
    seasonal = cfg.outputs / "seasonal"
    out_dir = ensure_dir(cfg.outputs / "storage")
    annual_dir = ensure_dir(out_dir / "annual_maps")

    if ske_choice == "pixelwise":
        ske_path = seasonal / "ske_pixelwise.tif"
        mask_path = (
            seasonal
            / "ske_pixelwise_support_mask.tif"
        )
        label = "pixelwise"
    elif ske_choice in {
        "high-confidence",
        "high_confidence",
    }:
        ske_path = (
            seasonal
            / "ske_pixelwise_high_confidence.tif"
        )
        mask_path = (
            seasonal
            / "ske_pixelwise_high_confidence_mask.tif"
        )
        label = "pixelwise_high_confidence"
    else:
        raise ValueError(
            "storage.ske_product must be "
            "pixelwise or high_confidence"
        )

    grid = h5_grid_metadata(insar_path)
    ske = read_tif(ske_path)
    ske_support = read_tif(mask_path) > 0
    gw_support = (
        read_tif(
            cfg.outputs
            / "groundwater"
            / "groundwater_support_mask.tif"
        )
        > 0
    )

    cosine_path = (
        seasonal / "seasonal_vector_cosine.tif"
    )
    if not cosine_path.exists():
        raise FileNotFoundError(
            f"Missing seasonal-vector cosine map: {cosine_path}"
        )
    cosine = read_tif(cosine_path)

    with h5py.File(insar_path, "r") as ih5:
        quantity = str(
            ih5.attrs.get(
                "quantity",
                "vertical_displacement",
            )
        )
        if quantity != "vertical_displacement":
            raise ValueError(
                "Storage budget requires "
                "vertical_displacement"
            )
        i_dates_all = days_to_dates(
            ih5["date_days"][:]
        )

    with h5py.File(head_path, "r") as hh5:
        h_dates_all = days_to_dates(
            hh5["date_days"][:]
        )

    dates, i_idx, h_idx = np.intersect1d(
        i_dates_all,
        h_dates_all,
        assume_unique=True,
        return_indices=True,
    )
    if len(dates) == 0:
        raise ValueError(
            "No common dates for storage budget"
        )

    analysis = cfg.section("analysis")
    start_raw = analysis.get("start_date")
    end_raw = analysis.get("end_date")
    start_analysis = (
        dates[0]
        if start_raw is None
        else np.datetime64(str(start_raw), "D")
    )
    end_analysis = (
        dates[-1]
        if end_raw is None
        else np.datetime64(str(end_raw), "D")
    )
    use = (
        (dates >= start_analysis)
        & (dates <= end_analysis)
    )
    dates = dates[use]
    i_idx = i_idx[use]
    h_idx = h_idx[use]

    if len(dates) < 2:
        raise ValueError(
            "No common dates for storage budget "
            "inside analysis interval"
        )

    ib = _nearest_index(
        dates,
        sec.get("baseline_date"),
        0,
    )
    ie = _nearest_index(
        dates,
        sec.get("end_date"),
        len(dates) - 1,
    )
    if ie < ib:
        raise ValueError(
            "storage.end_date precedes "
            "storage.baseline_date"
        )

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

    final_start = dates[ib]
    final_end = dates[ie]
    dx_final = (
        low_frequency_row(
            final_end,
            model,
            dates[0],
        )
        - low_frequency_row(
            final_start,
            model,
            dates[0],
        )
    )

    intervals = []
    for year in range(
        int(str(final_start)[:4]),
        int(str(final_end)[:4]) + 1,
    ):
        y0 = np.datetime64(
            f"{year}-01-01",
            "D",
        )
        y1 = np.datetime64(
            f"{year + 1}-01-01",
            "D",
        )
        start = max(y0, final_start)
        end = min(y1, final_end)
        if end <= start:
            continue
        dx = (
            low_frequency_row(
                end,
                model,
                dates[0],
            )
            - low_frequency_row(
                start,
                model,
                dates[0],
            )
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

    valid_fraction = float(
        sec.get("valid_fraction", 0.90)
    )
    min_valid = int(
        np.ceil(valid_fraction * len(dates))
    )

    area_rows = pixel_area_rows(
        grid["height"],
        grid["width"],
        grid["crs"],
        grid["transform"],
    )

    main_writers = {
        "domain": _mask_writer(
            out_dir / "storage_domain_mask.tif",
            grid,
        ),
        "total": _float_writer(
            out_dir
            / "total_gws_change_equivalent_mm.tif",
            grid,
        ),
        "recoverable": _float_writer(
            out_dir
            / "recoverable_gws_change_equivalent_mm.tif",
            grid,
        ),
        "irreversible": _float_writer(
            out_dir
            / "irreversible_gws_change_equivalent_mm.tif",
            grid,
        ),
        "head": _float_writer(
            out_dir / "head_lowfreq_change_m.tif",
            grid,
        ),
    }

    annual_writers = {}
    if export_annual_maps:
        for item in intervals:
            year = item["year"]
            annual_writers[year] = {
                "total": _float_writer(
                    annual_dir
                    / f"{year}_total_change_mm.tif",
                    grid,
                ),
                "recoverable": _float_writer(
                    annual_dir
                    / f"{year}_recoverable_change_mm.tif",
                    grid,
                ),
                "irreversible": _float_writer(
                    annual_dir
                    / f"{year}_irreversible_change_mm.tif",
                    grid,
                ),
                "head": _float_writer(
                    annual_dir
                    / f"{year}_head_lowfreq_change_m.tif",
                    grid,
                ),
                "recovery_compaction": _mask_writer(
                    annual_dir
                    / (
                        f"{year}_recovery_with_"
                        "continued_compaction.tif"
                    ),
                    grid,
                ),
            }

    blocks = list(
        block_slices(
            grid["height"],
            grid["width"],
            block_size,
        )
    )
    total_blocks = len(blocks)

    final_totals = np.zeros(4, dtype=float)
    obs_vt = np.zeros(
        ie - ib + 1,
        dtype=float,
    )
    obs_vr = np.zeros_like(obs_vt)

    annual_stats = {
        item["year"]: {
            "volumes": np.zeros(4, dtype=float),
            "domain_area": 0.0,
            "total_negative_area": 0.0,
            "recoverable_positive_area": 0.0,
            "irreversible_negative_area": 0.0,
            "recovery_compaction_area": 0.0,
            "sample_total": [],
            "sample_recoverable": [],
            "sample_irreversible": [],
            "sample_head": [],
        }
        for item in intervals
    }

    sensitivity = {
        threshold: {
            "pixels": 0,
            "area_m2": 0.0,
            "total_m3": 0.0,
            "recoverable_m3": 0.0,
            "irreversible_m3": 0.0,
        }
        for threshold in sensitivity_thresholds
    }

    domain_pixels = 0
    pat_d = 0
    pat_h = 0
    t0 = time.perf_counter()

    print("=" * 80, flush=True)
    print("PIXELWISE STORAGE BUDGET", flush=True)
    print("=" * 80, flush=True)
    print(f"Ske       : {ske_path}", flush=True)
    print(
        f"Epochs    : {len(dates)} "
        f"{dates[0]} -> {dates[-1]}",
        flush=True,
    )
    print(
        f"Interval  : {final_start} -> {final_end}",
        flush=True,
    )
    print(
        f"Annual maps: {export_annual_maps}",
        flush=True,
    )
    print(
        f"Cosine sensitivity: {sensitivity_thresholds}",
        flush=True,
    )
    print(f"Blocks    : {total_blocks}", flush=True)

    try:
        with h5py.File(insar_path, "r") as ih5, h5py.File(
            head_path,
            "r",
        ) as hh5:
            ids = ih5["displacement_mm"]
            hds = hh5["head_anomaly_m"]

            for jb, (
                r0,
                r1,
                c0,
                c1,
            ) in enumerate(
                blocks,
                start=1,
            ):
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

                (
                    dbeta,
                    _drmse,
                    dn,
                    _drss,
                    ndp,
                ) = fit_block_grouped(
                    dflat,
                    X,
                    min_obs,
                )
                (
                    hbeta,
                    _hrmse,
                    hn,
                    _hrss,
                    nhp,
                ) = fit_block_grouped(
                    hflat,
                    X,
                    min_obs,
                )
                pat_d += ndp
                pat_h += nhp

                ske_b = ske[
                    r0:r1,
                    c0:c1,
                ].ravel()
                cosine_b = cosine[
                    r0:r1,
                    c0:c1,
                ].ravel()

                support_b = (
                    ske_support[
                        r0:r1,
                        c0:c1,
                    ].ravel()
                    & gw_support[
                        r0:r1,
                        c0:c1,
                    ].ravel()
                )
                domain = (
                    support_b
                    & np.isfinite(ske_b)
                    & (dn >= min_valid)
                    & (hn >= min_valid)
                )
                domain_pixels += int(domain.sum())

                area_b = np.broadcast_to(
                    area_rows[r0:r1, None],
                    (bh, bw),
                ).ravel()

                dd_mm = dbeta @ dx_final
                dh_m = hbeta @ dx_final
                total_m = dd_mm / 1000.0
                recoverable_m = ske_b * dh_m
                irreversible_m = (
                    total_m - recoverable_m
                )

                good = (
                    domain
                    & np.isfinite(total_m)
                    & np.isfinite(recoverable_m)
                )

                final_totals[0] += float(
                    np.sum(
                        total_m[good] * area_b[good]
                    )
                )
                final_totals[1] += float(
                    np.sum(
                        recoverable_m[good]
                        * area_b[good]
                    )
                )
                final_totals[2] += float(
                    np.sum(
                        irreversible_m[good]
                        * area_b[good]
                    )
                )
                final_totals[3] += float(
                    np.sum(
                        np.maximum(
                            -irreversible_m[good],
                            0.0,
                        )
                        * area_b[good]
                    )
                )

                for threshold in sensitivity_thresholds:
                    sm = (
                        good
                        & np.isfinite(cosine_b)
                        & (cosine_b >= threshold)
                    )
                    ss = sensitivity[threshold]
                    ss["pixels"] += int(sm.sum())
                    ss["area_m2"] += float(
                        np.sum(area_b[sm])
                    )
                    ss["total_m3"] += float(
                        np.sum(
                            total_m[sm]
                            * area_b[sm]
                        )
                    )
                    ss["recoverable_m3"] += float(
                        np.sum(
                            recoverable_m[sm]
                            * area_b[sm]
                        )
                    )
                    ss["irreversible_m3"] += float(
                        np.sum(
                            irreversible_m[sm]
                            * area_b[sm]
                        )
                    )

                out_total = np.full(
                    bh * bw,
                    np.nan,
                    dtype="float32",
                )
                out_rec = out_total.copy()
                out_irr = out_total.copy()
                out_head = out_total.copy()

                out_total[good] = (
                    total_m[good] * 1000.0
                ).astype("float32")
                out_rec[good] = (
                    recoverable_m[good] * 1000.0
                ).astype("float32")
                out_irr[good] = (
                    irreversible_m[good] * 1000.0
                ).astype("float32")
                out_head[good] = (
                    dh_m[good]
                ).astype("float32")

                domain_out = np.full(
                    bh * bw,
                    255,
                    dtype="uint8",
                )
                domain_out[domain] = 1

                win = rasterio.windows.Window(
                    c0,
                    r0,
                    bw,
                    bh,
                )

                main_writers["domain"].write(
                    domain_out.reshape(bh, bw),
                    1,
                    window=win,
                )
                main_writers["total"].write(
                    out_total.reshape(bh, bw),
                    1,
                    window=win,
                )
                main_writers["recoverable"].write(
                    out_rec.reshape(bh, bw),
                    1,
                    window=win,
                )
                main_writers["irreversible"].write(
                    out_irr.reshape(bh, bw),
                    1,
                    window=win,
                )
                main_writers["head"].write(
                    out_head.reshape(bh, bw),
                    1,
                    window=win,
                )

                for item in intervals:
                    year = item["year"]
                    dx = item["dx"]

                    total_mm = dbeta @ dx
                    head_m = hbeta @ dx
                    rec_mm = (
                        ske_b * head_m * 1000.0
                    )
                    irr_mm = total_mm - rec_mm

                    ok = (
                        domain
                        & np.isfinite(total_mm)
                        & np.isfinite(rec_mm)
                    )

                    area_ok = area_b[ok]
                    total_m_y = (
                        total_mm[ok] / 1000.0
                    )
                    rec_m_y = rec_mm[ok] / 1000.0
                    irr_m_y = irr_mm[ok] / 1000.0

                    vt = float(
                        np.sum(total_m_y * area_ok)
                    )
                    vr = float(
                        np.sum(rec_m_y * area_ok)
                    )
                    vi = vt - vr
                    gross = float(
                        np.sum(
                            np.maximum(
                                -irr_m_y,
                                0.0,
                            )
                            * area_ok
                        )
                    )

                    stats = annual_stats[year]
                    stats["volumes"] += [
                        vt,
                        vr,
                        vi,
                        gross,
                    ]
                    stats["domain_area"] += float(
                        np.sum(area_ok)
                    )
                    stats[
                        "total_negative_area"
                    ] += float(
                        np.sum(
                            area_b[
                                ok & (total_mm < 0)
                            ]
                        )
                    )
                    stats[
                        "recoverable_positive_area"
                    ] += float(
                        np.sum(
                            area_b[
                                ok & (rec_mm > 0)
                            ]
                        )
                    )
                    stats[
                        "irreversible_negative_area"
                    ] += float(
                        np.sum(
                            area_b[
                                ok & (irr_mm < 0)
                            ]
                        )
                    )

                    key_mask = (
                        ok
                        & (
                            rec_mm
                            > diagnostic_threshold_mm
                        )
                        & (
                            irr_mm
                            < -diagnostic_threshold_mm
                        )
                    )
                    stats[
                        "recovery_compaction_area"
                    ] += float(
                        np.sum(area_b[key_mask])
                    )

                    sample_idx = np.flatnonzero(
                        ok
                    )[::200]
                    if sample_idx.size:
                        stats["sample_total"].append(
                            total_mm[sample_idx]
                        )
                        stats[
                            "sample_recoverable"
                        ].append(
                            rec_mm[sample_idx]
                        )
                        stats[
                            "sample_irreversible"
                        ].append(
                            irr_mm[sample_idx]
                        )
                        stats["sample_head"].append(
                            head_m[sample_idx]
                        )

                    if export_annual_maps:
                        at = np.full(
                            bh * bw,
                            np.nan,
                            dtype="float32",
                        )
                        ar = at.copy()
                        ai = at.copy()
                        ah = at.copy()

                        at[ok] = total_mm[
                            ok
                        ].astype("float32")
                        ar[ok] = rec_mm[
                            ok
                        ].astype("float32")
                        ai[ok] = irr_mm[
                            ok
                        ].astype("float32")
                        ah[ok] = head_m[
                            ok
                        ].astype("float32")

                        am = np.full(
                            bh * bw,
                            255,
                            dtype="uint8",
                        )
                        am[ok] = 0
                        am[key_mask] = 1

                        ww = annual_writers[year]
                        ww["total"].write(
                            at.reshape(bh, bw),
                            1,
                            window=win,
                        )
                        ww["recoverable"].write(
                            ar.reshape(bh, bw),
                            1,
                            window=win,
                        )
                        ww["irreversible"].write(
                            ai.reshape(bh, bw),
                            1,
                            window=win,
                        )
                        ww["head"].write(
                            ah.reshape(bh, bw),
                            1,
                            window=win,
                        )
                        ww[
                            "recovery_compaction"
                        ].write(
                            am.reshape(bh, bw),
                            1,
                            window=win,
                        )

                # Observed cumulative series retained as a descriptive product.
                db = (
                    dflat[ib:ie + 1]
                    - dflat[ib][None, :]
                ) / 1000.0
                hb = (
                    hflat[ib:ie + 1]
                    - hflat[ib][None, :]
                )
                rec = hb * ske_b[None, :]

                good_ts = (
                    domain[None, :]
                    & np.isfinite(db)
                    & np.isfinite(rec)
                )
                weights = area_b[None, :]

                obs_vt += np.sum(
                    np.where(
                        good_ts,
                        db * weights,
                        0.0,
                    ),
                    axis=1,
                )
                obs_vr += np.sum(
                    np.where(
                        good_ts,
                        rec * weights,
                        0.0,
                    ),
                    axis=1,
                )

                if (
                    jb == 1
                    or jb % max(report_every, 1) == 0
                    or jb == total_blocks
                ):
                    elapsed = (
                        time.perf_counter() - t0
                    )
                    eta = (
                        elapsed
                        * (total_blocks / jb - 1.0)
                    )
                    print(
                        f"[STORAGE] {jb:4d}/{total_blocks} "
                        f"({100*jb/total_blocks:5.1f}%) "
                        f"elapsed={elapsed/60:6.1f} min "
                        f"ETA={eta/60:6.1f} min "
                        f"domain={domain_pixels:,} "
                        f"patterns(d/h)={ndp}/{nhp}",
                        flush=True,
                    )
    finally:
        for dst in main_writers.values():
            dst.close()
        for per_year in annual_writers.values():
            for dst in per_year.values():
                dst.close()

    irr = read_tif(
        out_dir
        / "irreversible_gws_change_equivalent_mm.tif"
    )
    write_tif(
        out_dir
        / "negative_irreversible_change_magnitude_mm.tif",
        np.where(
            np.isfinite(irr) & (irr < 0),
            -irr,
            np.nan,
        ).astype("float32"),
        grid["crs"],
        grid["transform"],
    )

    pd.DataFrame(
        {
            "date":
                dates[ib:ie + 1].astype(str),
            "total_gws_change_m3": obs_vt,
            "recoverable_gws_change_m3": obs_vr,
            "irreversible_gws_change_m3":
                obs_vt - obs_vr,
        }
    ).to_csv(
        out_dir / "storage_cumulative_observed.csv",
        index=False,
    )

    annual_rows = []
    annual_map_rows = []

    for item in intervals:
        year = item["year"]
        stats = annual_stats[year]
        vals = stats["volumes"]
        area = stats["domain_area"]

        row = {
            "year": year,
            "start_date": str(item["start"]),
            "end_date": str(item["end"]),
            "complete_calendar_year":
                item["complete"],
            "total_gws_change_m3": vals[0],
            "recoverable_gws_change_m3":
                vals[1],
            "irreversible_gws_change_m3":
                vals[2],
            "net_irreversible_loss_magnitude_m3":
                max(0.0, -vals[2]),
            "gross_negative_irreversible_change_m3":
                vals[3],
        }
        annual_rows.append(row)

        annual_map_rows.append(
            {
                **row,
                "domain_area_km2":
                    area / 1e6,
                "fraction_area_total_negative":
                    (
                        stats["total_negative_area"]
                        / area
                        if area
                        else np.nan
                    ),
                "fraction_area_recoverable_positive":
                    (
                        stats[
                            "recoverable_positive_area"
                        ]
                        / area
                        if area
                        else np.nan
                    ),
                "fraction_area_irreversible_negative":
                    (
                        stats[
                            "irreversible_negative_area"
                        ]
                        / area
                        if area
                        else np.nan
                    ),
                "recovery_with_continued_compaction_area_km2":
                    (
                        stats[
                            "recovery_compaction_area"
                        ]
                        / 1e6
                    ),
                "fraction_area_recovery_with_continued_compaction":
                    (
                        stats[
                            "recovery_compaction_area"
                        ]
                        / area
                        if area
                        else np.nan
                    ),
                "sample_median_total_mm":
                    _sample_median(
                        stats["sample_total"]
                    ),
                "sample_median_recoverable_mm":
                    _sample_median(
                        stats[
                            "sample_recoverable"
                        ]
                    ),
                "sample_median_irreversible_mm":
                    _sample_median(
                        stats[
                            "sample_irreversible"
                        ]
                    ),
                "sample_median_head_change_m":
                    _sample_median(
                        stats["sample_head"]
                    ),
                "diagnostic_threshold_mm":
                    diagnostic_threshold_mm,
            }
        )

    pd.DataFrame(annual_rows).to_csv(
        out_dir / "storage_annual_change.csv",
        index=False,
    )
    pd.DataFrame(annual_map_rows).to_csv(
        out_dir / "annual_maps_summary.csv",
        index=False,
    )

    sensitivity_rows = []
    for threshold in sensitivity_thresholds:
        ss = sensitivity[threshold]
        area = ss["area_m2"]
        sensitivity_rows.append(
            {
                "min_vector_cosine": threshold,
                "pixels": ss["pixels"],
                "area_km2": area / 1e6,
                "total_gws_change_m3":
                    ss["total_m3"],
                "recoverable_gws_change_m3":
                    ss["recoverable_m3"],
                "irreversible_gws_change_m3":
                    ss["irreversible_m3"],
                "total_equivalent_mm":
                    (
                        ss["total_m3"]
                        / area
                        * 1000.0
                        if area
                        else np.nan
                    ),
                "recoverable_equivalent_mm":
                    (
                        ss["recoverable_m3"]
                        / area
                        * 1000.0
                        if area
                        else np.nan
                    ),
                "irreversible_equivalent_mm":
                    (
                        ss["irreversible_m3"]
                        / area
                        * 1000.0
                        if area
                        else np.nan
                    ),
            }
        )

    pd.DataFrame(sensitivity_rows).to_csv(
        out_dir
        / "storage_ske_cosine_sensitivity.csv",
        index=False,
    )

    result = {
        "status": "ok",
        "ske_product": label,
        "ske_path": str(ske_path),
        "partition_model": "jiang2018",
        "annual_low_frequency_model":
            "continuous_piecewise_linear_plus_annual_harmonic",
        "baseline_date": str(final_start),
        "end_date": str(final_end),
        "common_epochs": int(len(dates)),
        "analysis_window_applied": True,
        "storage_domain_pixels":
            int(domain_pixels),
        "valid_fraction": valid_fraction,
        "annual_maps_exported":
            export_annual_maps,
        "diagnostic_threshold_mm":
            diagnostic_threshold_mm,
        "ske_cosine_sensitivity":
            sensitivity_thresholds,
        "total_gws_change_m3":
            float(final_totals[0]),
        "recoverable_gws_change_m3":
            float(final_totals[1]),
        "irreversible_gws_change_m3":
            float(final_totals[2]),
        "net_irreversible_loss_magnitude_m3":
            float(
                max(
                    0.0,
                    -final_totals[2],
                )
            ),
        "gross_negative_irreversible_change_m3":
            float(final_totals[3]),
        "identity":
            "V_total = V_recoverable + V_irreversible",
        "polyline_knots":
            [str(x) for x in knots],
        "block_size": int(block_size),
        "elapsed_seconds":
            float(time.perf_counter() - t0),
        "mean_deformation_mask_patterns_per_block":
            float(pat_d / total_blocks),
        "mean_head_mask_patterns_per_block":
            float(pat_h / total_blocks),
    }

    write_json(
        out_dir / "storage_budget_summary.json",
        result,
    )
    return result
