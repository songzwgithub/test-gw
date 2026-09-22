from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import h5py
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import xy as raster_xy
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial.distance import cdist

from ..common import (
    affine_to_list,
    block_slices,
    days_to_dates,
    ensure_dir,
    h5_grid_metadata,
    write_json,
    write_tif,
)
from ..config import ProjectConfig
from ..temporal.fit import fit_series
from ..temporal.model import TimeModel, coefficient_indices, design_matrix


@dataclass
class LowRankRBFModel:
    station_ids: list[str]
    station_lonlat: np.ndarray
    station_xy: np.ndarray
    rank: int
    temporal_dates: np.ndarray
    temporal_components: np.ndarray
    score_coefficients: np.ndarray
    centers_xy: np.ndarray
    sigma_m: float
    ridge: float
    projected_crs: str
    baseline_start: str
    baseline_end: str


def _auto_utm_epsg(lon: float, lat: float) -> str:
    zone = int((lon + 180.0) // 6.0) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def _farthest_centers(points: np.ndarray, n: int) -> np.ndarray:
    n = min(max(1, int(n)), len(points))
    selected = [int(np.argmin(points[:, 0] + points[:, 1]))]
    min_dist = cdist(points, points[selected]).ravel()
    while len(selected) < n:
        j = int(np.argmax(min_dist))
        selected.append(j)
        min_dist = np.minimum(
            min_dist,
            cdist(points, points[[j]]).ravel(),
        )
    return points[selected]


def _rbf(points: np.ndarray, centers: np.ndarray, sigma_m: float) -> np.ndarray:
    return np.exp(
        -0.5 * (cdist(points, centers) / float(sigma_m)) ** 2
    )


def _interpolate_short_gaps(
    series: pd.Series,
    max_gap_days: int,
) -> pd.Series:
    if max_gap_days <= 0 or not series.isna().any():
        return series
    s = series.copy()
    miss = s.isna()
    runs = miss.ne(miss.shift()).cumsum()
    interp = s.interpolate(method="time", limit_area="inside")
    for _, idx in miss[miss].groupby(runs).groups.items():
        if len(idx) <= max_gap_days:
            s.loc[idx] = interp.loc[idx]
    return s


def _iterative_svd_impute(
    matrix: np.ndarray,
    rank: int,
    n_iter: int = 20,
    tol: float = 1e-6,
):
    obs = np.isfinite(matrix)
    if not obs.any():
        raise ValueError(
            "Groundwater matrix contains no finite observations"
        )

    filled = matrix.copy()
    row_mean = np.nanmean(matrix, axis=1)
    row_mean = np.where(np.isfinite(row_mean), row_mean, 0.0)
    filled[~obs] = np.repeat(
        row_mean[:, None],
        matrix.shape[1],
        axis=1,
    )[~obs]

    prev_missing = filled[~obs].copy()

    for _ in range(int(n_iter)):
        u, s, vt = np.linalg.svd(
            filled,
            full_matrices=False,
        )
        r = min(
            int(rank),
            len(s),
            max(1, matrix.shape[0] - 1),
        )
        recon = (u[:, :r] * s[:r]) @ vt[:r]
        filled[~obs] = recon[~obs]
        current = filled[~obs]

        if current.size:
            diff = float(
                np.mean((current - prev_missing) ** 2)
            )
            prev_missing = current.copy()
            if diff < tol:
                break

    u, s, vt = np.linalg.svd(
        filled,
        full_matrices=False,
    )
    r = min(
        int(rank),
        len(s),
        max(1, matrix.shape[0] - 1),
    )
    scores = u[:, :r] * s[:r]
    components = vt[:r]
    return filled, scores, components


def _prepare_well_matrix(cfg: ProjectConfig):
    sec = cfg.section("groundwater_field")
    aquifer = str(
        sec.get("aquifer", "confined")
    ).lower()

    gw = pd.read_csv(
        cfg.outputs / "canonical" / "groundwater.csv",
        parse_dates=["date"],
    )
    gw = gw[
        gw["aquifer_class"].astype(str).str.lower()
        == aquifer
    ].copy()

    if gw.empty:
        raise ValueError(
            f"No groundwater observations for "
            f"aquifer_class={aquifer!r}"
        )

    coord_tol = float(
        sec.get("coordinate_tolerance_deg", 1e-5)
    )

    for sid, g in gw.groupby("station_id"):
        if (
            g["lon"].max() - g["lon"].min() > coord_tol
            or g["lat"].max() - g["lat"].min() > coord_tol
        ):
            raise ValueError(
                f"Groundwater station {sid!r} has "
                "inconsistent coordinates"
            )

    analysis = cfg.section("analysis")
    requested_start = pd.Timestamp(
        analysis.get("start_date", gw["date"].min())
    )
    requested_end = pd.Timestamp(
        analysis.get("end_date", gw["date"].max())
    )

    start = max(
        requested_start,
        pd.Timestamp(gw["date"].min()),
    )
    end = min(
        requested_end,
        pd.Timestamp(gw["date"].max()),
    )

    if end <= start:
        raise ValueError(
            "No overlap between requested analysis period "
            "and groundwater observations"
        )

    dates = pd.date_range(start, end, freq="D")

    baseline_cfg = sec.get("baseline", {})
    bstart = max(
        pd.Timestamp(
            baseline_cfg.get("start", start)
        ),
        start,
    )
    bend = min(
        pd.Timestamp(
            baseline_cfg.get(
                "end",
                min(
                    end,
                    start + pd.Timedelta(days=365),
                ),
            )
        ),
        end,
    )

    max_gap = int(
        sec.get("max_gap_days", 7)
    )
    min_baseline_obs = int(
        sec.get("min_baseline_observations", 6)
    )

    meta = (
        gw.groupby("station_id", as_index=False)
        .agg(
            lon=("lon", "first"),
            lat=("lat", "first"),
        )
    )

    rows = []
    ids = []

    for sid, group in gw.groupby("station_id"):
        s = (
            group.groupby("date")["head_m"]
            .median()
            .reindex(dates)
        )
        s = _interpolate_short_gaps(
            s,
            max_gap,
        )

        bsel = s.loc[
            (s.index >= bstart)
            & (s.index <= bend)
        ]

        if int(bsel.notna().sum()) < min_baseline_obs:
            continue

        rows.append(
            (
                s - float(bsel.median())
            ).to_numpy(float)
        )
        ids.append(str(sid))

    if len(rows) < 4:
        raise ValueError(
            "At least four groundwater wells with valid "
            "baselines are required"
        )

    matrix = np.vstack(rows)

    keep = (
        np.isfinite(matrix).mean(axis=1)
        >= float(
            sec.get(
                "min_coverage_fraction",
                0.5,
            )
        )
    )

    matrix = matrix[keep]
    ids = [
        sid
        for sid, k in zip(ids, keep)
        if k
    ]

    if len(ids) < 4:
        raise ValueError(
            "Too few groundwater wells after coverage filtering"
        )

    min_active = float(
        sec.get(
            "min_active_well_fraction",
            0.0,
        )
    )

    if min_active > 0:
        keep_t = (
            np.isfinite(matrix).mean(axis=0)
            >= min_active
        )
        dates = dates[keep_t]
        matrix = matrix[:, keep_t]

    meta["station_id"] = (
        meta["station_id"].astype(str)
    )
    meta = (
        meta.set_index("station_id")
        .loc[ids]
        .reset_index()
    )

    return (
        dates,
        matrix,
        meta,
        str(bstart.date()),
        str(bend.date()),
    )


def _project_wells(
    meta: pd.DataFrame,
    projected_crs: str | None,
):
    lonlat = meta[
        ["lon", "lat"]
    ].to_numpy(float)

    if projected_crs is None:
        projected_crs = _auto_utm_epsg(
            float(
                np.mean(
                    lonlat[:, 0]
                )
            ),
            float(
                np.mean(
                    lonlat[:, 1]
                )
            ),
        )

    tr = Transformer.from_crs(
        "EPSG:4326",
        projected_crs,
        always_xy=True,
    )

    x, y = tr.transform(
        lonlat[:, 0],
        lonlat[:, 1],
    )

    return (
        lonlat,
        np.column_stack([x, y]),
        str(projected_crs),
    )


def _fit_spatial_scores(
    train_xy,
    scores,
    sigma_m,
    ridge,
    max_centers,
):
    centers = _farthest_centers(
        train_xy,
        min(
            max_centers,
            len(train_xy),
        ),
    )

    basis = np.column_stack(
        [
            np.ones(len(train_xy)),
            _rbf(
                train_xy,
                centers,
                sigma_m,
            ),
        ]
    )

    penalty = (
        float(ridge)
        * np.eye(
            basis.shape[1]
        )
    )
    penalty[0, 0] = 0.0

    coef = np.linalg.solve(
        basis.T @ basis + penalty,
        basis.T @ scores,
    )

    return centers, coef


def _predict_scores_xy(
    xy,
    centers,
    sigma_m,
    coef,
):
    basis = np.column_stack(
        [
            np.ones(len(xy)),
            _rbf(
                xy,
                centers,
                sigma_m,
            ),
        ]
    )
    return basis @ coef


def _spatial_fold_ids(
    xy: np.ndarray,
    n_folds: int,
    block_km: float,
    random_state: int,
):
    block_m = float(
        block_km
    ) * 1000.0

    bx = np.floor(
        (
            xy[:, 0]
            - xy[:, 0].min()
        ) / block_m
    ).astype(int)

    by = np.floor(
        (
            xy[:, 1]
            - xy[:, 1].min()
        ) / block_m
    ).astype(int)

    pairs = np.column_stack(
        [bx, by]
    )

    _, block_id = np.unique(
        pairs,
        axis=0,
        return_inverse=True,
    )

    unique = np.unique(
        block_id
    )

    rng = np.random.default_rng(
        random_state
    )
    rng.shuffle(unique)

    mapping = {
        int(b): i % n_folds
        for i, b in enumerate(
            unique
        )
    }

    return np.asarray(
        [
            mapping[int(b)]
            for b in block_id
        ],
        dtype=int,
    )


def _candidate_values(
    sec: dict[str, Any],
    xy: np.ndarray,
):
    ranks = (
        sec.get("rank_candidates")
        or [int(sec.get("rank", 4))]
    )

    rbf = sec.get(
        "rbf",
        {},
    )

    sigmas = rbf.get(
        "sigma_km_candidates"
    )

    if sigmas is None:
        if rbf.get("sigma_km") is not None:
            sigmas = [
                float(
                    rbf["sigma_km"]
                )
            ]
        else:
            d = cdist(
                xy,
                xy,
            )
            vals = d[
                d > 0
            ]
            sigmas = [
                float(
                    np.median(vals)
                    / 1000.0
                )
                if vals.size
                else 20.0
            ]

    ridges = (
        rbf.get(
            "ridge_candidates"
        )
        or [
            float(
                rbf.get(
                    "ridge",
                    1e-3,
                )
            )
        ]
    )

    return (
        [int(r) for r in ranks],
        [float(s) for s in sigmas],
        [float(r) for r in ridges],
    )


def _harmonic_signature(
    dates: np.ndarray,
    values: np.ndarray,
    period_days: float,
    degree: int,
):
    model = TimeModel(
        polynomial_degree=degree,
        periods_days=(
            period_days,
        ),
    )

    X, _ = design_matrix(
        dates,
        model,
    )

    beta, _rmse, _n, _rss = (
        fit_series(
            values,
            X,
            min_obs=max(
                24,
                model.n_parameters + 2,
            ),
        )
    )

    if not np.isfinite(
        beta
    ).all():
        return None

    p = coefficient_indices(
        model
    )["periodic"][0]

    s = float(
        beta[p["sin"]]
    )
    c = float(
        beta[p["cos"]]
    )

    amp = float(
        np.hypot(s, c)
    )

    phase = float(
        (
            np.arctan2(s, c)
            * period_days
            / (2.0 * np.pi)
        )
        % period_days
    )

    linear = (
        float(beta[1])
        if degree >= 1
        else 0.0
    )

    return (
        s,
        c,
        amp,
        phase,
        linear,
    )


def _cv_temporal_metrics(
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    period_days: float,
    degree: int,
):
    amp_err = []
    phase_err = []
    vec_err = []
    trend_err = []

    np_dates = dates.to_numpy(
        dtype="datetime64[D]"
    )

    for i in range(
        actual.shape[0]
    ):
        ok = (
            np.isfinite(
                actual[i]
            )
            & np.isfinite(
                predicted[i]
            )
        )

        if ok.sum() < max(
            24,
            degree + 5,
        ):
            continue

        a = _harmonic_signature(
            np_dates[ok],
            actual[i, ok],
            period_days,
            degree,
        )

        p = _harmonic_signature(
            np_dates[ok],
            predicted[i, ok],
            period_days,
            degree,
        )

        if a is None or p is None:
            continue

        amp_err.append(
            p[2] - a[2]
        )

        pdiff = (
            abs(
                p[3] - a[3]
            )
            % period_days
        )

        phase_err.append(
            min(
                pdiff,
                period_days - pdiff,
            )
        )

        vec_err.append(
            np.hypot(
                p[0] - a[0],
                p[1] - a[1],
            )
        )

        trend_err.append(
            p[4] - a[4]
        )

    if not amp_err:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    return (
        float(
            np.sqrt(
                np.mean(
                    np.square(
                        amp_err
                    )
                )
            )
        ),
        float(
            np.mean(
                phase_err
            )
        ),
        float(
            np.sqrt(
                np.mean(
                    np.square(
                        vec_err
                    )
                )
            )
        ),
        float(
            np.sqrt(
                np.mean(
                    np.square(
                        trend_err
                    )
                )
            )
        ),
    )


def _select_cv_best(
    table: pd.DataFrame,
):
    finite = table[
        np.isfinite(
            table["cv_rmse_m"]
        )
    ].copy()

    if finite.empty:
        raise ValueError(
            "Groundwater spatial cross-validation "
            "produced no valid candidate"
        )

    ordered = finite.sort_values(
        [
            "cv_rmse_m",
            "cv_harmonic_vector_rmse_m",
            "rank",
            "sigma_km",
            "ridge",
        ],
        na_position="last",
    )

    return ordered.index[0]


def select_groundwater_model(
    cfg: ProjectConfig,
    dates,
    matrix,
    meta,
    xy,
):
    sec = cfg.section(
        "groundwater_field"
    )

    ranks, sigmas_km, ridges = (
        _candidate_values(
            sec,
            xy,
        )
    )

    cv = sec.get(
        "cross_validation",
        {},
    )

    n_folds = min(
        int(
            cv.get(
                "folds",
                5,
            )
        ),
        len(meta),
    )

    fold_id = _spatial_fold_ids(
        xy,
        n_folds=n_folds,
        block_km=float(
            cv.get(
                "block_km",
                30.0,
            )
        ),
        random_state=int(
            cv.get(
                "random_state",
                20260919,
            )
        ),
    )

    max_centers = int(
        sec.get(
            "rbf",
            {},
        ).get(
            "max_centers",
            32,
        )
    )

    period_days = float(
        cv.get(
            "annual_period_days",
            365.2425,
        )
    )

    harmonic_degree = int(
        cv.get(
            "harmonic_polynomial_degree",
            1,
        )
    )

    svd_iterations = int(
        sec.get(
            "svd_iterations",
            20,
        )
    )

    t0 = time.perf_counter()

    print(
        "[GW-CV] full-series RMSE selection",
        flush=True,
    )

    fold_cache = {}

    for fold in range(
        n_folds
    ):
        train = (
            fold_id != fold
        )
        test = ~train

        if (
            train.sum() < 4
            or test.sum() == 0
        ):
            continue

        train_xy = xy[
            train
        ]
        test_xy = xy[
            test
        ]

        centers = _farthest_centers(
            train_xy,
            min(
                max_centers,
                len(train_xy),
            ),
        )

        sigma_cache = {}

        for sigma_km in sigmas_km:
            sigma_m = (
                float(
                    sigma_km
                )
                * 1000.0
            )

            btrain = np.column_stack(
                [
                    np.ones(
                        len(
                            train_xy
                        )
                    ),
                    _rbf(
                        train_xy,
                        centers,
                        sigma_m,
                    ),
                ]
            )

            btest = np.column_stack(
                [
                    np.ones(
                        len(
                            test_xy
                        )
                    ),
                    _rbf(
                        test_xy,
                        centers,
                        sigma_m,
                    ),
                ]
            )

            sigma_cache[
                float(
                    sigma_km
                )
            ] = {
                "btrain": btrain,
                "btest": btest,
                "btb": (
                    btrain.T
                    @ btrain
                ),
            }

        fold_cache[
            fold
        ] = {
            "train": train,
            "test": test,
            "sigma": sigma_cache,
        }

    svd_cache = {}

    total_svd = (
        len(ranks)
        * len(fold_cache)
    )
    svd_done = 0

    for rank in ranks:
        for fold, info in (
            fold_cache.items()
        ):
            train = info[
                "train"
            ]

            if train.sum() < max(
                4,
                rank + 1,
            ):
                continue

            _filled, scores, components = (
                _iterative_svd_impute(
                    matrix[train],
                    rank=rank,
                    n_iter=svd_iterations,
                )
            )

            svd_cache[
                (
                    int(rank),
                    int(fold),
                )
            ] = (
                scores,
                components,
            )

            svd_done += 1

            print(
                f"[GW-CV:SVD] "
                f"{svd_done}/{total_svd} "
                f"rank={rank} "
                f"fold={fold+1}/{n_folds}",
                flush=True,
            )

    rows = []

    total_candidates = (
        len(ranks)
        * len(sigmas_km)
        * len(ridges)
    )
    candidate_no = 0

    for rank in ranks:
        for sigma_km in sigmas_km:
            for ridge in ridges:
                sqerr = []
                amp_metrics = []
                phase_metrics = []
                vector_metrics = []
                trend_metrics = []
                n_test = 0

                for fold, info in (
                    fold_cache.items()
                ):
                    key = (
                        int(rank),
                        int(fold),
                    )

                    if key not in svd_cache:
                        continue

                    train = info[
                        "train"
                    ]
                    test = info[
                        "test"
                    ]

                    scores, components = (
                        svd_cache[key]
                    )

                    basis = info[
                        "sigma"
                    ][
                        float(
                            sigma_km
                        )
                    ]

                    btrain = basis[
                        "btrain"
                    ]
                    btest = basis[
                        "btest"
                    ]

                    penalty = (
                        float(
                            ridge
                        )
                        * np.eye(
                            btrain.shape[1]
                        )
                    )
                    penalty[
                        0,
                        0,
                    ] = 0.0

                    coef = np.linalg.solve(
                        basis["btb"]
                        + penalty,
                        btrain.T
                        @ scores,
                    )

                    pred_scores = (
                        btest
                        @ coef
                    )

                    pred = (
                        pred_scores
                        @ components
                    )

                    actual = matrix[
                        test
                    ]

                    ok = (
                        np.isfinite(
                            actual
                        )
                        & np.isfinite(
                            pred
                        )
                    )

                    if ok.any():
                        sqerr.append(
                            (
                                actual[ok]
                                - pred[ok]
                            ) ** 2
                        )
                        n_test += int(
                            ok.sum()
                        )

                    am, ph, hv, tr = (
                        _cv_temporal_metrics(
                            dates,
                            actual,
                            pred,
                            period_days,
                            harmonic_degree,
                        )
                    )

                    if np.isfinite(am):
                        amp_metrics.append(
                            am
                        )
                    if np.isfinite(ph):
                        phase_metrics.append(
                            ph
                        )
                    if np.isfinite(hv):
                        vector_metrics.append(
                            hv
                        )
                    if np.isfinite(tr):
                        trend_metrics.append(
                            tr
                        )

                row = {
                    "rank": rank,
                    "sigma_km": sigma_km,
                    "ridge": ridge,
                    "cv_rmse_m": (
                        float(
                            np.sqrt(
                                np.mean(
                                    np.concatenate(
                                        sqerr
                                    )
                                )
                            )
                        )
                        if sqerr
                        else np.inf
                    ),
                    "cv_annual_amplitude_rmse_m": (
                        float(
                            np.sqrt(
                                np.mean(
                                    np.square(
                                        amp_metrics
                                    )
                                )
                            )
                        )
                        if amp_metrics
                        else np.nan
                    ),
                    "cv_phase_mae_days": (
                        float(
                            np.mean(
                                phase_metrics
                            )
                        )
                        if phase_metrics
                        else np.nan
                    ),
                    "cv_harmonic_vector_rmse_m": (
                        float(
                            np.sqrt(
                                np.mean(
                                    np.square(
                                        vector_metrics
                                    )
                                )
                            )
                        )
                        if vector_metrics
                        else np.nan
                    ),
                    "cv_linear_trend_rmse_m_yr": (
                        float(
                            np.sqrt(
                                np.mean(
                                    np.square(
                                        trend_metrics
                                    )
                                )
                            )
                        )
                        if trend_metrics
                        else np.nan
                    ),
                    "n_test_values": n_test,
                }

                rows.append(
                    row
                )

                candidate_no += 1

                elapsed = (
                    time.perf_counter()
                    - t0
                )

                print(
                    f"[GW-CV] "
                    f"{candidate_no}/"
                    f"{total_candidates} "
                    f"rank={rank} "
                    f"sigma={sigma_km:g}km "
                    f"ridge={ridge:g} "
                    f"RMSE="
                    f"{row['cv_rmse_m']:.4f}m "
                    f"elapsed="
                    f"{elapsed/60:.1f}m",
                    flush=True,
                )

    table = pd.DataFrame(
        rows
    )

    best_idx = _select_cv_best(
        table
    )

    table[
        "selected"
    ] = False

    table.loc[
        best_idx,
        "selected",
    ] = True

    table = table.sort_values(
        [
            "selected",
            "cv_rmse_m",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    best = table.iloc[
        0
    ]

    print(
        "[GW-CV] selected "
        f"rank={int(best['rank'])}, "
        f"sigma="
        f"{float(best['sigma_km']):g}km, "
        f"ridge="
        f"{float(best['ridge']):g}, "
        f"RMSE="
        f"{float(best['cv_rmse_m']):.4f}m",
        flush=True,
    )

    return (
        int(
            best["rank"]
        ),
        float(
            best["sigma_km"]
        ),
        float(
            best["ridge"]
        ),
        table,
    )


def _inside_hull_mask(
    points_xy: np.ndarray,
    query_xy: np.ndarray,
):
    if len(points_xy) < 3:
        return np.ones(
            len(query_xy),
            dtype=bool,
        )

    return (
        Delaunay(
            points_xy
        ).find_simplex(
            query_xy
        )
        >= 0
    )


def _supported_interpolation_mask(
    source_dates: np.ndarray,
    query_dates: np.ndarray,
    max_gap_days: int | None,
) -> np.ndarray:
    src = np.asarray(
        source_dates,
        dtype="datetime64[D]",
    ).astype(
        np.int64
    )

    qry = np.asarray(
        query_dates,
        dtype="datetime64[D]",
    ).astype(
        np.int64
    )

    if len(src) == 0:
        return np.zeros(
            len(qry),
            dtype=bool,
        )

    if max_gap_days is None:
        return (
            (qry >= src[0])
            & (qry <= src[-1])
        )

    pos = np.searchsorted(
        src,
        qry,
        side="left",
    )

    exact = np.zeros(
        len(qry),
        dtype=bool,
    )

    inside_pos = (
        pos < len(src)
    )

    exact[
        inside_pos
    ] = (
        src[
            pos[
                inside_pos
            ]
        ]
        == qry[
            inside_pos
        ]
    )

    left = pos - 1
    right = pos

    bracketed = (
        (left >= 0)
        & (right < len(src))
    )

    allowed = exact.copy()

    if bracketed.any():
        gap = (
            src[
                right[
                    bracketed
                ]
            ]
            - src[
                left[
                    bracketed
                ]
            ]
        )

        allowed[
            bracketed
        ] |= (
            gap
            <= int(
                max_gap_days
            )
        )

    return allowed


def build_groundwater_field(
    cfg: ProjectConfig,
) -> dict[str, Any]:
    stage_t0 = (
        time.perf_counter()
    )

    dates, matrix, meta, bstart, bend = (
        _prepare_well_matrix(
            cfg
        )
    )

    sec = cfg.section(
        "groundwater_field"
    )

    lonlat, xy, projected_crs = (
        _project_wells(
            meta,
            sec.get(
                "projected_crs"
            ),
        )
    )

    rank, sigma_km, ridge, cv_table = (
        select_groundwater_model(
            cfg,
            dates,
            matrix,
            meta,
            xy,
        )
    )

    _filled, scores, components = (
        _iterative_svd_impute(
            matrix,
            rank=rank,
            n_iter=int(
                sec.get(
                    "svd_iterations",
                    20,
                )
            ),
        )
    )

    max_centers = int(
        sec.get(
            "rbf",
            {},
        ).get(
            "max_centers",
            32,
        )
    )

    centers, coef = (
        _fit_spatial_scores(
            xy,
            scores,
            sigma_m=(
                sigma_km
                * 1000.0
            ),
            ridge=ridge,
            max_centers=max_centers,
        )
    )

    model = LowRankRBFModel(
        station_ids=(
            meta[
                "station_id"
            ]
            .astype(str)
            .tolist()
        ),
        station_lonlat=lonlat,
        station_xy=xy,
        rank=components.shape[0],
        temporal_dates=dates.to_numpy(
            dtype="datetime64[D]"
        ),
        temporal_components=components,
        score_coefficients=coef,
        centers_xy=centers,
        sigma_m=(
            sigma_km
            * 1000.0
        ),
        ridge=ridge,
        projected_crs=projected_crs,
        baseline_start=bstart,
        baseline_end=bend,
    )

    insar_path = (
        cfg.outputs
        / "canonical"
        / "insar_stack.h5"
    )

    grid = h5_grid_metadata(
        insar_path
    )

    with h5py.File(
        insar_path,
        "r",
    ) as ih5:
        all_insar_dates = (
            days_to_dates(
                ih5[
                    "date_days"
                ][:]
            )
        )

    gw_start = (
        model.temporal_dates[0]
    )
    gw_end = (
        model.temporal_dates[-1]
    )

    candidate_dates = (
        all_insar_dates[
            (
                all_insar_dates
                >= gw_start
            )
            & (
                all_insar_dates
                <= gw_end
            )
        ]
    )

    max_temporal_gap = (
        sec.get(
            "max_temporal_interpolation_gap_days",
            20,
        )
    )

    max_temporal_gap = (
        None
        if max_temporal_gap is None
        else int(
            max_temporal_gap
        )
    )

    temporal_support = (
        _supported_interpolation_mask(
            model.temporal_dates,
            candidate_dates,
            max_temporal_gap,
        )
    )

    obs_dates = (
        candidate_dates[
            temporal_support
        ]
    )

    if len(obs_dates) < 12:
        raise ValueError(
            "Too few InSAR epochs overlap "
            "the groundwater temporal support"
        )

    src_days = (
        model.temporal_dates
        .astype(
            "datetime64[D]"
        )
        .astype(
            np.int64
        )
    )

    dst_days = (
        obs_dates
        .astype(
            "datetime64[D]"
        )
        .astype(
            np.int64
        )
    )

    comp = np.vstack(
        [
            np.interp(
                dst_days,
                src_days,
                row,
            )
            for row in (
                model.temporal_components
            )
        ]
    )

    out_dir = ensure_dir(
        cfg.outputs
        / "groundwater"
    )

    cv_table.to_csv(
        out_dir
        / "groundwater_model_cv.csv",
        index=False,
    )

    out_path = (
        out_dir
        / "groundwater_field.h5"
    )

    block_size = int(
        sec.get(
            "grid_block_size",
            256,
        )
    )

    transform = (
        grid[
            "transform"
        ]
    )

    height = grid[
        "height"
    ]
    width = grid[
        "width"
    ]

    grid_to_geo = (
        Transformer.from_crs(
            grid["crs"],
            "EPSG:4326",
            always_xy=True,
        )
    )

    geo_to_model = (
        Transformer.from_crs(
            "EPSG:4326",
            model.projected_crs,
            always_xy=True,
        )
    )

    domain_path = (
        cfg.outputs
        / "canonical"
        / "domain_mask.tif"
    )

    if domain_path.exists():
        with rasterio.open(
            domain_path
        ) as src:
            domain_full = (
                src.read(1)
                > 0
            )
    else:
        domain_full = np.ones(
            (
                height,
                width,
            ),
            dtype=bool,
        )

    if domain_full.shape != (
        height,
        width,
    ):
        raise ValueError(
            "domain_mask.tif geometry "
            "does not match the canonical grid"
        )

    quality_cfg = sec.get(
        "quality",
        {},
    )

    max_quality_distance_km = (
        quality_cfg.get(
            "max_nearest_well_km"
        )
    )

    max_quality_distance_m = (
        None
        if max_quality_distance_km is None
        else float(
            max_quality_distance_km
        )
        * 1000.0
    )

    prediction_support_full = (
        domain_full.astype(
            "uint8"
        )
    )

    inside_hull_full = np.zeros(
        (
            height,
            width,
        ),
        dtype="uint8",
    )

    high_confidence_full = np.zeros(
        (
            height,
            width,
        ),
        dtype="uint8",
    )

    nearest_distance_km_full = np.full(
        (
            height,
            width,
        ),
        np.nan,
        dtype="float32",
    )

    tree = cKDTree(
        model.station_xy
    )

    compression = str(
        sec.get(
            "hdf5_compression",
            "none",
        )
    ).strip().lower()

    if compression in {
        "",
        "none",
        "null",
        "off",
    }:
        compression_kwargs = {}
    elif compression == "lzf":
        compression_kwargs = {
            "compression": "lzf"
        }
    elif compression == "gzip":
        compression_kwargs = {
            "compression": "gzip",
            "compression_opts": int(
                sec.get(
                    "hdf5_gzip_level",
                    1,
                )
            ),
        }
    else:
        raise ValueError(
            "groundwater_field.hdf5_compression "
            "must be one of: none, lzf, gzip"
        )

    blocks = list(
        block_slices(
            height,
            width,
            block_size,
        )
    )

    total_blocks = len(
        blocks
    )

    raster_t0 = (
        time.perf_counter()
    )

    with h5py.File(
        out_path,
        "w",
        libver="latest",
    ) as h5:
        ds = h5.create_dataset(
            "head_anomaly_m",
            shape=(
                len(obs_dates),
                height,
                width,
            ),
            dtype="float32",
            chunks=(
                1,
                min(
                    block_size,
                    height,
                ),
                min(
                    block_size,
                    width,
                ),
            ),
            fillvalue=np.nan,
            **compression_kwargs,
        )

        h5.create_dataset(
            "date_days",
            data=(
                obs_dates
                - np.datetime64(
                    "1970-01-01",
                    "D",
                )
            ).astype(
                np.int32
            ),
        )

        h5.attrs[
            "height"
        ] = height
        h5.attrs[
            "width"
        ] = width
        h5.attrs[
            "crs"
        ] = grid[
            "crs"
        ]
        h5.attrs[
            "transform"
        ] = affine_to_list(
            transform
        )
        h5.attrs[
            "aquifer_class"
        ] = sec.get(
            "aquifer",
            "confined",
        )
        h5.attrs[
            "baseline_start"
        ] = bstart
        h5.attrs[
            "baseline_end"
        ] = bend
        h5.attrs[
            "method"
        ] = (
            "lowrank_rbf_spatial_cv_full_domain"
        )
        h5.attrs[
            "selection_criterion"
        ] = (
            "minimum_full_series_spatial_cv_rmse"
        )
        h5.attrs[
            "rank"
        ] = rank
        h5.attrs[
            "sigma_km"
        ] = sigma_km
        h5.attrs[
            "ridge"
        ] = ridge
        h5.attrs[
            "projected_crs"
        ] = projected_crs

        for block_no, (
            r0,
            r1,
            c0,
            c1,
        ) in enumerate(
            blocks,
            start=1,
        ):
            rr, cc = np.meshgrid(
                np.arange(
                    r0,
                    r1,
                ),
                np.arange(
                    c0,
                    c1,
                ),
                indexing="ij",
            )

            xs, ys = raster_xy(
                transform,
                rr.ravel(),
                cc.ravel(),
                offset="center",
            )

            lon, lat = (
                grid_to_geo.transform(
                    np.asarray(xs),
                    np.asarray(ys),
                )
            )

            mx, my = (
                geo_to_model.transform(
                    lon,
                    lat,
                )
            )

            qxy = np.column_stack(
                [
                    mx,
                    my,
                ]
            )

            domain_block = (
                domain_full[
                    r0:r1,
                    c0:c1,
                ]
                .reshape(-1)
            )

            values = np.full(
                (
                    len(obs_dates),
                    len(qxy),
                ),
                np.nan,
                dtype="float32",
            )

            inside_hull = np.zeros(
                len(qxy),
                dtype=bool,
            )

            nearest_m = np.full(
                len(qxy),
                np.nan,
                dtype=float,
            )

            if domain_block.any():
                q_domain = qxy[
                    domain_block
                ]

                pred_scores = (
                    _predict_scores_xy(
                        q_domain,
                        model.centers_xy,
                        model.sigma_m,
                        model.score_coefficients,
                    )
                )

                values[
                    :,
                    domain_block,
                ] = (
                    pred_scores
                    @ comp
                ).T.astype(
                    "float32",
                    copy=False,
                )

                inside_hull[
                    domain_block
                ] = _inside_hull_mask(
                    model.station_xy,
                    q_domain,
                )

                dist, _ = tree.query(
                    q_domain,
                    k=1,
                )

                nearest_m[
                    domain_block
                ] = dist

            ds[
                :,
                r0:r1,
                c0:c1,
            ] = values.reshape(
                len(obs_dates),
                r1 - r0,
                c1 - c0,
            )

            inside_hull_full[
                r0:r1,
                c0:c1,
            ] = (
                inside_hull.reshape(
                    r1 - r0,
                    c1 - c0,
                )
                .astype(
                    "uint8"
                )
            )

            nearest_distance_km_full[
                r0:r1,
                c0:c1,
            ] = (
                nearest_m.reshape(
                    r1 - r0,
                    c1 - c0,
                )
                / 1000.0
            ).astype(
                "float32"
            )

            high_conf = (
                domain_block
                & inside_hull
            )

            if (
                max_quality_distance_m
                is not None
            ):
                high_conf &= (
                    nearest_m
                    <= max_quality_distance_m
                )

            high_confidence_full[
                r0:r1,
                c0:c1,
            ] = (
                high_conf.reshape(
                    r1 - r0,
                    c1 - c0,
                )
                .astype(
                    "uint8"
                )
            )

            if (
                block_no == 1
                or block_no
                % max(
                    1,
                    total_blocks // 50,
                )
                == 0
                or block_no
                == total_blocks
            ):
                elapsed = (
                    time.perf_counter()
                    - raster_t0
                )

                frac = (
                    block_no
                    / total_blocks
                )

                eta = (
                    elapsed
                    * (
                        1.0 / frac
                        - 1.0
                    )
                )

                print(
                    f"[GW-RASTER] "
                    f"{block_no}/"
                    f"{total_blocks} "
                    f"({100*frac:5.1f}%) "
                    f"elapsed="
                    f"{elapsed/60:.1f}m "
                    f"ETA="
                    f"{eta/60:.1f}m",
                    flush=True,
                )

    write_tif(
        out_dir
        / "groundwater_support_mask.tif",
        prediction_support_full,
        grid["crs"],
        grid["transform"],
        nodata=0,
        dtype="uint8",
    )

    write_tif(
        out_dir
        / "groundwater_inside_well_hull.tif",
        inside_hull_full,
        grid["crs"],
        grid["transform"],
        nodata=0,
        dtype="uint8",
    )

    write_tif(
        out_dir
        / "groundwater_high_confidence_mask.tif",
        high_confidence_full,
        grid["crs"],
        grid["transform"],
        nodata=0,
        dtype="uint8",
    )

    write_tif(
        out_dir
        / "groundwater_nearest_well_distance_km.tif",
        nearest_distance_km_full,
        grid["crs"],
        grid["transform"],
        nodata=np.nan,
        dtype="float32",
    )

    np.savez_compressed(
        out_dir
        / "groundwater_model.npz",
        station_ids=np.asarray(
            model.station_ids
        ),
        station_lonlat=(
            model.station_lonlat
        ),
        station_xy=(
            model.station_xy
        ),
        rank=model.rank,
        temporal_dates=(
            model.temporal_dates
            .astype(str)
        ),
        temporal_components=(
            model.temporal_components
        ),
        score_coefficients=(
            model.score_coefficients
        ),
        centers_xy=(
            model.centers_xy
        ),
        sigma_m=(
            model.sigma_m
        ),
        ridge=model.ridge,
        projected_crs=(
            model.projected_crs
        ),
        baseline_start=(
            model.baseline_start
        ),
        baseline_end=(
            model.baseline_end
        ),
    )

    score_table = (
        meta.copy()
    )

    for k in range(
        model.rank
    ):
        score_table[
            f"score_{k+1}"
        ] = scores[
            :,
            k,
        ]

    score_table.to_csv(
        out_dir
        / "groundwater_spatial_scores.csv",
        index=False,
    )

    best = cv_table.iloc[
        0
    ]

    domain_pixels = int(
        domain_full.sum()
    )

    inside_hull_pixels = int(
        (
            inside_hull_full
            > 0
        ).sum()
    )

    high_confidence_pixels = int(
        (
            high_confidence_full
            > 0
        ).sum()
    )

    summary = {
        "status": "ok",
        "output": str(
            out_path
        ),
        "n_wells": len(
            model.station_ids
        ),
        "rank": rank,
        "sigma_km": sigma_km,
        "ridge": ridge,
        "selection_criterion": (
            "minimum_full_series_spatial_cv_rmse"
        ),
        "projected_crs": (
            projected_crs
        ),
        "baseline_start": bstart,
        "baseline_end": bend,
        "first_supported_date": str(
            obs_dates[0]
        ),
        "last_supported_date": str(
            obs_dates[-1]
        ),
        "epochs": len(
            obs_dates
        ),
        "max_temporal_interpolation_gap_days": (
            max_temporal_gap
        ),
        "domain_pixels": domain_pixels,
        "prediction_pixels": domain_pixels,
        "prediction_fraction_of_domain": 1.0,
        "inside_well_hull_pixels": (
            inside_hull_pixels
        ),
        "inside_well_hull_fraction": (
            inside_hull_pixels
            / domain_pixels
            if domain_pixels
            else np.nan
        ),
        "high_confidence_pixels": (
            high_confidence_pixels
        ),
        "high_confidence_fraction": (
            high_confidence_pixels
            / domain_pixels
            if domain_pixels
            else np.nan
        ),
        "quality_max_nearest_well_km": (
            max_quality_distance_km
        ),
        "hdf5_compression": compression,
        "cv_rmse_m": float(
            best[
                "cv_rmse_m"
            ]
        ),
        "cv_annual_amplitude_rmse_m": float(
            best[
                "cv_annual_amplitude_rmse_m"
            ]
        ),
        "cv_phase_mae_days": float(
            best[
                "cv_phase_mae_days"
            ]
        ),
        "cv_harmonic_vector_rmse_m": float(
            best[
                "cv_harmonic_vector_rmse_m"
            ]
        ),
        "cv_linear_trend_rmse_m_yr": float(
            best[
                "cv_linear_trend_rmse_m_yr"
            ]
        ),
    }

    write_json(
        out_dir
        / "groundwater_field_summary.json",
        summary,
    )

    print(
        "[GW-FIELD] done "
        f"total="
        f"{(time.perf_counter()-stage_t0)/60:.1f}m",
        flush=True,
    )

    return summary
