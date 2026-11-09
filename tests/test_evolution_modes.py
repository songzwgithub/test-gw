from __future__ import annotations

import numpy as np
import pandas as pd

from hydrogeo_insar.evolution.modes import (
    _groundwater_change_oof_audit,
    _nan_slope,
    build_same_season_pairs,
    classify_deformation_arrays,
    classify_groundwater_state_array,
    classify_trajectory_arrays,
)


def test_same_season_pairing_is_one_to_one_and_near_anniversary():
    dates = np.asarray(
        [
            "2024-01-02",
            "2024-02-03",
            "2024-03-05",
            "2025-01-03",
            "2025-02-02",
            "2025-03-06",
        ],
        dtype="datetime64[D]",
    )
    pairs = build_same_season_pairs(dates, 2024, 1, 8, 18)
    assert len(pairs) == 3
    assert [p["i1"] for p in pairs] == [0, 1, 2]
    assert [p["i2"] for p in pairs] == [3, 4, 5]
    assert max(abs(p["anniversary_offset_days"]) for p in pairs) <= 1


def test_deformation_evolution_classes():
    early = np.asarray([[-10.0, -10.0, -10.0, -10.0, 0.0]])
    late = np.asarray([[-10.0, -20.0, -4.0, -2.0, 0.0]])
    change = late - early

    deformation = classify_deformation_arrays(
        early,
        late,
        change,
        stable_rate_mm_yr=5.0,
        meaningful_rate_change_mm_yr=5.0,
    )
    assert deformation.tolist() == [[2, 3, 5, 5, 1]]


def test_groundwater_state_preserves_transition_zone():
    head = np.asarray([[-2.0, -0.8, -0.2, 0.8, 2.0, np.nan]])
    state = classify_groundwater_state_array(
        head,
        stable_head_change_m_yr=0.5,
        meaningful_head_change_m_yr=1.0,
    )
    assert state.tolist() == [[1, 0, 2, 0, 3, 0]]


def test_coupled_trajectory_classes_are_mutually_interpretable():
    recovery = np.asarray([[0.0, 2.0, 2.0, 2.0, 2.0, 0.0]])
    recent_head = np.asarray([[-2.0, 0.0, 0.0, 0.0, -2.0, 0.0]])
    decline_count = np.asarray([[2.0, 0.0, 0.0, 0.0, 2.0, 0.0]])
    deformation_mode = np.asarray([[2, 4, 2, 3, 3, 1]], dtype="uint8")
    late_rate = np.asarray([[-20.0, -8.0, -15.0, -25.0, -25.0, 0.0]])
    recent_change = np.asarray([[0.0, 10.0, 0.0, -20.0, -20.0, 0.0]])

    coupled = classify_trajectory_arrays(
        recovery,
        recent_head,
        decline_count,
        deformation_mode,
        late_rate,
        recent_change,
        stable_head_change_m_yr=0.5,
        meaningful_head_change_m_yr=1.0,
        stable_rate_mm_yr=5.0,
        meaningful_rate_change_mm_yr=5.0,
        recent_decline_min_intervals=2,
    )
    assert coupled.tolist() == [[1, 2, 3, 4, 5, 6]]


def test_nan_slope_handles_pixelwise_missing_intervals():
    stack = np.asarray(
        [
            [[0.0, 1.0, np.nan]],
            [[1.0, 3.0, 5.0]],
            [[2.0, 5.0, 7.0]],
        ]
    )
    slope = _nan_slope(stack, np.asarray([0.0, 1.0, 2.0]), min_valid=2)
    assert np.allclose(slope[0, :2], [1.0, 2.0])
    assert np.isclose(slope[0, 2], 2.0)

def test_groundwater_change_oof_audit_outputs_interval_metrics():
    rows = []
    for year1, offset in [(2022, 0.0), (2023, 1.0)]:
        for i in range(12):
            x = float((i % 4) * 1000.0)
            y = float((i // 4) * 1000.0)
            value = 0.0015 * x - 0.0008 * y + offset
            rows.append(
                {
                    "station_id": f"W{i:02d}",
                    "lon": 115.0 + x / 100000.0,
                    "lat": 38.0 + y / 100000.0,
                    "x": x,
                    "y": y,
                    "year1": year1,
                    "year2": year1 + 1,
                    "head_change_m_yr": value,
                }
            )

    frame = pd.DataFrame(rows)
    settings = {
        "min_wells_per_interval": 8,
        "folds": 3,
        "block_km": 1.0,
        "random_state": 1,
        "max_centers": 12,
    }
    best = {
        "sigma_km": 3.0,
        "ridge": 0.1,
    }

    predictions, by_interval, confusion = _groundwater_change_oof_audit(
        frame,
        ["2022_2023", "2023_2024"],
        settings,
        best,
        stable_head=0.5,
        meaningful_head=1.0,
    )

    assert len(predictions) == 24
    assert len(by_interval) == 2
    assert set(by_interval.columns) >= {
        "cv_rmse_m_yr",
        "cv_mae_m_yr",
        "cv_bias_m_yr",
        "pearson_r",
        "state_accuracy",
    }
    assert set(predictions["year1"]) == {2022, 2023}
    assert np.isfinite(by_interval["cv_rmse_m_yr"]).all()
    assert not confusion.empty
    assert {"all", "interval"} <= set(confusion["scope"])

