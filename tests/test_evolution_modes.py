from __future__ import annotations

import numpy as np

from hydrogeo_insar.evolution.modes import (
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
