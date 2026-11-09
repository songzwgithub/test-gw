import numpy as np
import pandas as pd

from hydrogeo_insar.io.groundwater import _local_spike_statistics


def test_local_spike_filter_keeps_trend_and_flags_gross_point():
    dates = pd.Series(pd.date_range("2020-01-01", periods=61, freq="D"))
    values = pd.Series(np.linspace(-50.0, -55.0, len(dates)))
    values.iloc[30] = -120.0

    flag, med, residual, threshold = _local_spike_statistics(
        dates,
        values,
        window_days=31,
        sigma=8.0,
        absolute_floor_m=15.0,
        min_neighbors=7,
        max_neighbor_gap_days=7,
    )

    assert flag.sum() == 1
    assert flag[30]
    assert residual[30] > threshold[30]


def test_local_spike_filter_does_not_flag_smooth_seasonal_curve():
    dates = pd.Series(pd.date_range("2020-01-01", periods=365, freq="D"))
    t = np.arange(len(dates))
    values = pd.Series(-50.0 + 20.0 * np.sin(2.0 * np.pi * t / 365.2425))

    flag, *_ = _local_spike_statistics(
        dates,
        values,
        window_days=31,
        sigma=8.0,
        absolute_floor_m=15.0,
        min_neighbors=7,
        max_neighbor_gap_days=7,
    )

    assert not flag.any()

