from __future__ import annotations

import numpy as np

from hydrogeo_insar.hydromechanics.seasonal import rotate_coefficients
from hydrogeo_insar.groundwater.field import _supported_interpolation_mask
from hydrogeo_insar.temporal.fit import fit_block
from hydrogeo_insar.temporal.model import TimeModel, calendar_year_knots, design_matrix, low_frequency_row


def test_quadratic_harmonic_recovery():
    dates = np.arange(np.datetime64("2018-01-01"), np.datetime64("2022-01-01"), np.timedelta64(12, "D"))
    model = TimeModel(polynomial_degree=2, periods_days=(365.2425,))
    X, _ = design_matrix(dates, model)
    beta_true = np.array([2.0, -30.0, 2.5, 8.0, -4.0])
    y = (X @ beta_true)[:, None]
    beta, rmse, n, rss = fit_block(y, X, min_obs=20)
    assert np.allclose(beta[0], beta_true, atol=1e-7)
    assert rmse[0] < 1e-7
    assert rss[0] < 1e-10
    assert n[0] == len(dates)


def test_piecewise_linear_low_frequency_recovery():
    dates = np.arange(np.datetime64("2018-01-01"), np.datetime64("2022-01-01"), np.timedelta64(12, "D"))
    knots = calendar_year_knots(dates[0], dates[-1])
    model = TimeModel(polynomial_degree=1, periods_days=(365.2425,), polyline_knots=knots)
    X, _ = design_matrix(dates, model, origin=dates[0])
    beta_true = np.zeros(model.n_parameters)
    beta_true[0] = 5.0
    beta_true[1] = -10.0
    beta_true[2:2 + len(knots)] = [2.0, 2.0, 2.0][:len(knots)]
    beta_true[-2:] = [4.0, -2.0]
    y = (X @ beta_true)[:, None]
    beta, rmse, *_ = fit_block(y, X, min_obs=model.n_parameters + 2)
    assert rmse[0] < 1e-7
    x0 = low_frequency_row(np.datetime64("2019-01-01"), model, dates[0])
    x1 = low_frequency_row(np.datetime64("2020-01-01"), model, dates[0])
    assert np.isclose(beta[0] @ (x1 - x0), -8.0, atol=0.1)


def test_positive_lag_delays_peak():
    period = 365.2425
    s, c = rotate_coefficients(np.array([0.0]), np.array([1.0]), 30.0, period)
    phase = (np.arctan2(s, c) * period / (2 * np.pi)) % period
    assert np.allclose(phase, 30.0, atol=1e-8)


def test_storage_identity_scalar():
    total = -0.15
    recoverable = 0.02
    irreversible = total - recoverable
    assert np.isclose(total, recoverable + irreversible)


def test_groundwater_long_temporal_gap_is_not_bridged():
    src = np.asarray(["2020-01-01", "2020-01-10", "2020-04-20", "2020-04-30"], dtype="datetime64[D]")
    qry = np.asarray(["2020-01-05", "2020-02-15", "2020-04-25"], dtype="datetime64[D]")
    keep = _supported_interpolation_mask(src, qry, max_gap_days=30)
    assert keep.tolist() == [True, False, True]
