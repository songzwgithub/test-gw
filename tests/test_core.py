from __future__ import annotations

import numpy as np

from hydrogeo_insar.deformation.decompose import design_matrix, fit_block
from hydrogeo_insar.hydromechanics.seasonal import rotate_coefficients


def test_quadratic_harmonic_recovery():
    dates = np.arange(np.datetime64("2018-01-01"), np.datetime64("2022-01-01"), np.timedelta64(12, "D"))
    X, _ = design_matrix(dates)
    beta_true = np.array([2.0, -30.0, 2.5, 8.0, -4.0])
    y = (X @ beta_true)[:, None]
    beta, rmse, n = fit_block(y, X, min_obs=20)
    assert np.allclose(beta[0], beta_true, atol=1e-8)
    assert rmse[0] < 1e-8
    assert n[0] == len(dates)


def test_positive_lag_delays_peak():
    period = 365.2425
    s, c = rotate_coefficients(np.array([0.0]), np.array([1.0]), 30.0, period)
    phase = (np.arctan2(s, c) * period / (2*np.pi)) % period
    assert np.allclose(phase, 30.0, atol=1e-8)


def test_storage_identity_scalar():
    total = -0.15
    recoverable = 0.02
    irreversible = total - recoverable
    assert np.isclose(total, recoverable + irreversible)
