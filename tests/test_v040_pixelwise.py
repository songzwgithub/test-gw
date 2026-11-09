from __future__ import annotations

import numpy as np

from hydrogeo_insar.temporal.fit import fit_block, fit_block_grouped


def test_grouped_fit_matches_general_fit():
    rng = np.random.default_rng(42)
    X = np.column_stack([
        np.ones(20),
        np.linspace(-1, 1, 20),
        np.sin(np.linspace(0, 2*np.pi, 20)),
    ])
    beta_true = rng.normal(size=(8, 3))
    y = X @ beta_true.T
    y[3:5, 2:5] = np.nan
    y[10, 6:] = np.nan

    b0, r0, n0, rss0 = fit_block(y, X, min_obs=6)
    b1, r1, n1, rss1, npat = fit_block_grouped(y, X, min_obs=6)

    assert npat >= 1
    assert np.array_equal(n0, n1)
    assert np.allclose(b0, b1, atol=1e-9, equal_nan=True)
    assert np.allclose(r0, r1, atol=1e-9, equal_nan=True)
    assert np.allclose(rss0, rss1, atol=1e-9, equal_nan=True)


def test_pixelwise_ske_equation():
    h = np.array([3.0, 4.0])
    ske_true = 0.0015
    d = ske_true * h
    ske = np.dot(d, h) / np.dot(h, h)
    assert np.isclose(ske, ske_true)


def test_storage_identity():
    total = np.array([-0.10, -0.02, 0.03])
    recoverable = np.array([0.01, -0.01, 0.02])
    irreversible = total - recoverable
    assert np.allclose(total, recoverable + irreversible)
