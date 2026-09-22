from __future__ import annotations

import numpy as np


def fit_block(y: np.ndarray, X: np.ndarray, min_obs: int):
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y)
    n = valid.sum(axis=0)
    y0 = np.where(valid, y, 0.0)
    w = valid.astype(float)
    xtx = np.einsum("tp,ti,tj->pij", w, X, X, optimize=True)
    xty = np.einsum("tp,ti,tp->pi", w, X, y0, optimize=True)
    beta = np.full((y.shape[1], X.shape[1]), np.nan, dtype=float)
    good = n >= max(min_obs, X.shape[1] + 1)
    if good.any():
        ridge = 1e-12 * np.eye(X.shape[1])[None, :, :]
        try:
            beta[good] = np.linalg.solve(
                xtx[good] + ridge,
                xty[good][..., None],
            )[..., 0]
        except np.linalg.LinAlgError:
            for ii in np.flatnonzero(good):
                vi = valid[:, ii]
                beta[ii] = np.linalg.lstsq(X[vi], y[vi, ii], rcond=None)[0]
    pred = X @ np.nan_to_num(beta.T, nan=0.0)
    residual = np.where(valid, y - pred, np.nan)
    rss = np.nansum(residual**2, axis=0)
    rmse = np.sqrt(rss / np.maximum(n, 1))
    rmse[~good] = np.nan
    rss[~good] = np.nan
    return beta, rmse, n, rss


def fit_block_grouped(y: np.ndarray, X: np.ndarray, min_obs: int):
    """Fast LS grouped by identical temporal-validity masks."""
    y = np.asarray(y)
    X = np.asarray(X, dtype=np.float64)
    if y.ndim != 2 or X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError("Expected y=(time,pixel), X=(time,param)")

    _, n_pixel = y.shape
    n_param = X.shape[1]
    valid = np.isfinite(y)
    n_obs = valid.sum(axis=0).astype(np.int32)
    good_idx = np.flatnonzero(n_obs >= max(int(min_obs), n_param + 1))

    beta = np.full((n_pixel, n_param), np.nan, dtype=np.float64)
    rmse = np.full(n_pixel, np.nan, dtype=np.float64)
    rss = np.full(n_pixel, np.nan, dtype=np.float64)

    if good_idx.size == 0:
        return beta, rmse, n_obs, rss, 0

    packed = np.packbits(valid[:, good_idx].T, axis=1)
    _, inverse = np.unique(packed, axis=0, return_inverse=True)
    n_patterns = int(inverse.max()) + 1

    for gid in range(n_patterns):
        cols = good_idx[inverse == gid]
        if cols.size == 0:
            continue
        mask = valid[:, cols[0]]
        Xm = X[mask]
        Y = np.asarray(y[np.ix_(mask, cols)], dtype=np.float64)
        B = np.linalg.pinv(Xm) @ Y
        beta[cols] = B.T
        residual = Y - Xm @ B
        rg = np.einsum("tp,tp->p", residual, residual, optimize=True)
        rss[cols] = rg
        rmse[cols] = np.sqrt(rg / int(mask.sum()))

    return beta, rmse, n_obs, rss, n_patterns


def fit_series(y: np.ndarray, X: np.ndarray, min_obs: int | None = None):
    yy = np.asarray(y, dtype=float)
    if min_obs is None:
        min_obs = X.shape[1] + 2
    beta, rmse, n, rss = fit_block(yy[:, None], X, min_obs=min_obs)
    return beta[0], float(rmse[0]), int(n[0]), float(rss[0])
