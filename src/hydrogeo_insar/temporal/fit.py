from __future__ import annotations

import numpy as np


def fit_block(y: np.ndarray, X: np.ndarray, min_obs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted-by-availability least squares for a T x P matrix of time series.

    Returns beta [P,n_param], RMSE [P], n_obs [P], RSS [P].
    """
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
            beta[good] = np.linalg.solve(xtx[good] + ridge, xty[good][..., None])[..., 0]
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


def fit_series(y: np.ndarray, X: np.ndarray, min_obs: int | None = None):
    yy = np.asarray(y, dtype=float)
    if min_obs is None:
        min_obs = X.shape[1] + 2
    beta, rmse, n, rss = fit_block(yy[:, None], X, min_obs=min_obs)
    return beta[0], float(rmse[0]), int(n[0]), float(rss[0])
