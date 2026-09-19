from __future__ import annotations

import numpy as np

from .fit import fit_block
from .model import TimeModel, design_matrix


def aicc(rss: np.ndarray, n: np.ndarray, k: int) -> np.ndarray:
    rss = np.asarray(rss, dtype=float)
    n = np.asarray(n, dtype=float)
    out = np.full_like(rss, np.inf, dtype=float)
    valid = np.isfinite(rss) & (rss > 0) & (n > k + 1)
    base = n[valid] * np.log(rss[valid] / n[valid]) + 2.0 * k
    correction = 2.0 * k * (k + 1.0) / np.maximum(n[valid] - k - 1.0, 1.0)
    out[valid] = base + correction
    return out


def choose_global_polynomial_degree(
    dates: np.ndarray,
    y: np.ndarray,
    candidates: list[int],
    periods_days: tuple[float, ...] = (365.2425,),
    min_obs: int = 24,
) -> tuple[int, list[dict[str, float]]]:
    """Choose one polynomial degree by median pixel/series AICc.

    y is T x P. A global degree avoids mixing different harmonic definitions spatially.
    """
    rows = []
    best = None
    best_score = np.inf
    for degree in sorted(set(int(v) for v in candidates)):
        model = TimeModel(polynomial_degree=degree, periods_days=periods_days)
        X, _ = design_matrix(dates, model)
        _beta, _rmse, n, rss = fit_block(y, X, min_obs=min_obs)
        vals = aicc(rss, n, model.n_parameters)
        finite = np.isfinite(vals)
        score = float(np.nanmedian(vals[finite])) if finite.any() else np.inf
        rows.append({"polynomial_degree": degree, "median_aicc": score, "valid_series": int(finite.sum())})
        if score < best_score:
            best_score = score
            best = degree
    if best is None:
        raise ValueError("No candidate temporal model could be fitted")
    return int(best), rows
