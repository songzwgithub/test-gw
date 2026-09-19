from __future__ import annotations

import numpy as np
from scipy.stats import f as f_dist

from .fit import fit_block
from .model import TimeModel, design_matrix


def aicc(rss: np.ndarray, n: np.ndarray, k: int) -> np.ndarray:
    rss = np.asarray(rss, dtype=float)
    n = np.asarray(n, dtype=float)
    out = np.full_like(rss, np.inf, dtype=float)
    valid = np.isfinite(rss) & (rss >= 0) & (n > k + 1)
    safe = np.maximum(rss[valid], np.finfo(float).tiny)
    base = n[valid] * np.log(safe / n[valid]) + 2.0 * k
    correction = 2.0 * k * (k + 1.0) / (n[valid] - k - 1.0)
    out[valid] = base + correction
    return out


def choose_global_polynomial_degree(
    dates: np.ndarray,
    y: np.ndarray,
    candidates: list[int],
    periods_days: tuple[float, ...] = (365.2425,),
    min_obs: int = 24,
) -> tuple[int, list[dict[str, float]]]:
    """Choose one global polynomial degree by median AICc."""
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


def choose_linear_or_quadratic_f_test(
    dates: np.ndarray,
    y: np.ndarray,
    periods_days: tuple[float, ...] = (365.2425,),
    min_obs: int = 24,
    alpha: float = 0.05,
) -> tuple[int, list[dict[str, float]]]:
    """Global linear-vs-quadratic selection using per-series nested-model F tests.

    The quadratic model is selected when more than half of valid series show a
    significant improvement at the requested alpha level. This keeps a single
    harmonic definition across the reconstructed field.
    """
    m1 = TimeModel(polynomial_degree=1, periods_days=periods_days)
    m2 = TimeModel(polynomial_degree=2, periods_days=periods_days)
    X1, _ = design_matrix(dates, m1)
    X2, _ = design_matrix(dates, m2)
    _b1, _r1, n1, rss1 = fit_block(y, X1, min_obs=min_obs)
    _b2, _r2, n2, rss2 = fit_block(y, X2, min_obs=min_obs)
    n = np.minimum(n1, n2).astype(float)
    valid = np.isfinite(rss1) & np.isfinite(rss2) & (n > m2.n_parameters)
    pvals = np.full(y.shape[1], np.nan, dtype=float)
    if valid.any():
        df1 = m2.n_parameters - m1.n_parameters
        df2 = n[valid] - m2.n_parameters
        num = np.maximum(rss1[valid] - rss2[valid], 0.0) / df1
        den = np.maximum(rss2[valid] / df2, np.finfo(float).tiny)
        fval = num / den
        pvals[valid] = f_dist.sf(fval, df1, df2)
    sig = np.isfinite(pvals) & (pvals < float(alpha))
    fraction = float(sig.sum() / max(1, np.isfinite(pvals).sum()))
    degree = 2 if fraction > 0.5 else 1
    rows = [
        {"method": "nested_f_test", "alpha": float(alpha), "quadratic_significant_fraction": fraction,
         "valid_series": int(np.isfinite(pvals).sum()), "selected_degree": degree}
    ]
    return degree, rows
