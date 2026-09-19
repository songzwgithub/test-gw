from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeModel:
    polynomial_degree: int = 1
    periods_days: tuple[float, ...] = (365.2425,)

    @property
    def n_parameters(self) -> int:
        return self.polynomial_degree + 1 + 2 * len(self.periods_days)


def decimal_time(dates: np.ndarray, origin: np.datetime64 | None = None, year_days: float = 365.2425) -> np.ndarray:
    dates = np.asarray(dates, dtype="datetime64[D]")
    if origin is None:
        origin = dates[0]
    return (dates - np.datetime64(origin, "D")).astype("timedelta64[D]").astype(float) / float(year_days)


def design_matrix(dates: np.ndarray, model: TimeModel, origin: np.datetime64 | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build one shared temporal design matrix for InSAR, groundwater and extensometer series.

    Polynomial coefficients follow y = c0 + c1*t + c2*t^2 + ... in decimal years.
    Each periodic term contributes sin(2*pi*t/P) and cos(2*pi*t/P).
    """
    dates = np.asarray(dates, dtype="datetime64[D]")
    t_year = decimal_time(dates, origin=origin)
    cols = [np.ones(len(dates), dtype=float)]
    for degree in range(1, model.polynomial_degree + 1):
        cols.append(t_year**degree)
    t_days = t_year * 365.2425
    for period in model.periods_days:
        angle = 2.0 * np.pi * t_days / float(period)
        cols.extend([np.sin(angle), np.cos(angle)])
    return np.column_stack(cols), t_year


def coefficient_indices(model: TimeModel) -> dict[str, object]:
    out: dict[str, object] = {
        "intercept": 0,
        "polynomial": list(range(1, model.polynomial_degree + 1)),
        "periodic": [],
    }
    j = model.polynomial_degree + 1
    periodic = []
    for period in model.periods_days:
        periodic.append({"period_days": period, "sin": j, "cos": j + 1})
        j += 2
    out["periodic"] = periodic
    return out


def evaluate_polynomial(beta: np.ndarray, t_year: np.ndarray, degree: int) -> np.ndarray:
    beta = np.asarray(beta)
    t = np.asarray(t_year, dtype=float)
    out = np.zeros(np.broadcast_shapes(beta.shape[:-1] + (1,), t.shape), dtype=float)
    for k in range(degree + 1):
        out = out + beta[..., k, None] * t**k
    return out
