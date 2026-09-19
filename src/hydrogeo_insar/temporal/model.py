from __future__ import annotations

from dataclasses import dataclass

import numpy as np

YEAR_DAYS = 365.2425


@dataclass(frozen=True)
class TimeModel:
    """Linear time-function model shared by all modules.

    Low-frequency terms are polynomial terms plus optional continuous
    piecewise-linear hinges. Periodic terms are appended last.
    """

    polynomial_degree: int = 1
    periods_days: tuple[float, ...] = (YEAR_DAYS,)
    polyline_knots: tuple[str, ...] = ()

    @property
    def n_parameters(self) -> int:
        return self.polynomial_degree + 1 + len(self.polyline_knots) + 2 * len(self.periods_days)


def decimal_time(
    dates: np.ndarray,
    origin: np.datetime64 | None = None,
    year_days: float = YEAR_DAYS,
) -> np.ndarray:
    dates = np.asarray(dates, dtype="datetime64[D]")
    if origin is None:
        origin = dates[0]
    return (dates - np.datetime64(origin, "D")).astype("timedelta64[D]").astype(float) / float(year_days)


def design_matrix(
    dates: np.ndarray,
    model: TimeModel,
    origin: np.datetime64 | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build polynomial + polyline + periodic design matrix.

    Polynomial terms follow y = c0 + c1*t + c2*t^2 + ... in decimal years.
    A polyline knot adds max(0, t - t_k), allowing slope changes while keeping
    the low-frequency curve continuous. Each periodic term contributes sine
    and cosine coefficients.
    """
    dates = np.asarray(dates, dtype="datetime64[D]")
    if origin is None:
        origin = dates[0]
    origin = np.datetime64(origin, "D")
    t_year = decimal_time(dates, origin=origin)

    cols = [np.ones(len(dates), dtype=float)]
    for degree in range(1, model.polynomial_degree + 1):
        cols.append(t_year**degree)

    for knot in model.polyline_knots:
        tk = float(decimal_time(np.asarray([np.datetime64(knot, "D")]), origin=origin)[0])
        cols.append(np.maximum(0.0, t_year - tk))

    t_days = t_year * YEAR_DAYS
    for period in model.periods_days:
        angle = 2.0 * np.pi * t_days / float(period)
        cols.extend([np.sin(angle), np.cos(angle)])

    return np.column_stack(cols), t_year


def coefficient_indices(model: TimeModel) -> dict[str, object]:
    out: dict[str, object] = {
        "intercept": 0,
        "polynomial": list(range(1, model.polynomial_degree + 1)),
        "polyline": [],
        "periodic": [],
    }
    j = model.polynomial_degree + 1

    pl = []
    for knot in model.polyline_knots:
        pl.append({"knot": str(knot), "index": j})
        j += 1
    out["polyline"] = pl

    periodic = []
    for period in model.periods_days:
        periodic.append({"period_days": period, "sin": j, "cos": j + 1})
        j += 2
    out["periodic"] = periodic
    return out


def low_frequency_row(
    date: np.datetime64 | str,
    model: TimeModel,
    origin: np.datetime64,
) -> np.ndarray:
    """Design row containing only low-frequency terms."""
    row, _ = design_matrix(np.asarray([np.datetime64(date, "D")]), model, origin=origin)
    row = row[0]
    idx = coefficient_indices(model)
    for info in idx["periodic"]:
        row[int(info["sin"])] = 0.0
        row[int(info["cos"])] = 0.0
    return row


def evaluate_low_frequency(
    beta: np.ndarray,
    dates: np.ndarray,
    model: TimeModel,
    origin: np.datetime64,
) -> np.ndarray:
    """Evaluate only polynomial/polyline terms for one or many parameter vectors."""
    dates = np.asarray(dates, dtype="datetime64[D]")
    X, _ = design_matrix(dates, model, origin=origin)
    idx = coefficient_indices(model)
    for info in idx["periodic"]:
        X[:, int(info["sin"])] = 0.0
        X[:, int(info["cos"])] = 0.0
    return np.asarray(beta) @ X.T


def calendar_year_knots(start: np.datetime64, end: np.datetime64) -> tuple[str, ...]:
    """January-1 knots strictly inside [start, end]."""
    start = np.datetime64(start, "D")
    end = np.datetime64(end, "D")
    y0 = int(str(start)[:4])
    y1 = int(str(end)[:4])
    knots = []
    for year in range(y0 + 1, y1 + 1):
        d = np.datetime64(f"{year}-01-01", "D")
        if start < d < end:
            knots.append(str(d))
    return tuple(knots)


def evaluate_polynomial(beta: np.ndarray, t_year: np.ndarray, degree: int) -> np.ndarray:
    beta = np.asarray(beta)
    t = np.asarray(t_year, dtype=float)
    out = np.zeros(np.broadcast_shapes(beta.shape[:-1] + (1,), t.shape), dtype=float)
    for k in range(degree + 1):
        out = out + beta[..., k, None] * t**k
    return out
