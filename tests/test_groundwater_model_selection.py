import pandas as pd

from hydrogeo_insar.groundwater.field import _select_cv_best


def test_groundwater_cv_selects_minimum_full_rmse():
    table = pd.DataFrame(
        {
            "rank": [4, 5],
            "sigma_km": [20.0, 25.0],
            "ridge": [0.3, 0.3],
            "cv_rmse_m": [7.49, 7.74],
            "cv_harmonic_vector_rmse_m": [3.21, 3.20],
        }
    )
    idx = _select_cv_best(table)
    assert idx == 0
