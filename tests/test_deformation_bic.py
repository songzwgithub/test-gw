import numpy as np

from hydrogeo_insar.deformation.decompose import (
    _bic_classes,
    _bic_from_rss,
)


def test_bic_nested_models_prefers_lower_bic():
    n = np.array([100.0])
    linear = _bic_from_rss(
        np.array([1000.0]),
        n,
        4,
    )
    quadratic = _bic_from_rss(
        np.array([700.0]),
        n,
        5,
    )
    delta = linear - quadratic
    assert delta[0] > 0


def test_bic_evidence_classes():
    delta = np.array([-5.0, -1.0, 0.0, 1.0, 5.0, np.nan])
    preferred, evidence = _bic_classes(delta, 2.0)

    assert preferred.tolist() == [1, 1, 1, 2, 2, 0]
    assert evidence.tolist() == [1, 2, 2, 2, 3, 0]
