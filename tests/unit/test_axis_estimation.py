import numpy as np

from step2point.core.shower import Shower
from step2point.metrics.spatial import estimate_shower_axis


def test_axis_estimation_returns_normalized_vector():
    s = Shower(
        1,
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 3.0]),
    )
    _, axis = estimate_shower_axis(s)
    assert np.isfinite(axis).all()
    assert np.isclose(np.linalg.norm(axis), 1.0)
