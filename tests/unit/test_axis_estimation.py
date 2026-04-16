import numpy as np

from step2point.core.shower import Shower
from step2point.metrics.spatial import estimate_shower_axis, longitudinal_radial_phi


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


def test_axis_override_is_honored():
    s = Shower(
        2,
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 3.0]),
    )
    _, axis = estimate_shower_axis(s, axis_override=[1.0, 0.0, 0.0])
    np.testing.assert_allclose(axis, np.array([1.0, 0.0, 0.0]))


def test_longitudinal_coordinate_can_be_shifted_to_zero():
    s = Shower(
        3,
        np.array([10.0, 11.0, 12.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
    )
    long, _, _ = longitudinal_radial_phi(s, axis_override=[1.0, 0.0, 0.0], longitudinal_origin="first_deposit")
    assert np.isclose(np.min(long), 0.0)
    np.testing.assert_allclose(long, np.array([0.0, 1.0, 2.0]))


def test_longitudinal_coordinate_can_be_kept_at_centroid_origin():
    s = Shower(
        4,
        np.array([10.0, 11.0, 12.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
    )
    long, _, _ = longitudinal_radial_phi(s, axis_override=[1.0, 0.0, 0.0], longitudinal_origin="centroid")
    np.testing.assert_allclose(long, np.array([-1.0, 0.0, 1.0]))


def test_longitudinal_coordinate_can_be_shifted_by_minimum_projection():
    s = Shower(
        5,
        np.array([10.0, 11.0, 12.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
    )
    long, _, _ = longitudinal_radial_phi(s, axis_override=[-1.0, 0.0, 0.0], longitudinal_origin="min_projection")
    np.testing.assert_allclose(long, np.array([2.0, 1.0, 0.0]))


def test_first_deposit_origin_ignores_large_radius_tail():
    s = Shower(
        6,
        np.array([5.0, 6.0, 7.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
    )
    long, _, _ = longitudinal_radial_phi(s, axis_override=[1.0, 0.0, 0.0], longitudinal_origin="first_deposit")
    np.testing.assert_allclose(long, np.array([0.0, 1.0, 2.0]))
