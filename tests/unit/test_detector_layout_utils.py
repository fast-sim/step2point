from __future__ import annotations

import numpy as np

from step2point.core.shower import Shower
from step2point.vis.detector_layout_utils import (
    PROJECTIONS,
    WorldBounds,
    filter_geometry_to_bounds,
    overlay_color_spec,
    resolve_overlay_bounds,
)


def test_world_bounds_point_mask_filters_on_all_three_axes():
    shower = Shower(
        shower_id=0,
        x=np.array([0.0, 2.0, 4.0], dtype=np.float32),
        y=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        z=np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        E=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )
    bounds = WorldBounds(xlim=(1.0, 4.0), ylim=(15.0, 25.0), zlim=(-0.5, 0.5))
    mask = bounds.point_mask(shower)
    np.testing.assert_array_equal(mask, np.array([False, True, False]))


def test_world_bounds_returns_projection_specific_bounds():
    bounds = WorldBounds(xlim=(-1.0, 1.0), ylim=(2.0, 3.0), zlim=(4.0, 5.0))
    assert bounds.bounds_for_projection(PROJECTIONS["xy"]) == ((-1.0, 1.0), (2.0, 3.0))
    assert bounds.bounds_for_projection(PROJECTIONS["xz"]) == ((-1.0, 1.0), (4.0, 5.0))
    assert bounds.bounds_for_projection(PROJECTIONS["zy"]) == ((4.0, 5.0), (2.0, 3.0))


def test_filter_geometry_to_bounds_keeps_only_intersecting_primitives():
    segments = [
        np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        np.array([[5.0, 5.0], [6.0, 6.0]], dtype=np.float64),
    ]
    polygons = [
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        np.array([[5.0, 5.0], [6.0, 5.0], [6.0, 6.0]], dtype=np.float64),
    ]
    filtered_segments, filtered_polygons = filter_geometry_to_bounds(
        segments,
        polygons,
        WorldBounds(xlim=(-1.0, 2.0), ylim=(-1.0, 2.0)),
        PROJECTIONS["xy"],
    )
    assert len(filtered_segments) == 1
    assert len(filtered_polygons) == 1


def test_world_bounds_resolved_uses_user_limits_or_fallbacks():
    bounds = WorldBounds.resolved(
        xlim=None,
        ylim=(2.0, 3.0),
        zlim=None,
        fallback_x=(-1.0, 1.0),
        fallback_y=(0.0, 1.0),
        fallback_z=(4.0, 5.0),
    )
    assert bounds.xlim == (-1.0, 1.0)
    assert bounds.ylim == (2.0, 3.0)
    assert bounds.zlim == (4.0, 5.0)


def test_presentation_overlay_bounds_do_not_inherit_detector_fallbacks():
    bounds = resolve_overlay_bounds(
        WorldBounds(xlim=(-250.0, 250.0)),
        fallback_x=(-10.0, 10.0),
        fallback_y=(1307.0, 1309.0),
        fallback_z=(-5.0, 5.0),
        presentation_single_layer=True,
    )
    assert bounds == WorldBounds(xlim=(-250.0, 250.0))


def test_debug_overlay_bounds_inherit_detector_fallbacks():
    bounds = resolve_overlay_bounds(
        WorldBounds(xlim=(-2.0, 2.0)),
        fallback_x=(-10.0, 10.0),
        fallback_y=(1307.0, 1309.0),
        fallback_z=(-5.0, 5.0),
        presentation_single_layer=False,
    )
    assert bounds == WorldBounds(xlim=(-2.0, 2.0), ylim=(1307.0, 1309.0), zlim=(-5.0, 5.0))


def test_overlay_color_spec_uses_cluster_labels_when_available():
    shower = Shower(
        shower_id=0,
        x=np.array([0.0, 1.0, 2.0], dtype=np.float32),
        y=np.array([0.0, 1.0, 2.0], dtype=np.float32),
        z=np.array([0.0, 1.0, 2.0], dtype=np.float32),
        E=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        metadata={"cluster_label": np.array([10, 20, 10], dtype=np.int64)},
    )
    values, cmap = overlay_color_spec(shower, np.array([True, True, True]))
    assert values.shape == (3, 4)
    np.testing.assert_allclose(values[0], values[2])
    assert not np.allclose(values[0], values[1])
    assert cmap is None


def test_overlay_color_spec_falls_back_to_log_energy():
    shower = Shower(
        shower_id=0,
        x=np.array([0.0, 1.0], dtype=np.float32),
        y=np.array([0.0, 1.0], dtype=np.float32),
        z=np.array([0.0, 1.0], dtype=np.float32),
        E=np.array([1.0, 10.0], dtype=np.float32),
    )
    values, cmap = overlay_color_spec(shower, np.array([True, True]))
    np.testing.assert_allclose(values, np.array([0.0, 1.0]))
    assert cmap == "inferno"
