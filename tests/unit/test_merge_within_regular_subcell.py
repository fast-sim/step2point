import numpy as np

from step2point.algorithms.merge_within_regular_subcell import MergeWithinRegularSubcell
from step2point.core.shower import Shower
from step2point.geometry.dd4hep.factory_geometry import BarrelLayerGeometry, BarrelLayout


def _simple_layout() -> BarrelLayout:
    layer = BarrelLayerGeometry(
        layer_index=1,
        layer_center_radius_mm=10.0,
        half_thickness_mm=1.0,
        sensitive_radius_mm=10.0,
        sensitive_half_thickness_mm=0.5,
        half_tangent_mm=2.0,
        half_z_mm=2.0,
        pitch_tangent_mm=2.0,
        pitch_z_mm=2.0,
        modules=1,
        module_angles_rad=(0.0,),
        module_centers_xy_mm=((0.0, 10.0),),
    )
    return BarrelLayout(
        collection_name="TestCollection",
        detector_name="TestBarrel",
        readout_xml_path="",
        detector_xml_path="",
        segmentation_type="CartesianGridXY",
        numsides=1,
        det_z_mm=4.0,
        rmin_mm=9.0,
        gap_mm=0.0,
        total_thickness_mm=2.0,
        inner_angle_rad=2.0 * np.pi,
        inner_face_length_mm=4.0,
        outer_face_length_mm=4.0,
        sect_center_radius_mm=10.0,
        envelope_rotation_z_rad=0.0,
        pitch_tangent_mm=2.0,
        pitch_z_mm=2.0,
        cell_id_encoding="module:4,layer:4,x:16:-8,y:-8",
        layers=(layer,),
    )


def _cell_id(module: int, layer: int, cell_x: int, cell_y: int) -> np.uint64:
    value = (
        (module & 0xF)
        | ((layer & 0xF) << 4)
        | ((cell_x & 0xFF) << 16)
        | ((cell_y & 0xFF) << 24)
    )
    return np.uint64(value)


def test_regular_grid_weighted_splits_one_cell_into_four_subcells():
    shower = Shower(
        1,
        x=np.array([0.9, 0.9, -0.9, -0.9], dtype=np.float32),
        y=np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32),
        z=np.array([-0.9, 0.9, -0.9, 0.9], dtype=np.float32),
        E=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        cell_id=np.array([_cell_id(1, 1, 0, 0)] * 4, dtype=np.uint64),
    )

    result = MergeWithinRegularSubcell(layout=_simple_layout(), x_bins=2, y_bins=2).compress(shower)

    assert result.shower.n_points == 4
    assert result.shower.total_energy == shower.total_energy
    assert len(np.unique(result.shower.cell_id)) == 1
    np.testing.assert_allclose(np.sort(result.shower.E), [1.0, 2.0, 3.0, 4.0])


def test_regular_grid_center_mode_places_outputs_at_subcell_centers():
    shower = Shower(
        1,
        x=np.array([0.9, 0.9, -0.9, -0.9], dtype=np.float32),
        y=np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32),
        z=np.array([-0.9, 0.9, -0.9, 0.9], dtype=np.float32),
        E=np.ones(4, dtype=np.float32),
        cell_id=np.array([_cell_id(1, 1, 0, 0)] * 4, dtype=np.uint64),
    )

    result = MergeWithinRegularSubcell(
        layout=_simple_layout(),
        x_bins=2,
        y_bins=2,
        position_mode="center",
    ).compress(shower)

    np.testing.assert_allclose(np.sort(result.shower.x), [-0.5, -0.5, 0.5, 0.5])
    np.testing.assert_allclose(np.sort(result.shower.z), [-0.5, -0.5, 0.5, 0.5])
    np.testing.assert_allclose(result.shower.y, np.full(4, 10.0))


def test_regular_grid_requires_cell_id():
    shower = Shower(
        1,
        x=np.array([0.0], dtype=np.float32),
        y=np.array([10.0], dtype=np.float32),
        z=np.array([0.0], dtype=np.float32),
        E=np.array([1.0], dtype=np.float32),
    )

    try:
        MergeWithinRegularSubcell(layout=_simple_layout()).compress(shower)
    except ValueError as exc:
        assert "cell_id" in str(exc)
    else:
        raise AssertionError("Expected MergeWithinRegularSubcell to require cell_id.")
