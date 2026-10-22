import numpy as np
import pytest

from step2point.algorithms.merge_within_regular_subcell import MergeWithinRegularSubcell
from step2point.core.shower import Shower
from step2point.geometry.dd4hep.factory_geometry import BarrelLayerGeometry, BarrelLayout


def _simple_layout(collection_name="TestCollection", det_id=1) -> BarrelLayout:
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
        collection_name=[collection_name],
        detector_name="TestBarrel",
        det_id=det_id,
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
        cell_id_encoding="system:4,module:4,layer:4,x:16:-8,y:-8",
        layers=(layer,),
    )


def _cell_id(module: int, layer: int, cell_x: int, cell_y: int, system: int = 1) -> np.uint64:
    value = (
        (system & 0xF)
        | ((module & 0xF) << 4)
        | ((layer & 0xF) << 8)
        | ((cell_x & 0xFFFF) << 16)
        | ((cell_y & 0xFF) << 32)
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

    shower.metadata = {
        "subdetector": np.zeros(4, dtype=np.int64),
        "subdetector_names": [
            "TestCollection",
        ],
    }

    result = MergeWithinRegularSubcell(
        layout=[_simple_layout()],
        collection_name=["TestCollection"],
        x_bins=[2],
        y_bins=[2]
    ).compress(shower)

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

    shower.metadata = {
        "subdetector": np.array(
            [0, 0, 0, 0],
            dtype=np.int64,
        ),
        "subdetector_names": [
            "TestCollection",
        ],
    }

    result = MergeWithinRegularSubcell(
        layout=[_simple_layout()],
        collection_name=["TestCollection"],
        x_bins=[2],
        y_bins=[2],
        position_mode=["center"],
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
        MergeWithinRegularSubcell(layout=[_simple_layout()]).compress(shower)
    except ValueError as exc:
        assert "cell_id" in str(exc)
    else:
        raise AssertionError("Expected MergeWithinRegularSubcell to require cell_id.")


def test_regular_grid_weighted_splits_one_cell_into_four_subcells_multi_collections():
    layout_ecal = _simple_layout(collection_name="EcalCollection", det_id=1)
    layout_hcal = _simple_layout(collection_name="HcalCollection", det_id=2)

    shower = Shower(
        1,
        x=np.array([0.9, 0.9, -0.9, -0.9], dtype=np.float32),
        y=np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32),
        z=np.array([-0.9, 0.9, -0.9, 0.9], dtype=np.float32),
        E=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        cell_id=np.array(
            [
                _cell_id(1, 1, 0, 0, system=1), # ECalBarrelCollection
                _cell_id(1, 1, 0, 0, system=1),
                _cell_id(1, 1, 0, 0, system=2), # HCalBarrelCollection
                _cell_id(1, 1, 0, 0, system=2),
            ],
            dtype=np.uint64,
        ),
    )

    shower.metadata = {
        "subdetector": np.array(
            [0, 0, 1, 1],
            dtype=np.int64,
        ),
        "subdetector_names": [
            "EcalCollection",
            "HcalCollection",
        ],
    }

    result = MergeWithinRegularSubcell(
        layout=[layout_ecal, layout_hcal],
        collection_name=["EcalCollection", "HcalCollection"],
        x_bins=[2, 2],
        y_bins=[2, 2],
        position_mode=["weighted", "weighted"],
    ).compress(shower)

    assert result.shower.n_points == 4
    assert result.shower.total_energy == shower.total_energy
    assert len(np.unique(result.shower.cell_id)) == 2
    np.testing.assert_allclose(np.sort(result.shower.E), [1.0, 2.0, 3.0, 4.0])


def test_regular_grid_center_mode_places_outputs_at_subcell_centers_multi_collections():
    layout_ecal = _simple_layout(collection_name="ECalBarrelCollection", det_id=1)
    layout_hcal = _simple_layout(collection_name="HCalBarrelCollection", det_id=2)

    shower = Shower(
        1,
        x=np.array([0.9, 0.9, -0.9, -0.9, 0.9, 0.9, -0.9, -0.9], dtype=np.float32),
        y=np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32),
        z=np.array([-0.9, 0.9, -0.9, 0.9, -0.9, 0.9, -0.9, 0.9], dtype=np.float32),
        E=np.ones(8, dtype=np.float32),
        cell_id=np.array(
            [
                _cell_id(1, 1, 0, 0, system=1), # ECalBarrelCollection
                _cell_id(1, 1, 0, 0, system=1),
                _cell_id(1, 1, 0, 0, system=1),
                _cell_id(1, 1, 0, 0, system=1),
                _cell_id(1, 1, 0, 0, system=2), # HCalBarrelCollection
                _cell_id(1, 1, 0, 0, system=2),
                _cell_id(1, 1, 0, 0, system=2),
                _cell_id(1, 1, 0, 0, system=2),
            ],
            dtype=np.uint64,
        ),
    )

    shower.metadata = {
        "subdetector": np.array(
            [0, 0, 0, 0, 1, 1, 1, 1],
            dtype=np.int64,
        ),
        "subdetector_names": [
            "ECalBarrelCollection",
            "HCalBarrelCollection",
        ],
    }

    result = MergeWithinRegularSubcell(
        layout=[layout_ecal, layout_hcal],
        collection_name=["ECalBarrelCollection", "HCalBarrelCollection"],
        x_bins=[2, 2],
        y_bins=[2, 2],
        position_mode=["center", "center"],
    ).compress(shower)

    assert result.shower.n_points == 8

    np.testing.assert_allclose(
        np.sort(result.shower.x),
        [-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5],
    )

    np.testing.assert_allclose(
        np.sort(result.shower.z),
        [-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5],
    )

    np.testing.assert_allclose(
        result.shower.y,
        np.full(8, 10.0),
    )


def test_regular_grid_requires_cell_id_multi_collections():
    layout_ecal = _simple_layout(collection_name="ECalBarrelCollection", det_id=1)
    layout_hcal = _simple_layout(collection_name="HCalBarrelCollection", det_id=2)

    shower = Shower(
        1,
        x=np.array([0.0], dtype=np.float32),
        y=np.array([10.0], dtype=np.float32),
        z=np.array([0.0], dtype=np.float32),
        E=np.array([1.0], dtype=np.float32),
    )

    try:
        MergeWithinRegularSubcell(
            layout=[layout_ecal, layout_hcal],
            x_bins=[2, 2],
            y_bins=[2, 2],
            position_mode=["center", "center"],
        ).compress(shower)
    except ValueError as exc:
        assert "cell_id" in str(exc)
    else:
        raise AssertionError("Expected MergeWithinRegularSubcell to require cell_id.")



''' Additional Multi-collection tests '''
CORNERS_X = [0.9, 0.9, -0.9, -0.9]
CORNERS_Z = [-0.9, 0.9, -0.9, 0.9]

def _algo(position_mode="weighted"):
    """Build a MergeWithinRegularSubcell Object for two collections (Ecal, Hcal) with a 2x2 sub-grid"""
    return MergeWithinRegularSubcell(
        layout=[_simple_layout("EcalCollection", det_id=1), _simple_layout("HcalCollection", det_id=2)],
        collection_name=["EcalCollection", "HcalCollection"],
        x_bins=[2, 2],
        y_bins=[2, 2],
        position_mode=[position_mode, position_mode],
    )

def _shower(x, z, E, cell_ids, subdetector, names=("EcalCollection", "HcalCollection")):
    """Build a Shower on the y=10 sensitive plane, with per-hit subdetector ids and names in metadata"""
    n = len(E)
    s = Shower(
        1,
        x=np.asarray(x, dtype=np.float32),
        y=np.full(n, 10.0, dtype=np.float32),
        z=np.asarray(z, dtype=np.float32),
        E=np.asarray(E, dtype=np.float32),
        cell_id=np.asarray(cell_ids, dtype=np.uint64),
    )
    s.metadata = {"subdetector": np.asarray(subdetector, dtype=np.int64), "subdetector_names": list(names)}
    return s

@pytest.mark.parametrize("position_mode", ["weighted", "center"])
def test_same_cell_id_in_different_subdetectors_is_not_merged(position_mode):
    """Check to ensure hits with identical cell_id and sub-cell position stay separate if they belong to different
    subdetectors"""
    same = _cell_id(1, 1, 0, 0, system=1)  # identical value in both collections
    shower = _shower(
        CORNERS_X * 2, CORNERS_Z * 2, [1, 2, 3, 4, 10, 20, 30, 40],
        [same] * 8, [0] * 4 + [1] * 4,
    )
    result = _algo(position_mode).compress(shower)
    assert result.shower.n_points == 8
    np.testing.assert_allclose(np.sort(result.shower.E), [1, 2, 3, 4, 10, 20, 30, 40])
    np.testing.assert_allclose(result.shower.total_energy, shower.total_energy)

def test_hits_in_same_subdetector_and_subcell_still_merge():
    """Ensure Hits in the same subdetector and sub-cell are merged"""
    same = _cell_id(1, 1, 0, 0, system=1)
    shower = _shower([0.9] * 4, [-0.9] * 4, [1, 2, 10, 20], [same] * 4, [0, 0, 1, 1])
    result = _algo().compress(shower)
    assert result.shower.n_points == 2
    np.testing.assert_allclose(np.sort(result.shower.E), [3, 30])


def test_hits_outside_configured_collections_are_never_merged():
    """Check to ensure that hits from a subdetector not configured for clustering pass through unmerged,
    even when they share a cell_id and position, while configured collections still merge"""
    same = _cell_id(1, 1, 0, 0, system=1)
    shower = _shower(
        [0.9] * 4, [-0.9] * 4, [1, 2, 5, 6], [same] * 4, [0, 0, 2, 2],
        names=("EcalCollection", "HcalCollection", "Other"),
    )
    result = _algo().compress(shower)
    assert result.shower.n_points == 3  # 1 merged pair + 2 untouched
    np.testing.assert_allclose(np.sort(result.shower.E), [3, 5, 6])

def test_output_is_ordered_by_cell_id_when_cell_ids_do_not_collide():
    """Without collisions, output stays sorted by cell_id regardless of subdetector order.
    The strict regression references compare points in order, so the key must not reorder them"""
    # Ordering contract: the strict regression references compare points in order.
    shower = _shower(
        [10.0, 2.0], [0.0, 0.0], [1.0, 2.0],
        [_cell_id(1, 1, 5, 0, system=1), _cell_id(1, 1, 1, 0, system=2)],
        [0, 1],
    )
    out = _algo().compress(shower).shower
    assert np.all(np.diff(out.cell_id.astype(np.int64)) >= 0)