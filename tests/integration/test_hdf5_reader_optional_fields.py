from pathlib import Path

import h5py
import numpy as np

from step2point.io.step2point_hdf5 import Step2PointHDF5Reader

PION_DATA = Path(__file__).resolve().parents[1] / "data" / "ODD_pionM_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV.h5"


def test_hdf5_reader_allows_missing_optional_fields():
    shower = next(Step2PointHDF5Reader(str(PION_DATA)).iter_showers())
    assert shower.n_points > 0
    assert shower.cell_id is not None
    assert shower.t is not None


def test_hdf5_reader_preserves_cluster_labels(tmp_path: Path):
    path = tmp_path / "debug.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["algorithm"] = "merge_within_cell"
        h5.attrs["debug_output"] = True
        steps = h5.create_group("steps")
        steps.create_dataset("event_id", data=np.array([0, 0, 1], dtype=np.int32))
        steps.create_dataset("energy", data=np.array([1.0, 2.0, 3.0], dtype=np.float32))
        steps.create_dataset(
            "position",
            data=np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], dtype=np.float32),
        )
        steps.create_dataset("cluster_label", data=np.array([4, 4, 9], dtype=np.int64))

    showers = list(Step2PointHDF5Reader(str(path)).iter_showers())
    assert len(showers) == 2
    assert showers[0].metadata["algorithm"] == "merge_within_cell"
    assert showers[0].metadata["debug_output"] is True
    np.testing.assert_array_equal(showers[0].metadata["cluster_label"], np.array([4, 4], dtype=np.int64))
    np.testing.assert_array_equal(showers[1].metadata["cluster_label"], np.array([9], dtype=np.int64))
