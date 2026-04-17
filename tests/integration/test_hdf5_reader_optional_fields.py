from pathlib import Path

from step2point.io.step2point_hdf5 import Step2PointHDF5Reader

PION_DATA = Path(__file__).resolve().parents[1] / "data" / "ODD_pionM_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV.h5"


def test_hdf5_reader_allows_missing_optional_fields():
    shower = next(Step2PointHDF5Reader(str(PION_DATA)).iter_showers())
    assert shower.n_points > 0
    assert shower.cell_id is not None
    assert shower.t is not None
