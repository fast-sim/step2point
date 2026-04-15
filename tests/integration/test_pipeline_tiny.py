from pathlib import Path

from step2point.algorithms.identity import IdentityCompression
from step2point.algorithms.merge_within_cell import MergeWithinCell
from step2point.core.pipeline import Pipeline
from step2point.io.step2point_hdf5 import Step2PointHDF5Reader
from step2point.validation.conservation import CellCountRatioValidator, EnergyConservationValidator

DATA = Path(__file__).resolve().parents[1] / "data" / "tiny_showers.h5"


def test_pipeline_identity_tiny():
    reader = Step2PointHDF5Reader(str(DATA))
    report = Pipeline(reader, IdentityCompression(), [EnergyConservationValidator(), CellCountRatioValidator()]).run()
    assert len(report.compression_stats) == 3
    ratios = [row["energy_ratio"] for row in report.validation_results if row["validator"] == "energy_conservation"]
    assert all(abs(x - 1.0) < 1e-12 for x in ratios)


def test_pipeline_merge_tiny():
    reader = Step2PointHDF5Reader(str(DATA))
    report = Pipeline(reader, MergeWithinCell(), [EnergyConservationValidator(), CellCountRatioValidator()]).run()
    assert len(report.compression_stats) == 3
    for stat in report.compression_stats:
        assert stat["n_points_after"] <= stat["n_points_before"]
