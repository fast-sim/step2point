import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

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


def test_run_pipeline_writes_debug_cluster_labels(tmp_path):
    outdir = tmp_path / "pipeline_debug"
    subprocess.run(
        [
            sys.executable,
            "examples/run_step2point_pipeline.py",
            "--input",
            str(DATA),
            "--algorithm",
            "merge_within_cell",
            "--output",
            str(outdir),
            "--debug-events",
            "0",
            "2",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    debug_h5 = outdir / "debug_merge_within_cell.h5"
    assert debug_h5.exists()
    with h5py.File(debug_h5, "r") as h5:
        assert bool(h5.attrs["debug_output"]) is True
        np.testing.assert_array_equal(h5.attrs["debug_event_indices"], np.asarray([0, 2], dtype=np.int32))
        steps = h5["steps"]
        assert "cluster_label" in steps
        cluster_label = np.asarray(steps["cluster_label"], dtype=np.int64)
        event_id = np.asarray(steps["event_id"], dtype=np.int32)
        cell_id = np.asarray(steps["cell_id"], dtype=np.uint64)
        assert len(cluster_label) == len(event_id) == len(cell_id)
        for shower_id in np.unique(event_id):
            mask = event_id == shower_id
            _, inverse = np.unique(cell_id[mask], return_inverse=True)
            np.testing.assert_array_equal(cluster_label[mask], inverse.astype(np.int64))
