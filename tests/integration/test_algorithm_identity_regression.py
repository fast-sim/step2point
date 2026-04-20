from __future__ import annotations

from step2point.io.step2point_hdf5 import Step2PointHDF5Reader
from tests.integration.algorithm_regression_helpers import (
    DATA,
    assert_showers_equal,
    assert_summary_equals,
    run_pipeline,
)


def test_run_step2point_pipeline_writes_identity_hdf5(tmp_path):
    outdir = run_pipeline(tmp_path, "identity")

    output_h5 = outdir / "compressed_identity.h5"
    summary = outdir / "compression_summary_identity.txt"
    assert output_h5.exists()
    assert summary.exists()
    assert_summary_equals(summary, "identity")

    showers = list(Step2PointHDF5Reader(str(output_h5)).iter_showers())
    assert len(showers) > 0
    assert sum(shower.n_points for shower in showers) > 0


def test_identity_output_matches_input(tmp_path):
    outdir = run_pipeline(tmp_path, "identity")
    assert_showers_equal(DATA, outdir / "compressed_identity.h5")
