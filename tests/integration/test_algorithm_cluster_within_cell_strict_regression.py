from __future__ import annotations

from tests.integration.algorithm_regression_helpers import (
    CLUSTER_WITHIN_CELL_REFERENCE,
    assert_showers_equal,
    assert_summary_equals,
    run_pipeline,
)


def test_cluster_within_cell_output_matches_reference(tmp_path):
    outdir = run_pipeline(tmp_path, "cluster_within_cell")
    assert_summary_equals(outdir / "compression_summary_cluster_within_cell.txt", "cluster_within_cell")
    assert_showers_equal(CLUSTER_WITHIN_CELL_REFERENCE, outdir / "compressed_cluster_within_cell.h5")
