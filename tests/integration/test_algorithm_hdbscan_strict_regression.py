from __future__ import annotations

import os

import pytest

from tests.integration.algorithm_regression_helpers import (
    HDBSCAN_REFERENCE,
    assert_showers_equal,
    assert_summary_equals,
    run_pipeline,
)

pytestmark = pytest.mark.strict_regression


@pytest.mark.skipif(
    os.environ.get("STEP2POINT_ENABLE_STRICT_HDBSCAN") != "1",
    reason="strict HDBSCAN regression is only enabled in the dedicated pinned CI job",
)
def test_hdbscan_clustering_output_matches_reference(tmp_path):
    outdir = run_pipeline(tmp_path, "hdbscan_clustering", extra_args=["--use-time"])
    assert_summary_equals(outdir / "compression_summary_hdbscan_clustering.txt", "hdbscan_clustering")
    assert_showers_equal(HDBSCAN_REFERENCE, outdir / "compressed_hdbscan_clustering.h5")
