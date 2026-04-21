from __future__ import annotations

import pytest

from tests.integration.algorithm_regression_helpers import (
    HDBSCAN_REFERENCE,
    assert_showers_equal,
    assert_summary_approx,
    run_pipeline,
)

pytest.importorskip("sklearn")


def test_hdbscan_clustering_output_matches_reference(tmp_path):
    outdir = run_pipeline(tmp_path, "hdbscan_clustering")
    assert_summary_approx(outdir / "compression_summary_hdbscan_clustering.txt", "hdbscan_clustering")
    assert_showers_equal(HDBSCAN_REFERENCE, outdir / "compressed_hdbscan_clustering.h5")
