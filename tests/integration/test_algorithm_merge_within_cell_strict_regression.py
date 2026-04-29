from __future__ import annotations

import pytest

from tests.integration.algorithm_regression_helpers import (
    MERGE_REFERENCE,
    assert_showers_equal,
    assert_summary_equals,
    run_pipeline,
)

pytestmark = pytest.mark.strict_regression


def test_merge_within_cell_output_matches_reference(tmp_path):
    outdir = run_pipeline(tmp_path, "merge_within_cell")
    assert_summary_equals(outdir / "compression_summary_merge_within_cell.txt", "merge_within_cell")
    assert_showers_equal(MERGE_REFERENCE, outdir / "compressed_merge_within_cell.h5")
