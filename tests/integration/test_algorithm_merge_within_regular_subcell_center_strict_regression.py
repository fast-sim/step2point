from __future__ import annotations

import pytest

from tests.integration.algorithm_regression_helpers import (
    REGULAR_SUBCELL_CENTER_REFERENCE,
    assert_showers_equal,
    assert_summary_equals,
    find_odd_xml,
    run_pipeline,
)

pytestmark = [pytest.mark.odd_geometry, pytest.mark.strict_regression]


def test_merge_within_regular_subcell_center_3x3_output_matches_reference(tmp_path):
    outdir = run_pipeline(
        tmp_path,
        "merge_within_regular_subcell",
        extra_args=[
            "--compact-xml",
            str(find_odd_xml()),
            "--collection-name",
            "ECalBarrelCollection",
            "--grid-x",
            "3",
            "--grid-y",
            "3",
            "--position-mode",
            "center",
        ],
    )
    assert_summary_equals(
        outdir / "compression_summary_merge_within_regular_subcell.txt",
        "merge_within_regular_subcell_center_3x3",
    )
    assert_showers_equal(
        REGULAR_SUBCELL_CENTER_REFERENCE,
        outdir / "compressed_merge_within_regular_subcell.h5",
    )
