from __future__ import annotations

from tests.integration.algorithm_regression_helpers import (
    REGULAR_GRID_REFERENCE,
    assert_showers_equal,
    assert_summary_equals,
    find_odd_xml,
    run_pipeline,
)


def test_merge_within_regular_subcell_output_matches_reference(tmp_path):
    outdir = run_pipeline(
        tmp_path,
        "merge_within_regular_subcell",
        extra_args=[
            "--compact-xml",
            str(find_odd_xml()),
            "--collection-name",
            "ECalBarrelCollection",
            "--grid-x",
            "2",
            "--grid-y",
            "2",
            "--position-mode",
            "weighted",
        ],
    )
    assert_summary_equals(
        outdir / "compression_summary_merge_within_regular_subcell.txt",
        "merge_within_regular_subcell",
    )
    assert_showers_equal(
        REGULAR_GRID_REFERENCE,
        outdir / "compressed_merge_within_regular_subcell.h5",
    )
