from __future__ import annotations

import pytest

from tests.integration.algorithm_regression_helpers import (
    DATA,
    ODD_BARREL_ENCODING,
    assert_energy_conserved_against_input,
    assert_summary_fields,
    assert_total_points_in_range,
    run_pipeline,
)

pytestmark = pytest.mark.loose_regression


def test_hdbscan_output_matches_loose_regression_bounds_with_time(tmp_path):
    outdir = run_pipeline(
        tmp_path,
        "hdbscan",
        extra_args=["--use-time", "--hdbscan-cell-id-encoding", ODD_BARREL_ENCODING],
    )

    output_h5 = outdir / "compressed_hdbscan.h5"
    summary_path = outdir / "compression_summary_hdbscan.txt"

    assert_summary_fields(
        summary_path,
        expected_exact={
            "compression_stats": "10",
            "validation_results": "30",
            "total_n_points_before": "35820",
            "output_hdf5": "compressed_hdbscan.h5",
        },
        expected_numeric_ranges={
            "mean_n_points_before": (3582.0, 3582.0),
            "mean_n_points_after": (265.0, 280.0),
            "mean_compression_ratio": (0.073, 0.078),
            "total_n_points_after": (2650.0, 2800.0),
            "total_compression_ratio": (0.073, 0.078),
        },
    )
    assert_total_points_in_range(output_h5, lower=2600, upper=2800)
    assert_energy_conserved_against_input(DATA, output_h5)

def test_hdbscan_output_matches_loose_regression_bounds_without_time(tmp_path):
    outdir = run_pipeline(
        tmp_path,
        "hdbscan",
        extra_args=["--hdbscan-cell-id-encoding", ODD_BARREL_ENCODING],
    )

    output_h5 = outdir / "compressed_hdbscan.h5"
    summary_path = outdir / "compression_summary_hdbscan.txt"

    assert_summary_fields(
        summary_path,
        expected_exact={
            "compression_stats": "10",
            "validation_results": "30",
            "total_n_points_before": "35820",
            "output_hdf5": "compressed_hdbscan.h5",
        },
        expected_numeric_ranges={
            "mean_n_points_before": (3582.0, 3582.0),
            "mean_n_points_after": (260.0, 280.0),
            "mean_compression_ratio": (0.072, 0.078),
            "total_n_points_after": (2600.0, 2800.0),
            "total_compression_ratio": (0.072, 0.078),
        },
    )
    assert_total_points_in_range(output_h5, lower=2600, upper=2800)
    assert_energy_conserved_against_input(DATA, output_h5)
