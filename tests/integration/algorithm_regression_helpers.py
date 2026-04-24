from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from step2point.io.step2point_hdf5 import Step2PointHDF5Reader

DATA = Path("tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV.h5")
MERGE_REFERENCE = Path("tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_merge_within_cell_reference.h5")
REGULAR_SUBCELL_WEIGHTED_REFERENCE = Path(
    "tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_merge_within_regular_subcell_3x3_weighted_reference.h5"
)
REGULAR_SUBCELL_CENTER_REFERENCE = Path(
    "tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_merge_within_regular_subcell_3x3_center_reference.h5"
)
HDBSCAN_REFERENCE = Path(
    "tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_hdbscan_clustering_reference.h5"
)

FLOAT_STRICT_RTOL = 1e-7
FLOAT_STRICT_ATOL = 1e-10
FLOAT_LOOSE_RTOL = 0.0
FLOAT_LOOSE_ATOL = 1e-7


def find_odd_xml() -> Path:
    candidates = [
        Path("../OpenDataDetector/xml/OpenDataDetector.xml"),
        Path("OpenDataDetector/xml/OpenDataDetector.xml"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("OpenDataDetector compact XML not found in ../OpenDataDetector or OpenDataDetector.")


def run_pipeline(tmp_path: Path, algorithm: str, extra_args: list[str] | None = None) -> Path:
    outdir = tmp_path / f"pipeline_out_{algorithm}"
    env = dict(os.environ)
    env["PYTHONPATH"] = "src" if "PYTHONPATH" not in env else f"src:{env['PYTHONPATH']}"
    cmd = [
        sys.executable,
        "examples/run_step2point_pipeline.py",
        "--input",
        str(DATA),
        "--algorithm",
        algorithm,
        "--output",
        str(outdir),
    ]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(
        cmd,
        check=True,
        env=env,
    )
    return outdir


def assert_showers_equal(left_path: Path, right_path: Path) -> None:
    left = list(Step2PointHDF5Reader(str(left_path)).iter_showers())
    right = list(Step2PointHDF5Reader(str(right_path)).iter_showers())
    assert len(left) == len(right)

    for lhs, rhs in zip(left, right, strict=True):
        assert lhs.shower_id == rhs.shower_id
        np.testing.assert_allclose(lhs.x, rhs.x, rtol=FLOAT_STRICT_RTOL, atol=FLOAT_STRICT_ATOL)
        np.testing.assert_allclose(lhs.y, rhs.y, rtol=FLOAT_STRICT_RTOL, atol=FLOAT_STRICT_ATOL)
        np.testing.assert_allclose(lhs.z, rhs.z, rtol=FLOAT_STRICT_RTOL, atol=FLOAT_STRICT_ATOL)
        np.testing.assert_allclose(lhs.E, rhs.E, rtol=FLOAT_STRICT_RTOL, atol=FLOAT_STRICT_ATOL)
        if lhs.t is None or rhs.t is None:
            assert lhs.t is rhs.t
        else:
            np.testing.assert_allclose(lhs.t, rhs.t, rtol=FLOAT_STRICT_RTOL, atol=FLOAT_STRICT_ATOL)
        if lhs.cell_id is None or rhs.cell_id is None:
            assert lhs.cell_id is rhs.cell_id
        else:
            np.testing.assert_array_equal(lhs.cell_id, rhs.cell_id)
        if lhs.pdg is None or rhs.pdg is None:
            assert lhs.pdg is rhs.pdg
        else:
            np.testing.assert_array_equal(lhs.pdg, rhs.pdg)


def assert_summary_equals(summary_path: Path, case: str) -> None:
    expected_by_case = {
        "identity": (
            "compression_stats=10\n"
            "validation_results=30\n"
            "mean_n_points_before=3582.000000\n"
            "mean_n_points_after=3582.000000\n"
            "mean_compression_ratio=1.000000\n"
            "total_n_points_before=35820\n"
            "total_n_points_after=35820\n"
            "total_compression_ratio=1.000000\n"
            "output_hdf5=compressed_identity.h5\n"
        ),
        "merge_within_cell": (
            "compression_stats=10\n"
            "validation_results=30\n"
            "mean_n_points_before=3582.000000\n"
            "mean_n_points_after=360.400000\n"
            "mean_compression_ratio=0.100820\n"
            "total_n_points_before=35820\n"
            "total_n_points_after=3604\n"
            "total_compression_ratio=0.100614\n"
            "output_hdf5=compressed_merge_within_cell.h5\n"
        ),
        "merge_within_regular_subcell_weighted_3x3": (
            "compression_stats=10\n"
            "validation_results=30\n"
            "mean_n_points_before=3582.000000\n"
            "mean_n_points_after=655.000000\n"
            "mean_compression_ratio=0.183212\n"
            "total_n_points_before=35820\n"
            "total_n_points_after=6550\n"
            "total_compression_ratio=0.182859\n"
            "output_hdf5=compressed_merge_within_regular_subcell.h5\n"
        ),
        "merge_within_regular_subcell_center_3x3": (
            "compression_stats=10\n"
            "validation_results=30\n"
            "mean_n_points_before=3582.000000\n"
            "mean_n_points_after=655.000000\n"
            "mean_compression_ratio=0.183212\n"
            "total_n_points_before=35820\n"
            "total_n_points_after=6550\n"
            "total_compression_ratio=0.182859\n"
            "output_hdf5=compressed_merge_within_regular_subcell.h5\n"
        ),
        "hdbscan_clustering": (
            "compression_stats=10\n"
            "validation_results=30\n"
            "mean_n_points_before=3582.000000\n"
            "mean_n_points_after=266.400000\n"
            "mean_compression_ratio=0.074382\n"
            "total_n_points_before=35820\n"
            "total_n_points_after=2664\n"
            "total_compression_ratio=0.074372\n"
            "output_hdf5=compressed_hdbscan_clustering.h5\n"
        ),
    }
    expected = expected_by_case[case]
    assert summary_path.read_text() == expected


def parse_summary(summary_path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in summary_path.read_text().splitlines():
        if not line.strip():
            continue
        key, value = line.split("=", 1)
        parsed[key] = value
    return parsed


def assert_summary_fields(
    summary_path: Path,
    *,
    expected_exact: dict[str, str] | None = None,
    expected_numeric_ranges: dict[str, tuple[float, float]] | None = None,
) -> None:
    parsed = parse_summary(summary_path)

    if expected_exact:
        for key, expected in expected_exact.items():
            assert parsed[key] == expected

    if expected_numeric_ranges:
        for key, (lower, upper) in expected_numeric_ranges.items():
            value = float(parsed[key])
            assert lower <= value <= upper, f"{key}={value} outside expected range [{lower}, {upper}]"


def assert_energy_conserved_against_input(input_path: Path, output_path: Path) -> None:
    input_showers = list(Step2PointHDF5Reader(str(input_path)).iter_showers())
    output_showers = list(Step2PointHDF5Reader(str(output_path)).iter_showers())
    assert len(input_showers) == len(output_showers)

    for input_shower, output_shower in zip(input_showers, output_showers, strict=True):
        assert input_shower.shower_id == output_shower.shower_id
        np.testing.assert_allclose(
            np.sum(input_shower.E, dtype=np.float64),
            np.sum(output_shower.E, dtype=np.float64),
            rtol=FLOAT_LOOSE_RTOL,
            atol=FLOAT_LOOSE_ATOL,
        )


def assert_total_points_in_range(output_path: Path, *, lower: int, upper: int) -> None:
    showers = list(Step2PointHDF5Reader(str(output_path)).iter_showers())
    total_n_points = sum(shower.n_points for shower in showers)
    assert lower <= total_n_points <= upper, f"total_n_points={total_n_points} outside expected range [{lower}, {upper}]"
