from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from step2point.io.step2point_hdf5 import Step2PointHDF5Reader
from step2point.metrics.shower_shapes import weighted_moment
from step2point.metrics.spatial import estimate_shower_axis, longitudinal_radial_phi

DATA = Path("tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV.h5")
MERGE_REFERENCE = Path("tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_merge_within_cell_reference.h5")
REGULAR_SUBCELL_WEIGHTED_REFERENCE = Path(
    "tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_merge_within_regular_subcell_3x3_weighted_reference.h5"
)
REGULAR_SUBCELL_CENTER_REFERENCE = Path(
    "tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_merge_within_regular_subcell_3x3_center_reference.h5"
)
HDBSCAN_REFERENCE = Path(
    "tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_hdbscan_reference.h5"
)
ODD_BARREL_ENCODING = "system:8,barrel:3,module:4,stave:1,layer:6,slice:5,x:32:-16,y:-16"
CLUSTER_WITHIN_CELL_REFERENCE = Path(
    "tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_cluster_within_cell_reference.h5"
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


def _reference_origin_and_axis(shower) -> tuple[np.ndarray, np.ndarray]:
    primary = shower.primary or {}
    vertex = np.asarray(primary.get("vertex"), dtype=np.float64)
    momentum = np.asarray(primary.get("momentum"), dtype=np.float64)
    momentum_norm = np.linalg.norm(momentum)
    if vertex.shape == (3,) and momentum.shape == (3,) and np.all(np.isfinite(vertex)) and np.isfinite(momentum_norm):
        if momentum_norm > 0.0:
            return vertex, momentum / momentum_norm
    return estimate_shower_axis(shower)


def _energy_weighted_centroid(shower) -> np.ndarray:
    coordinates = np.column_stack((shower.x, shower.y, shower.z)).astype(np.float64)
    return np.average(coordinates, axis=0, weights=np.asarray(shower.E, dtype=np.float64))


def _normalized_profile_l1(
    reference_values: np.ndarray,
    reference_weights: np.ndarray,
    output_values: np.ndarray,
    output_weights: np.ndarray,
    *,
    bins: int,
) -> float:
    edges = np.histogram_bin_edges(reference_values, bins=bins).astype(np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_histogram, _ = np.histogram(reference_values, bins=edges, weights=reference_weights)
    output_histogram, _ = np.histogram(output_values, bins=edges, weights=output_weights)
    normalization = np.sum(np.abs(reference_histogram), dtype=np.float64)
    assert normalization > 0.0
    return float(np.sum(np.abs(reference_histogram - output_histogram), dtype=np.float64) / normalization)


def assert_reference_physics_observables_close(
    reference_path: Path,
    output_path: Path,
    *,
    max_total_point_fraction: float,
    max_shower_point_fraction: float,
    spatial_centroid_atol: float,
    time_centroid_atol: float,
    moment_rtol: float,
    max_profile_l1: float,
    profile_bins: int = 8,
) -> None:
    """Compare stable shower observables without requiring identical clusters."""
    reference_showers = list(Step2PointHDF5Reader(str(reference_path)).iter_showers())
    output_showers = list(Step2PointHDF5Reader(str(output_path)).iter_showers())
    assert len(reference_showers) == len(output_showers)

    reference_total_points = sum(shower.n_points for shower in reference_showers)
    output_total_points = sum(shower.n_points for shower in output_showers)
    total_point_fraction = abs(output_total_points - reference_total_points) / reference_total_points
    assert total_point_fraction <= max_total_point_fraction, (
        f"total point-count difference {total_point_fraction:.6g} exceeds {max_total_point_fraction:.6g}"
    )

    for reference, output in zip(reference_showers, output_showers, strict=True):
        assert reference.shower_id == output.shower_id
        point_fraction = abs(output.n_points - reference.n_points) / reference.n_points
        assert point_fraction <= max_shower_point_fraction, (
            f"shower {reference.shower_id} point-count difference {point_fraction:.6g} "
            f"exceeds {max_shower_point_fraction:.6g}"
        )

        reference_energy = np.asarray(reference.E, dtype=np.float64)
        output_energy = np.asarray(output.E, dtype=np.float64)
        np.testing.assert_allclose(
            np.sum(output_energy),
            np.sum(reference_energy),
            rtol=FLOAT_LOOSE_RTOL,
            atol=FLOAT_LOOSE_ATOL,
            err_msg=f"shower {reference.shower_id} total energy differs from reference",
        )
        np.testing.assert_allclose(
            _energy_weighted_centroid(output),
            _energy_weighted_centroid(reference),
            rtol=0.0,
            atol=spatial_centroid_atol,
            err_msg=f"shower {reference.shower_id} spatial centroid differs from reference",
        )

        origin, axis = _reference_origin_and_axis(reference)
        reference_longitudinal, reference_radial, _ = longitudinal_radial_phi(reference, centroid=origin, axis=axis)
        output_longitudinal, output_radial, _ = longitudinal_radial_phi(output, centroid=origin, axis=axis)
        reference_moments = {
            "longitudinal_m1": weighted_moment(reference_longitudinal, reference_energy, 1),
            "longitudinal_m2": weighted_moment(reference_longitudinal, reference_energy, 2),
            "radial_m1": weighted_moment(reference_radial, reference_energy, 1),
            "radial_m2": weighted_moment(reference_radial, reference_energy, 2),
        }
        output_moments = {
            "longitudinal_m1": weighted_moment(output_longitudinal, output_energy, 1),
            "longitudinal_m2": weighted_moment(output_longitudinal, output_energy, 2),
            "radial_m1": weighted_moment(output_radial, output_energy, 1),
            "radial_m2": weighted_moment(output_radial, output_energy, 2),
        }
        profiles = {
            "longitudinal": (reference_longitudinal, output_longitudinal),
            "radial": (reference_radial, output_radial),
        }

        assert (reference.t is None) == (output.t is None)
        if reference.t is not None and output.t is not None:
            reference_time = np.asarray(reference.t, dtype=np.float64)
            output_time = np.asarray(output.t, dtype=np.float64)
            np.testing.assert_allclose(
                np.average(output_time, weights=output_energy),
                np.average(reference_time, weights=reference_energy),
                rtol=0.0,
                atol=time_centroid_atol,
                err_msg=f"shower {reference.shower_id} time centroid differs from reference",
            )
            reference_moments.update(
                {
                    "time_m1": weighted_moment(reference_time, reference_energy, 1),
                    "time_m2": weighted_moment(reference_time, reference_energy, 2),
                }
            )
            output_moments.update(
                {
                    "time_m1": weighted_moment(output_time, output_energy, 1),
                    "time_m2": weighted_moment(output_time, output_energy, 2),
                }
            )
            profiles["time"] = (reference_time, output_time)

        for name, expected in reference_moments.items():
            np.testing.assert_allclose(
                output_moments[name],
                expected,
                rtol=moment_rtol,
                atol=FLOAT_LOOSE_ATOL,
                err_msg=f"shower {reference.shower_id} {name} differs from reference",
            )

        for name, (reference_values, output_values) in profiles.items():
            distance = _normalized_profile_l1(
                reference_values,
                reference_energy,
                output_values,
                output_energy,
                bins=profile_bins,
            )
            assert distance <= max_profile_l1, (
                f"shower {reference.shower_id} {name} profile L1 distance {distance:.6g} exceeds {max_profile_l1:.6g}"
            )


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
        "hdbscan": (
            "compression_stats=10\n"
            "validation_results=30\n"
            "mean_n_points_before=3582.000000\n"
            "mean_n_points_after=275.500000\n"
            "mean_compression_ratio=0.076907\n"
            "total_n_points_before=35820\n"
            "total_n_points_after=2755\n"
            "total_compression_ratio=0.076912\n"
            "output_hdf5=compressed_hdbscan.h5\n"
        ),
        "cluster_within_cell_agglomerative": (
            "compression_stats=10\n"
            "validation_results=30\n"
            "mean_n_points_before=3582.000000\n"
            "mean_n_points_after=783.500000\n"
            "mean_compression_ratio=0.219064\n"
            "total_n_points_before=35820\n"
            "total_n_points_after=7835\n"
            "total_compression_ratio=0.218733\n"
            "output_hdf5=compressed_cluster_within_cell_agglomerative.h5\n"
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
