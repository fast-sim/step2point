from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from step2point.io.step2point_hdf5 import Step2PointHDF5Reader

DATA = Path("tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV.h5")
MERGE_REFERENCE = Path("tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_merge_within_cell_reference.h5")


def run_pipeline(tmp_path: Path, algorithm: str) -> Path:
    outdir = tmp_path / f"pipeline_out_{algorithm}"
    env = dict(os.environ)
    env["PYTHONPATH"] = "src" if "PYTHONPATH" not in env else f"src:{env['PYTHONPATH']}"
    subprocess.run(
        [
            sys.executable,
            "examples/run_step2point_pipeline.py",
            "--input",
            str(DATA),
            "--algorithm",
            algorithm,
            "--output",
            str(outdir),
        ],
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
        np.testing.assert_allclose(lhs.x, rhs.x)
        np.testing.assert_allclose(lhs.y, rhs.y)
        np.testing.assert_allclose(lhs.z, rhs.z)
        np.testing.assert_allclose(lhs.E, rhs.E)
        if lhs.t is None or rhs.t is None:
            assert lhs.t is rhs.t
        else:
            np.testing.assert_allclose(lhs.t, rhs.t)
        if lhs.cell_id is None or rhs.cell_id is None:
            assert lhs.cell_id is rhs.cell_id
        else:
            np.testing.assert_array_equal(lhs.cell_id, rhs.cell_id)
        if lhs.pdg is None or rhs.pdg is None:
            assert lhs.pdg is rhs.pdg
        else:
            np.testing.assert_array_equal(lhs.pdg, rhs.pdg)
