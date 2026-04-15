from __future__ import annotations

from pathlib import Path

import numpy as np

from step2point.algorithms.identity import IdentityCompression
from step2point.algorithms.merge_within_cell import MergeWithinCell
from step2point.io.step2point_hdf5 import Step2PointHDF5Reader
from step2point.metrics.energy import aggregate_cell_energy, energy_ratio
from step2point.metrics.shower_shapes import shower_moments
from step2point.validation.sanity import ShowerSanityValidator

DATA = Path(__file__).resolve().parents[1] / "data" / "tiny_showers.h5"


def _read_showers():
    return list(Step2PointHDF5Reader(str(DATA)).iter_showers())


def test_identity_preserves_energy_cells_points_and_moments():
    validator = ShowerSanityValidator()
    for shower in _read_showers():
        out = IdentityCompression().compress(shower).shower
        assert np.isclose(energy_ratio(shower, out), 1.0, rtol=1e-12, atol=0.0)
        assert len(out.E) == len(shower.E)
        pre_cells, pre_e = aggregate_cell_energy(shower)
        post_cells, post_e = aggregate_cell_energy(out)
        assert np.array_equal(pre_cells, post_cells)
        assert np.allclose(pre_e, post_e)
        pre_m = shower_moments(shower)
        post_m = shower_moments(out)
        for key in pre_m:
            assert np.isclose(pre_m[key], post_m[key], rtol=1e-12, atol=1e-12)
        result = validator.run(shower, out)
        assert result.metrics["passed"]


def test_merge_within_cell_preserves_total_energy_and_cell_spectrum():
    validator = ShowerSanityValidator()
    algo = MergeWithinCell()
    for shower in _read_showers():
        out = algo.compress(shower).shower
        assert np.isclose(energy_ratio(shower, out), 1.0, rtol=1e-7, atol=0.0)
        assert len(out.E) <= len(shower.E)
        pre_cells, pre_e = aggregate_cell_energy(shower)
        post_cells, post_e = aggregate_cell_energy(out)
        assert np.array_equal(pre_cells, post_cells)
        assert np.allclose(pre_e, post_e)
        assert validator.run(shower, out).metrics["passed"]
