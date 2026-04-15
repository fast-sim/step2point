import numpy as np

from step2point.core.shower import Shower
from step2point.metrics.energy import aggregate_cell_energy


def test_aggregate_cell_energy():
    s = Shower(
        1,
        np.array([0, 1, 2]),
        np.array([0, 0, 0]),
        np.array([1, 2, 3]),
        np.array([1.0, 2.0, 3.0]),
        cell_id=np.array([11, 11, 12], dtype=np.uint64),
    )
    cells, energy = aggregate_cell_energy(s)
    assert cells.tolist() == [11, 12]
    assert np.allclose(energy, [3.0, 3.0])
