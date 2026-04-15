import numpy as np

from step2point.algorithms.identity import IdentityCompression
from step2point.core.shower import Shower


def test_identity_keeps_energy_and_point_count():
    s = Shower(1, np.array([0, 1]), np.array([0, 0]), np.array([1, 2]), np.array([2.0, 3.0]))
    result = IdentityCompression().compress(s)
    assert result.shower.total_energy == s.total_energy
    assert result.shower.n_points == s.n_points
