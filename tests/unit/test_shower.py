import numpy as np

from step2point.core.shower import Shower


def test_shower_n_points_and_energy():
    shower = Shower(shower_id=1, x=np.array([0, 1]), y=np.array([0, 1]), z=np.array([0, 1]), E=np.array([1, 2]))
    assert shower.n_points == 2
    assert shower.total_energy == 3.0


def test_shower_arrays_are_contiguous_and_typed():
    shower = Shower(shower_id=1, x=[0, 1], y=[0, 1], z=[0, 1], E=[1, 2])
    assert shower.x.dtype == np.float32
    assert shower.E.flags["C_CONTIGUOUS"]
