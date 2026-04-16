import numpy as np

from step2point.core.shower import Shower
from step2point.validation.observables import aggregate_observables, compute_shower_observables


def test_compute_shower_observables_returns_expected_keys():
    shower = Shower(
        shower_id=3,
        x=np.array([0.0, 1.0, 2.0]),
        y=np.array([0.0, 0.0, 0.0]),
        z=np.array([0.0, 0.0, 0.0]),
        E=np.array([1.0, 2.0, 3.0]),
    )
    observables = compute_shower_observables(shower, axis_override=[1.0, 0.0, 0.0])
    assert set(("long_profile", "r_profile", "phi_profile", "mean_long", "var_long", "total_energy", "num_steps")).issubset(
        observables
    )
    assert observables["num_steps"] == 3
    assert observables["total_energy"] == 6.0


def test_aggregate_observables_collects_scalar_metrics():
    shower = Shower(
        shower_id=4,
        x=np.array([0.0, 1.0]),
        y=np.array([0.0, 0.0]),
        z=np.array([0.0, 0.0]),
        E=np.array([1.0, 1.0]),
    )
    rows = [compute_shower_observables(shower, axis_override=[1.0, 0.0, 0.0]) for _ in range(2)]
    summary = aggregate_observables(rows)
    assert len(summary["mean_long"]) == 2
    assert summary["num_steps"] == [2.0, 2.0]
