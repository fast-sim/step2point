import numpy as np
import pytest

from step2point.algorithms.greedy_agglomerative import GreedyAgglomerativeClustering
from step2point.core.shower import Shower
from step2point.metrics.energy import energy_ratio


def _make_shower_same_cell_two_blobs() -> Shower:
    rng = np.random.default_rng(42)
    n_per_blob = 12
    x = np.concatenate(
        [
            rng.normal(0.0, 0.1, n_per_blob),
            rng.normal(4.0, 0.1, n_per_blob),
        ]
    ).astype(np.float32)
    y = np.concatenate(
        [
            rng.normal(0.0, 0.1, n_per_blob),
            rng.normal(0.0, 0.1, n_per_blob),
        ]
    ).astype(np.float32)
    z = rng.normal(10.0, 0.05, 2 * n_per_blob).astype(np.float32)
    e = (rng.exponential(0.5, 2 * n_per_blob) + 0.01).astype(np.float32)
    t = rng.normal(3.0, 0.1, 2 * n_per_blob).astype(np.float32)
    cell_id = np.full(2 * n_per_blob, 17, dtype=np.uint64)
    return Shower(shower_id=0, x=x, y=y, z=z, E=e, t=t, cell_id=cell_id)


def test_greedy_agglomerative_requires_cell_id():
    shower = Shower(
        shower_id=0,
        x=np.array([0.0, 1.0], dtype=np.float32),
        y=np.array([0.0, 0.0], dtype=np.float32),
        z=np.array([0.0, 0.0], dtype=np.float32),
        E=np.array([1.0, 1.0], dtype=np.float32),
    )
    algo = GreedyAgglomerativeClustering(max_link_distance=1.0, max_cluster_distance=2.0)
    with pytest.raises(ValueError, match="cell_id"):
        algo.compress(shower)


def test_greedy_agglomerative_does_not_mutate_input():
    shower = _make_shower_same_cell_two_blobs()
    x_orig = shower.x.copy()
    y_orig = shower.y.copy()
    z_orig = shower.z.copy()
    e_orig = shower.E.copy()
    t_orig = shower.t.copy()
    cell_id_orig = shower.cell_id.copy()

    algo = GreedyAgglomerativeClustering(max_link_distance=0.5, max_cluster_distance=1.5)
    algo.compress(shower)

    np.testing.assert_array_equal(shower.x, x_orig)
    np.testing.assert_array_equal(shower.y, y_orig)
    np.testing.assert_array_equal(shower.z, z_orig)
    np.testing.assert_array_equal(shower.E, e_orig)
    np.testing.assert_array_equal(shower.t, t_orig)
    np.testing.assert_array_equal(shower.cell_id, cell_id_orig)


def test_greedy_agglomerative_compresses_two_blobs_in_one_cell():
    shower = _make_shower_same_cell_two_blobs()
    algo = GreedyAgglomerativeClustering(max_link_distance=0.5, max_cluster_distance=1.5)
    result = algo.compress(shower)
    assert result.shower.n_points == 2
    assert np.isclose(energy_ratio(shower, result.shower), 1.0, rtol=1e-6)
    assert result.shower.cell_id is not None
    np.testing.assert_array_equal(result.shower.cell_id, np.array([17, 17], dtype=np.uint64))


def test_greedy_agglomerative_max_cluster_distance_blocks_long_chain():
    shower = Shower(
        shower_id=0,
        x=np.array([0.0, 0.9, 1.8], dtype=np.float32),
        y=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        z=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        E=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        cell_id=np.array([5, 5, 5], dtype=np.uint64),
    )
    algo = GreedyAgglomerativeClustering(max_link_distance=1.0, max_cluster_distance=1.5)
    result = algo.compress(shower)
    assert result.shower.n_points == 2
    labels = result.debug_data["cluster_label"]
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_greedy_agglomerative_rejects_inverted_distance_limits():
    with pytest.raises(ValueError, match="greater than or equal"):
        GreedyAgglomerativeClustering(max_link_distance=2.0, max_cluster_distance=1.0)
