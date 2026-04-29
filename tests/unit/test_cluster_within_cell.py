import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering, KMeans

from step2point.algorithms.cluster_within_cell import ClusterWithinCell
from step2point.core.shower import Shower
from step2point.metrics.energy import energy_ratio
from step2point.validation.sanity import ShowerSanityValidator


def _make_multicell_shower(n_per_cell=20, seed=42):
    """Deposits spread across 2 cells, with spatial structure within each."""
    rng = np.random.default_rng(seed)

    # Cell A (cell_id=100): two sub-clusters at (10,10,500) and (10,20,500)
    n_half = n_per_cell // 2
    xa1 = rng.normal(10, 0.3, n_half).astype(np.float32)
    ya1 = rng.normal(10, 0.3, n_half).astype(np.float32)
    xa2 = rng.normal(10, 0.3, n_half).astype(np.float32)
    ya2 = rng.normal(20, 0.3, n_half).astype(np.float32)
    xa = np.concatenate([xa1, xa2])
    ya = np.concatenate([ya1, ya2])
    za = rng.normal(500, 0.1, n_per_cell).astype(np.float32)
    ea = rng.exponential(0.5, n_per_cell).astype(np.float32) + 0.01
    ta = rng.normal(10, 0.3, n_per_cell).astype(np.float32)
    cida = np.full(n_per_cell, 100, dtype=np.uint64)

    # Cell B (cell_id=200): one tight cluster at (50,50,600)
    xb = rng.normal(50, 0.3, n_per_cell).astype(np.float32)
    yb = rng.normal(50, 0.3, n_per_cell).astype(np.float32)
    zb = rng.normal(600, 0.1, n_per_cell).astype(np.float32)
    eb = rng.exponential(0.5, n_per_cell).astype(np.float32) + 0.01
    tb = rng.normal(12, 0.3, n_per_cell).astype(np.float32)
    cidb = np.full(n_per_cell, 200, dtype=np.uint64)

    return Shower(
        shower_id=0,
        x=np.concatenate([xa, xb]),
        y=np.concatenate([ya, yb]),
        z=np.concatenate([za, zb]),
        E=np.concatenate([ea, eb]),
        t=np.concatenate([ta, tb]),
        cell_id=np.concatenate([cida, cidb]),
    )


def _default_clusterer(distance_threshold=1.0):
    return AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)


def test_cluster_within_cell_does_not_mutate_input():
    shower = _make_multicell_shower()
    x_orig = shower.x.copy()
    y_orig = shower.y.copy()
    z_orig = shower.z.copy()
    E_orig = shower.E.copy()
    t_orig = shower.t.copy()
    cell_id_orig = shower.cell_id.copy()

    algo = ClusterWithinCell(clusterer=_default_clusterer())
    algo.compress(shower)

    np.testing.assert_array_equal(shower.x, x_orig)
    np.testing.assert_array_equal(shower.y, y_orig)
    np.testing.assert_array_equal(shower.z, z_orig)
    np.testing.assert_array_equal(shower.E, E_orig)
    np.testing.assert_array_equal(shower.t, t_orig)
    np.testing.assert_array_equal(shower.cell_id, cell_id_orig)


def test_cluster_within_cell_empty_shower():
    shower = Shower(
        shower_id=0,
        x=np.empty(0, dtype=np.float32),
        y=np.empty(0, dtype=np.float32),
        z=np.empty(0, dtype=np.float32),
        E=np.empty(0, dtype=np.float32),
        t=np.empty(0, dtype=np.float32),
        cell_id=np.empty(0, dtype=np.uint64),
    )
    algo = ClusterWithinCell(clusterer=_default_clusterer())
    result = algo.compress(shower)
    assert result.shower.n_points == 0
    assert result.stats["energy_after"] == 0.0


def test_cluster_within_cell_single_point_shower():
    shower = Shower(
        shower_id=0,
        x=np.array([100.0], dtype=np.float32),
        y=np.array([200.0], dtype=np.float32),
        z=np.array([300.0], dtype=np.float32),
        E=np.array([5.0], dtype=np.float32),
        t=np.array([1.0], dtype=np.float32),
        cell_id=np.array([42], dtype=np.uint64),
    )
    algo = ClusterWithinCell(clusterer=_default_clusterer())
    result = algo.compress(shower)
    assert result.shower.n_points == 1
    assert np.isclose(result.shower.E[0], 5.0)


def test_cluster_within_cell_compresses_and_preserves_energy():
    shower = _make_multicell_shower()
    algo = ClusterWithinCell(clusterer=_default_clusterer())
    result = algo.compress(shower)
    assert result.shower.n_points < shower.n_points
    assert np.isclose(energy_ratio(shower, result.shower), 1.0, rtol=1e-6)


def test_cluster_within_cell_output_passes_sanity():
    shower = _make_multicell_shower()
    algo = ClusterWithinCell(clusterer=_default_clusterer())
    result = algo.compress(shower)
    vr = ShowerSanityValidator().run(shower, result.shower)
    assert vr.metrics["passed"]


def test_cluster_within_cell_stats_are_populated():
    shower = _make_multicell_shower()
    algo = ClusterWithinCell(clusterer=_default_clusterer())
    result = algo.compress(shower)
    assert result.algorithm == "cluster_within_cell"
    assert result.stats["n_points_before"] == shower.n_points
    assert result.stats["n_points_after"] == result.shower.n_points
    assert result.stats["compression_ratio"] < 1.0


def test_cluster_within_cell_requires_cell_id():
    shower = Shower(
        shower_id=0,
        x=np.array([0, 1], dtype=np.float32),
        y=np.array([0, 1], dtype=np.float32),
        z=np.array([0, 1], dtype=np.float32),
        E=np.array([1, 2], dtype=np.float32),
        t=np.array([0, 1], dtype=np.float32),
    )
    algo = ClusterWithinCell(clusterer=_default_clusterer())
    with pytest.raises(ValueError, match="cell_id"):
        algo.compress(shower)


def test_cluster_within_cell_works_without_time():
    shower = _make_multicell_shower()
    shower_no_t = Shower(
        shower_id=shower.shower_id,
        x=shower.x, y=shower.y, z=shower.z, E=shower.E,
        cell_id=shower.cell_id,
    )
    algo = ClusterWithinCell(clusterer=_default_clusterer())
    result = algo.compress(shower_no_t)
    assert result.shower.n_points < shower_no_t.n_points
    assert result.shower.t is None
    assert np.isclose(energy_ratio(shower_no_t, result.shower), 1.0, rtol=1e-6)


def test_cluster_within_cell_preserves_cell_id():
    shower = _make_multicell_shower()
    algo = ClusterWithinCell(clusterer=_default_clusterer())
    result = algo.compress(shower)
    # Every output cell_id must be one of the input cell_ids
    input_cells = set(shower.cell_id.tolist())
    output_cells = set(result.shower.cell_id.tolist())
    assert output_cells.issubset(input_cells)
    # Both input cells should appear in output
    assert output_cells == input_cells


def test_cluster_within_cell_tight_threshold_preserves_more_points():
    shower = _make_multicell_shower()
    algo_tight = ClusterWithinCell(clusterer=_default_clusterer(distance_threshold=0.1))
    algo_loose = ClusterWithinCell(clusterer=_default_clusterer(distance_threshold=100.0))
    result_tight = algo_tight.compress(shower)
    result_loose = algo_loose.compress(shower)
    # Tight threshold merges less -> more output points
    assert result_tight.shower.n_points > result_loose.shower.n_points
    # Both conserve energy
    assert np.isclose(energy_ratio(shower, result_tight.shower), 1.0, rtol=1e-6)
    assert np.isclose(energy_ratio(shower, result_loose.shower), 1.0, rtol=1e-6)


def test_cluster_within_cell_large_threshold_behaves_like_merge_within_cell():
    """Very large distance_threshold merges everything in each cell to ~1 point."""
    shower = _make_multicell_shower()
    algo = ClusterWithinCell(clusterer=_default_clusterer(distance_threshold=1e6))
    result = algo.compress(shower)
    n_unique_cells = len(np.unique(shower.cell_id))
    # Should produce exactly 1 point per cell (= MergeWithinCell behaviour)
    assert result.shower.n_points == n_unique_cells
    assert np.isclose(energy_ratio(shower, result.shower), 1.0, rtol=1e-6)


def test_cluster_within_cell_works_with_different_clusterers():
    """Verify it works with KMeans as an alternative clusterer."""
    shower = _make_multicell_shower()
    algo = ClusterWithinCell(clusterer=KMeans(n_clusters=2, n_init=1, random_state=42))
    result = algo.compress(shower)
    n_unique_cells = len(np.unique(shower.cell_id))
    # KMeans(n_clusters=2) should produce exactly 2 clusters per cell
    assert result.shower.n_points == 2 * n_unique_cells
    assert np.isclose(energy_ratio(shower, result.shower), 1.0, rtol=1e-6)
