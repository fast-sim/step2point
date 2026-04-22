import numpy as np
import pytest

from step2point.algorithms.hdbscan_clustering import HDBSCANClustering, _default_layer_extractor
from step2point.core.shower import Shower
from step2point.metrics.energy import energy_ratio
from step2point.validation.sanity import ShowerSanityValidator

pytest.importorskip("sklearn")


def _make_clustered_shower(n_per_cluster=30, seed=42):
    """Two well-separated blobs in x,y within the same layer (layer 1)."""
    rng = np.random.default_rng(seed)
    # Cluster A: centred at (100, 100), layer 1
    xa = rng.normal(100, 2, n_per_cluster).astype(np.float32)
    ya = rng.normal(100, 2, n_per_cluster).astype(np.float32)
    za = rng.normal(500, 1, n_per_cluster).astype(np.float32)
    ea = rng.exponential(0.5, n_per_cluster).astype(np.float32) + 0.01
    ta = rng.normal(10, 0.3, n_per_cluster).astype(np.float32)
    # layer 1 encoded at bit 19: cell_id = (1 << 19) | unique_low_bits
    cida = np.array([(1 << 19) | i for i in range(n_per_cluster)], dtype=np.uint64)

    # Cluster B: centred at (200, 200), layer 1
    xb = rng.normal(200, 2, n_per_cluster).astype(np.float32)
    yb = rng.normal(200, 2, n_per_cluster).astype(np.float32)
    zb = rng.normal(500, 1, n_per_cluster).astype(np.float32)
    eb = rng.exponential(0.5, n_per_cluster).astype(np.float32) + 0.01
    tb = rng.normal(10, 0.3, n_per_cluster).astype(np.float32)
    cidb = np.array([(1 << 19) | (n_per_cluster + i) for i in range(n_per_cluster)], dtype=np.uint64)

    return Shower(
        shower_id=0,
        x=np.concatenate([xa, xb]),
        y=np.concatenate([ya, yb]),
        z=np.concatenate([za, zb]),
        E=np.concatenate([ea, eb]),
        t=np.concatenate([ta, tb]),
        cell_id=np.concatenate([cida, cidb]),
    )


def test_hdbscan_does_not_mutate_input():
    shower = _make_clustered_shower()
    x_orig = shower.x.copy()
    y_orig = shower.y.copy()
    z_orig = shower.z.copy()
    E_orig = shower.E.copy()
    t_orig = shower.t.copy()
    cell_id_orig = shower.cell_id.copy()

    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3)
    algo.compress(shower)

    np.testing.assert_array_equal(shower.x, x_orig)
    np.testing.assert_array_equal(shower.y, y_orig)
    np.testing.assert_array_equal(shower.z, z_orig)
    np.testing.assert_array_equal(shower.E, E_orig)
    np.testing.assert_array_equal(shower.t, t_orig)
    np.testing.assert_array_equal(shower.cell_id, cell_id_orig)


def test_hdbscan_empty_shower():
    shower = Shower(
        shower_id=0,
        x=np.empty(0, dtype=np.float32),
        y=np.empty(0, dtype=np.float32),
        z=np.empty(0, dtype=np.float32),
        E=np.empty(0, dtype=np.float32),
        t=np.empty(0, dtype=np.float32),
        cell_id=np.empty(0, dtype=np.uint64),
    )
    algo = HDBSCANClustering(min_cluster_size=2, min_samples=2)
    result = algo.compress(shower)
    assert result.shower.n_points == 0
    assert result.stats["energy_after"] == 0.0


def test_hdbscan_single_point_shower():
    shower = Shower(
        shower_id=0,
        x=np.array([100.0], dtype=np.float32),
        y=np.array([200.0], dtype=np.float32),
        z=np.array([300.0], dtype=np.float32),
        E=np.array([5.0], dtype=np.float32),
        t=np.array([1.0], dtype=np.float32),
        cell_id=np.array([1 << 19], dtype=np.uint64),
    )
    algo = HDBSCANClustering(min_cluster_size=2, min_samples=2)
    result = algo.compress(shower)
    # Single point is below min_samples, should be preserved as singleton
    assert result.shower.n_points == 1
    assert np.isclose(result.shower.E[0], 5.0)


def test_hdbscan_repeated_cell_id():
    """Multiple points sharing the same cell_id (same layer) cluster correctly."""
    rng = np.random.default_rng(77)
    n = 40
    shared_cell_id = np.uint64((1 << 19) | 42)
    shower = Shower(
        shower_id=0,
        x=rng.normal(100, 2, n).astype(np.float32),
        y=rng.normal(100, 2, n).astype(np.float32),
        z=rng.normal(500, 1, n).astype(np.float32),
        E=rng.exponential(0.5, n).astype(np.float32) + 0.01,
        t=rng.normal(10, 0.3, n).astype(np.float32),
        cell_id=np.full(n, shared_cell_id, dtype=np.uint64),
    )
    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3)
    result = algo.compress(shower)
    assert result.shower.n_points <= shower.n_points
    assert np.isclose(energy_ratio(shower, result.shower), 1.0, rtol=1e-6)


def test_hdbscan_compresses_and_preserves_energy():
    shower = _make_clustered_shower()
    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3)
    result = algo.compress(shower)
    assert result.shower.n_points < shower.n_points
    assert np.isclose(energy_ratio(shower, result.shower), 1.0, rtol=1e-6)


def test_hdbscan_output_passes_sanity():
    shower = _make_clustered_shower()
    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3)
    result = algo.compress(shower)
    vr = ShowerSanityValidator().run(shower, result.shower)
    assert vr.metrics["passed"]


def test_hdbscan_stats_are_populated():
    shower = _make_clustered_shower()
    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3)
    result = algo.compress(shower)
    assert result.algorithm == "hdbscan_clustering"
    assert result.stats["n_points_before"] == shower.n_points
    assert result.stats["n_points_after"] == result.shower.n_points
    assert result.stats["compression_ratio"] < 1.0


def test_hdbscan_requires_cell_id():
    shower = Shower(
        shower_id=0,
        x=np.array([0, 1], dtype=np.float32),
        y=np.array([0, 1], dtype=np.float32),
        z=np.array([0, 1], dtype=np.float32),
        E=np.array([1, 2], dtype=np.float32),
        t=np.array([0, 1], dtype=np.float32),
    )
    algo = HDBSCANClustering(min_cluster_size=2, min_samples=1)
    with pytest.raises(ValueError, match="cell_id"):
        algo.compress(shower)


def test_hdbscan_works_without_time():
    shower = _make_clustered_shower()
    shower_no_t = Shower(
        shower_id=shower.shower_id,
        x=shower.x, y=shower.y, z=shower.z, E=shower.E,
        cell_id=shower.cell_id,
    )
    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3)
    result = algo.compress(shower_no_t)
    assert result.shower.n_points < shower_no_t.n_points
    assert result.shower.t is None
    assert np.isclose(energy_ratio(shower_no_t, result.shower), 1.0, rtol=1e-6)


def test_hdbscan_use_time_true_requires_time():
    shower = _make_clustered_shower()
    shower_no_t = Shower(
        shower_id=shower.shower_id,
        x=shower.x, y=shower.y, z=shower.z, E=shower.E,
        cell_id=shower.cell_id,
    )
    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3, use_time=True)
    with pytest.raises(ValueError, match="use_time"):
        algo.compress(shower_no_t)


def test_hdbscan_use_time_false_ignores_time():
    """use_time=False clusters on (x, y) only even when time is present."""
    shower = _make_clustered_shower()
    assert shower.t is not None
    algo_with_t = HDBSCANClustering(min_cluster_size=5, min_samples=3, use_time=True)
    algo_without_t = HDBSCANClustering(min_cluster_size=5, min_samples=3, use_time=False)
    result_with = algo_with_t.compress(shower)
    result_without = algo_without_t.compress(shower)
    # Both conserve energy
    assert np.isclose(energy_ratio(shower, result_with.shower), 1.0, rtol=1e-6)
    assert np.isclose(energy_ratio(shower, result_without.shower), 1.0, rtol=1e-6)
    # use_time=False should still output time (min per cluster)
    assert result_without.shower.t is not None


def test_hdbscan_custom_layer_extractor():
    shower = _make_clustered_shower()
    # Use a trivial layer extractor that puts everything in layer 0
    algo = HDBSCANClustering(
        min_cluster_size=5,
        min_samples=3,
        layer_extractor=lambda cid: np.zeros(len(cid), dtype=np.int32),
    )
    result = algo.compress(shower)
    assert result.shower.n_points < shower.n_points


def test_default_layer_extractor():
    cell_ids = np.array([0, 1 << 19, 2 << 19, (3 << 19) | 0xFF], dtype=np.uint64)
    layers = _default_layer_extractor(cell_ids)
    np.testing.assert_array_equal(layers, [0, 1, 2, 3])


def test_hdbscan_no_subdetector_metadata():
    """Algorithm works when subdetector is absent from metadata."""
    shower = _make_clustered_shower()
    shower.metadata.pop("subdetector", None)
    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3)
    result = algo.compress(shower)
    assert result.shower.n_points > 0
    assert np.isclose(energy_ratio(shower, result.shower), 1.0, rtol=1e-6)


def test_hdbscan_multiple_layers():
    """Two clusters in different layers are processed independently."""
    rng = np.random.default_rng(99)
    n = 30
    # Layer 1 cluster
    x1 = rng.normal(100, 2, n).astype(np.float32)
    y1 = rng.normal(100, 2, n).astype(np.float32)
    z1 = rng.normal(500, 1, n).astype(np.float32)
    e1 = rng.exponential(0.5, n).astype(np.float32) + 0.01
    t1 = rng.normal(10, 0.3, n).astype(np.float32)
    cid1 = np.array([(1 << 19) | i for i in range(n)], dtype=np.uint64)

    # Layer 2 cluster at a different position
    x2 = rng.normal(300, 2, n).astype(np.float32)
    y2 = rng.normal(300, 2, n).astype(np.float32)
    z2 = rng.normal(600, 1, n).astype(np.float32)
    e2 = rng.exponential(0.5, n).astype(np.float32) + 0.01
    t2 = rng.normal(12, 0.3, n).astype(np.float32)
    cid2 = np.array([(2 << 19) | i for i in range(n)], dtype=np.uint64)

    shower = Shower(
        shower_id=0,
        x=np.concatenate([x1, x2]),
        y=np.concatenate([y1, y2]),
        z=np.concatenate([z1, z2]),
        E=np.concatenate([e1, e2]),
        t=np.concatenate([t1, t2]),
        cell_id=np.concatenate([cid1, cid2]),
    )
    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3)
    result = algo.compress(shower)
    assert result.shower.n_points < shower.n_points
    assert np.isclose(energy_ratio(shower, result.shower), 1.0, rtol=1e-6)


def test_hdbscan_all_noise_fallback():
    """When min_cluster_size is larger than the slice, all points are noise."""
    shower = _make_clustered_shower(n_per_cluster=10)
    # min_cluster_size=100 forces HDBSCAN to label everything as noise
    algo = HDBSCANClustering(min_cluster_size=100, min_samples=3)
    result = algo.compress(shower)
    # All noise is bundled into one cluster per layer, preserving energy
    assert result.shower.n_points > 0
    assert np.isclose(energy_ratio(shower, result.shower), 1.0, rtol=1e-6)
