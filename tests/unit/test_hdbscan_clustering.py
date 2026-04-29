import numpy as np
import pytest

from step2point.algorithms.hdbscan_clustering import HDBSCANClustering
from step2point.core.shower import Shower
from step2point.metrics.energy import energy_ratio
from step2point.validation.sanity import ShowerSanityValidator

DD4HEP_ENCODING = "system:8,layer:6,hit:50"


def _encode_cell_id(system: int, layer: int, hit_index: int) -> np.uint64:
    return np.uint64(system | (layer << 8) | (hit_index << 14))


def _make_clustered_shower(n_per_cluster=30, seed=42):
    """Two well-separated blobs in x,y within the same layer (layer 1)."""
    rng = np.random.default_rng(seed)
    # Cluster A: centred at (100, 100), layer 1
    xa = rng.normal(100, 2, n_per_cluster).astype(np.float32)
    ya = rng.normal(100, 2, n_per_cluster).astype(np.float32)
    za = rng.normal(500, 1, n_per_cluster).astype(np.float32)
    ea = rng.exponential(0.5, n_per_cluster).astype(np.float32) + 0.01
    ta = rng.normal(10, 0.3, n_per_cluster).astype(np.float32)
    cida = np.array([_encode_cell_id(3, 1, i) for i in range(n_per_cluster)], dtype=np.uint64)

    # Cluster B: centred at (200, 200), layer 1
    xb = rng.normal(200, 2, n_per_cluster).astype(np.float32)
    yb = rng.normal(200, 2, n_per_cluster).astype(np.float32)
    zb = rng.normal(500, 1, n_per_cluster).astype(np.float32)
    eb = rng.exponential(0.5, n_per_cluster).astype(np.float32) + 0.01
    tb = rng.normal(10, 0.3, n_per_cluster).astype(np.float32)
    cidb = np.array([_encode_cell_id(3, 1, n_per_cluster + i) for i in range(n_per_cluster)], dtype=np.uint64)

    return Shower(
        shower_id=0,
        x=np.concatenate([xa, xb]),
        y=np.concatenate([ya, yb]),
        z=np.concatenate([za, zb]),
        E=np.concatenate([ea, eb]),
        t=np.concatenate([ta, tb]),
        cell_id=np.concatenate([cida, cidb]),
    )


def _make_clustered_shower_with_outlier(seed=123):
    shower = _make_clustered_shower(seed=seed)
    return Shower(
        shower_id=shower.shower_id,
        x=np.concatenate([shower.x, np.array([500.0], dtype=np.float32)]),
        y=np.concatenate([shower.y, np.array([500.0], dtype=np.float32)]),
        z=np.concatenate([shower.z, np.array([900.0], dtype=np.float32)]),
        E=np.concatenate([shower.E, np.array([0.25], dtype=np.float32)]),
        t=np.concatenate([shower.t, np.array([50.0], dtype=np.float32)]),
        cell_id=np.concatenate([shower.cell_id, np.array([_encode_cell_id(3, 1, 999)], dtype=np.uint64)]),
        metadata=dict(shower.metadata),
        primary=dict(shower.primary),
    )


def test_hdbscan_does_not_mutate_input():
    shower = _make_clustered_shower()
    x_orig = shower.x.copy()
    y_orig = shower.y.copy()
    z_orig = shower.z.copy()
    E_orig = shower.E.copy()
    t_orig = shower.t.copy()
    cell_id_orig = shower.cell_id.copy()

    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3, cell_id_encoding=DD4HEP_ENCODING)
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
    algo = HDBSCANClustering(min_cluster_size=2, min_samples=2, cell_id_encoding=DD4HEP_ENCODING)
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
        cell_id=np.array([_encode_cell_id(3, 1, 0)], dtype=np.uint64),
    )
    algo = HDBSCANClustering(min_cluster_size=2, min_samples=2, cell_id_encoding=DD4HEP_ENCODING)
    result = algo.compress(shower)
    # Single point is below min_samples, should be preserved as singleton
    assert result.shower.n_points == 1
    assert np.isclose(result.shower.E[0], 5.0)


def test_hdbscan_repeated_cell_id():
    """Multiple points sharing the same cell_id (same layer) cluster correctly."""
    rng = np.random.default_rng(77)
    n = 40
    shared_cell_id = _encode_cell_id(3, 1, 42)
    shower = Shower(
        shower_id=0,
        x=rng.normal(100, 2, n).astype(np.float32),
        y=rng.normal(100, 2, n).astype(np.float32),
        z=rng.normal(500, 1, n).astype(np.float32),
        E=rng.exponential(0.5, n).astype(np.float32) + 0.01,
        t=rng.normal(10, 0.3, n).astype(np.float32),
        cell_id=np.full(n, shared_cell_id, dtype=np.uint64),
    )
    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3, cell_id_encoding=DD4HEP_ENCODING)
    result = algo.compress(shower)
    assert result.shower.n_points <= shower.n_points
    assert np.isclose(energy_ratio(shower, result.shower), 1.0, rtol=1e-6)


def test_hdbscan_compresses_and_preserves_energy():
    shower = _make_clustered_shower()
    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3, cell_id_encoding=DD4HEP_ENCODING)
    result = algo.compress(shower)
    assert result.shower.n_points < shower.n_points
    assert np.isclose(energy_ratio(shower, result.shower), 1.0, rtol=1e-6)


def test_hdbscan_assigns_representative_cell_id_to_each_cluster():
    shower = _make_clustered_shower()
    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3, cell_id_encoding=DD4HEP_ENCODING)
    result = algo.compress(shower)
    assert result.shower.cell_id is not None
    assert len(result.shower.cell_id) == result.shower.n_points
    assert result.shower.metadata["approximate_cell_id"] is True
    assert set(np.asarray(result.shower.cell_id, dtype=np.uint64)).issubset(set(np.asarray(shower.cell_id, dtype=np.uint64)))


def test_hdbscan_standalone_outlier_policy_keeps_outlier_separate():
    shower = _make_clustered_shower_with_outlier()
    nearest = HDBSCANClustering(
        min_cluster_size=5,
        min_samples=3,
        outlier_policy="nearest_cluster",
        cell_id_encoding=DD4HEP_ENCODING,
    ).compress(shower).shower
    standalone = HDBSCANClustering(
        min_cluster_size=5,
        min_samples=3,
        outlier_policy="standalone",
        cell_id_encoding=DD4HEP_ENCODING,
    ).compress(shower).shower
    assert standalone.n_points >= nearest.n_points + 1
    assert np.isclose(energy_ratio(shower, nearest), 1.0, rtol=1e-6)
    assert np.isclose(energy_ratio(shower, standalone), 1.0, rtol=1e-6)


def test_hdbscan_output_passes_sanity():
    shower = _make_clustered_shower()
    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3, cell_id_encoding=DD4HEP_ENCODING)
    result = algo.compress(shower)
    vr = ShowerSanityValidator().run(shower, result.shower)
    assert vr.metrics["passed"]


def test_hdbscan_stats_are_populated():
    shower = _make_clustered_shower()
    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3, cell_id_encoding=DD4HEP_ENCODING)
    result = algo.compress(shower)
    assert result.algorithm == "hdbscan"
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
    algo = HDBSCANClustering(min_cluster_size=2, min_samples=1, cell_id_encoding=DD4HEP_ENCODING)
    with pytest.raises(ValueError, match="cell_id"):
        algo.compress(shower)


def test_hdbscan_works_without_time():
    shower = _make_clustered_shower()
    shower_no_t = Shower(
        shower_id=shower.shower_id,
        x=shower.x, y=shower.y, z=shower.z, E=shower.E,
        cell_id=shower.cell_id,
    )
    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3, cell_id_encoding=DD4HEP_ENCODING)
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
    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3, use_time=True, cell_id_encoding=DD4HEP_ENCODING)
    with pytest.raises(ValueError, match="use_time"):
        algo.compress(shower_no_t)


def test_hdbscan_use_time_false_ignores_time():
    """use_time=False clusters on (x, y, z) only even when time is present."""
    shower = _make_clustered_shower()
    assert shower.t is not None
    algo_with_t = HDBSCANClustering(min_cluster_size=5, min_samples=3, use_time=True, cell_id_encoding=DD4HEP_ENCODING)
    algo_without_t = HDBSCANClustering(min_cluster_size=5, min_samples=3, use_time=False, cell_id_encoding=DD4HEP_ENCODING)
    result_with = algo_with_t.compress(shower)
    result_without = algo_without_t.compress(shower)
    # Both conserve energy
    assert np.isclose(energy_ratio(shower, result_with.shower), 1.0, rtol=1e-6)
    assert np.isclose(energy_ratio(shower, result_without.shower), 1.0, rtol=1e-6)
    # use_time=False should still output time (min per cluster)
    assert result_without.shower.t is not None


def test_hdbscan_requires_cell_id_decoding_rule():
    with pytest.raises(ValueError, match="cell_id can be decoded"):
        HDBSCANClustering(min_cluster_size=5, min_samples=3)


def test_hdbscan_no_subdetector_metadata():
    """Algorithm works when subdetector is absent from metadata."""
    shower = _make_clustered_shower()
    shower.metadata.pop("subdetector", None)
    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3, cell_id_encoding=DD4HEP_ENCODING)
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
    cid1 = np.array([_encode_cell_id(3, 1, i) for i in range(n)], dtype=np.uint64)

    # Layer 2 cluster at a different position
    x2 = rng.normal(300, 2, n).astype(np.float32)
    y2 = rng.normal(300, 2, n).astype(np.float32)
    z2 = rng.normal(600, 1, n).astype(np.float32)
    e2 = rng.exponential(0.5, n).astype(np.float32) + 0.01
    t2 = rng.normal(12, 0.3, n).astype(np.float32)
    cid2 = np.array([_encode_cell_id(3, 2, i) for i in range(n)], dtype=np.uint64)

    shower = Shower(
        shower_id=0,
        x=np.concatenate([x1, x2]),
        y=np.concatenate([y1, y2]),
        z=np.concatenate([z1, z2]),
        E=np.concatenate([e1, e2]),
        t=np.concatenate([t1, t2]),
        cell_id=np.concatenate([cid1, cid2]),
    )
    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3, cell_id_encoding=DD4HEP_ENCODING)
    result = algo.compress(shower)
    assert result.shower.n_points < shower.n_points
    assert np.isclose(energy_ratio(shower, result.shower), 1.0, rtol=1e-6)


def test_hdbscan_all_noise_fallback():
    """When min_cluster_size is larger than the slice, all points are noise."""
    shower = _make_clustered_shower(n_per_cluster=10)
    # min_cluster_size=100 forces HDBSCAN to label everything as noise
    algo = HDBSCANClustering(min_cluster_size=100, min_samples=3, cell_id_encoding=DD4HEP_ENCODING)
    result = algo.compress(shower)
    # All noise is bundled into one cluster per layer, preserving energy
    assert result.shower.n_points > 0
    assert np.isclose(energy_ratio(shower, result.shower), 1.0, rtol=1e-6)


def test_hdbscan_multiple_cell_id_encodings():
    shower = _make_clustered_shower()
    half = shower.n_points // 2
    shower.metadata["subdetector"] = np.concatenate(
        [
            np.zeros(half, dtype=np.uint8),
            np.ones(shower.n_points - half, dtype=np.uint8),
        ]
    )
    second_encoding = "system:8,layer:6,other:50"
    result = HDBSCANClustering(
        min_cluster_size=5,
        min_samples=3,
        cell_id_encoding=(DD4HEP_ENCODING, second_encoding),
    ).compress(shower)
    assert result.shower.n_points > 0
    assert np.isclose(energy_ratio(shower, result.shower), 1.0, rtol=1e-6)
