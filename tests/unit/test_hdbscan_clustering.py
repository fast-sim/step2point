import numpy as np
import pytest

from step2point.core.shower import Shower
from step2point.metrics.energy import energy_ratio
from step2point.validation.sanity import ShowerSanityValidator

pytest.importorskip("sklearn")

from step2point.algorithms.hdbscan_clustering import HDBSCANClustering, _default_layer_extractor  # noqa: E402


def _make_clustered_shower(n_per_cluster=30, seed=42):
    """Two well-separated blobs in x,y with distinct layers encoded in cell_id."""
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


def test_hdbscan_compresses_and_preserves_energy():
    shower = _make_clustered_shower()
    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3, noise_handle="nn")
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


def test_hdbscan_noise_handle_drop_may_lose_energy():
    shower = _make_clustered_shower()
    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3, noise_handle="drop")
    result = algo.compress(shower)
    # drop may lose energy (ratio <= 1) or preserve it if there's no noise
    assert result.stats["energy_after"] <= result.stats["energy_before"] + 1e-6


def test_hdbscan_noise_handle_singleton():
    shower = _make_clustered_shower()
    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3, noise_handle="singleton")
    result = algo.compress(shower)
    assert np.isclose(energy_ratio(shower, result.shower), 1.0, rtol=1e-6)


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


def test_hdbscan_requires_time():
    shower = Shower(
        shower_id=0,
        x=np.array([0, 1], dtype=np.float32),
        y=np.array([0, 1], dtype=np.float32),
        z=np.array([0, 1], dtype=np.float32),
        E=np.array([1, 2], dtype=np.float32),
        cell_id=np.array([1, 2], dtype=np.uint64),
    )
    algo = HDBSCANClustering(min_cluster_size=2, min_samples=1)
    with pytest.raises(ValueError, match="time"):
        algo.compress(shower)


def test_hdbscan_invalid_noise_handle():
    with pytest.raises(ValueError, match="noise_handle"):
        HDBSCANClustering(min_cluster_size=2, min_samples=1, noise_handle="bad")


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
