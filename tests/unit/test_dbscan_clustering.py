import numpy as np
import pytest

from step2point.algorithms.dbscan_clustering import DBSCANClustering
from step2point.core.shower import Shower
from step2point.metrics.energy import energy_ratio

DD4HEP_ENCODING = "system:8,layer:6,hit:50"
NEIGHBOUR_ENCODING = "system:8,layer:6,x:8,y:8"


def _encode_cell_id(system: int, layer: int, hit_index: int) -> np.uint64:
    return np.uint64(system | (layer << 8) | (hit_index << 14))


def _encode_neighbour_cell_id(system: int, layer: int, x_bin: int, y_bin: int) -> np.uint64:
    return np.uint64(system | (layer << 8) | (np.uint64(x_bin) << 14) | (np.uint64(y_bin) << 22))


def _make_clustered_shower(n_per_cluster: int = 30, seed: int = 42) -> Shower:
    rng = np.random.default_rng(seed)
    xa = rng.normal(100, 2, n_per_cluster).astype(np.float32)
    ya = rng.normal(100, 2, n_per_cluster).astype(np.float32)
    za = rng.normal(500, 1, n_per_cluster).astype(np.float32)
    ea = (rng.exponential(0.5, n_per_cluster) + 0.01).astype(np.float32)
    ta = rng.normal(10, 0.3, n_per_cluster).astype(np.float32)
    cida = np.array([_encode_cell_id(3, 1, i) for i in range(n_per_cluster)], dtype=np.uint64)

    xb = rng.normal(200, 2, n_per_cluster).astype(np.float32)
    yb = rng.normal(200, 2, n_per_cluster).astype(np.float32)
    zb = rng.normal(500, 1, n_per_cluster).astype(np.float32)
    eb = (rng.exponential(0.5, n_per_cluster) + 0.01).astype(np.float32)
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


def test_dbscan_requires_cell_id_when_scope_needs_it():
    with pytest.raises(ValueError, match="cell_id can be decoded"):
        DBSCANClustering()


def test_dbscan_does_not_mutate_input():
    shower = _make_clustered_shower()
    x_orig = shower.x.copy()
    y_orig = shower.y.copy()
    z_orig = shower.z.copy()
    e_orig = shower.E.copy()
    t_orig = shower.t.copy()
    cell_id_orig = shower.cell_id.copy()

    DBSCANClustering(eps=1.0, min_samples=3, cell_id_encoding=DD4HEP_ENCODING).compress(shower)

    np.testing.assert_array_equal(shower.x, x_orig)
    np.testing.assert_array_equal(shower.y, y_orig)
    np.testing.assert_array_equal(shower.z, z_orig)
    np.testing.assert_array_equal(shower.E, e_orig)
    np.testing.assert_array_equal(shower.t, t_orig)
    np.testing.assert_array_equal(shower.cell_id, cell_id_orig)


def test_dbscan_compresses_and_preserves_energy():
    shower = _make_clustered_shower()
    result = DBSCANClustering(eps=1.0, min_samples=3, cell_id_encoding=DD4HEP_ENCODING).compress(shower)
    assert result.shower.n_points < shower.n_points
    assert np.isclose(energy_ratio(shower, result.shower), 1.0, rtol=1e-6)


def test_dbscan_assigns_representative_cell_id_to_each_cluster():
    shower = _make_clustered_shower()
    result = DBSCANClustering(eps=1.0, min_samples=3, cell_id_encoding=DD4HEP_ENCODING).compress(shower)
    assert result.shower.cell_id is not None
    assert len(result.shower.cell_id) == result.shower.n_points
    assert result.shower.metadata["approximate_cell_id"] is True
    assert set(np.asarray(result.shower.cell_id, dtype=np.uint64)).issubset(set(np.asarray(shower.cell_id, dtype=np.uint64)))


def test_dbscan_standalone_outlier_policy_keeps_outlier_separate():
    shower = _make_clustered_shower()
    outlier = Shower(
        shower_id=shower.shower_id,
        x=np.concatenate([shower.x, np.array([500.0], dtype=np.float32)]),
        y=np.concatenate([shower.y, np.array([500.0], dtype=np.float32)]),
        z=np.concatenate([shower.z, np.array([900.0], dtype=np.float32)]),
        E=np.concatenate([shower.E, np.array([0.25], dtype=np.float32)]),
        t=np.concatenate([shower.t, np.array([50.0], dtype=np.float32)]),
        cell_id=np.concatenate([shower.cell_id, np.array([_encode_cell_id(3, 1, 999)], dtype=np.uint64)]),
    )
    nearest = DBSCANClustering(
        eps=1.0,
        min_samples=3,
        outlier_policy="nearest_cluster",
        cell_id_encoding=DD4HEP_ENCODING,
    ).compress(outlier).shower
    standalone = DBSCANClustering(
        eps=1.0,
        min_samples=3,
        outlier_policy="standalone",
        cell_id_encoding=DD4HEP_ENCODING,
    ).compress(outlier).shower
    assert standalone.n_points >= nearest.n_points + 1
    assert np.isclose(energy_ratio(outlier, nearest), 1.0, rtol=1e-6)
    assert np.isclose(energy_ratio(outlier, standalone), 1.0, rtol=1e-6)


def test_dbscan_cell_id_neighbour_scope_partitions_disconnected_cells():
    rng = np.random.default_rng(11)
    x = np.concatenate([rng.normal(0.0, 0.05, 10), rng.normal(1.0, 0.05, 10), rng.normal(10.0, 0.05, 10)]).astype(np.float32)
    y = np.concatenate([rng.normal(0.0, 0.05, 10), rng.normal(0.0, 0.05, 10), rng.normal(10.0, 0.05, 10)]).astype(np.float32)
    z = rng.normal(100.0, 0.05, 30).astype(np.float32)
    e = (rng.exponential(0.5, 30) + 0.01).astype(np.float32)
    t = rng.normal(5.0, 0.1, 30).astype(np.float32)
    cell_id = np.concatenate(
        [
            np.full(10, _encode_neighbour_cell_id(3, 1, 0, 0), dtype=np.uint64),
            np.full(10, _encode_neighbour_cell_id(3, 1, 1, 0), dtype=np.uint64),
            np.full(10, _encode_neighbour_cell_id(3, 1, 5, 5), dtype=np.uint64),
        ]
    )
    shower = Shower(shower_id=0, x=x, y=y, z=z, E=e, t=t, cell_id=cell_id)
    algo = DBSCANClustering(eps=1.0, min_samples=3, merge_scope="cell_id_neighbour", cell_id_encoding=NEIGHBOUR_ENCODING)
    partitions = algo._partition_indices(shower)
    partition_sizes = sorted(len(partition) for partition in partitions)
    assert partition_sizes == [10, 20]


def test_dbscan_use_time_true_requires_time():
    shower = _make_clustered_shower()
    shower_no_t = Shower(shower_id=shower.shower_id, x=shower.x, y=shower.y, z=shower.z, E=shower.E, cell_id=shower.cell_id)
    algo = DBSCANClustering(eps=1.0, min_samples=3, use_time=True, cell_id_encoding=DD4HEP_ENCODING)
    with pytest.raises(ValueError, match="use_time"):
        algo.compress(shower_no_t)
