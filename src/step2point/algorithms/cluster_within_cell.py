"""Adaptive clustering of deposits within each detector cell.

This algorithm groups deposits by ``cell_id`` and runs a pluggable
scikit-learn-compatible clusterer within each cell. The number of output
points per cell is determined by the clusterer, not fixed in advance.
Each cluster is merged into a single point: energy-weighted centroid
position, summed energy, minimum time.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone

from step2point.algorithms.base import CompressionAlgorithm
from step2point.core.results import CompressionResult
from step2point.core.shower import Shower


def _cluster_one_cell(clusterer, positions: np.ndarray) -> np.ndarray:
    """Cluster a single cell's positions, returning integer labels.

    Module-level (not a method) so that joblib can pickle it for
    process-based parallelism.
    """
    if len(positions) <= 1:
        return np.zeros(len(positions), dtype=np.int64)
    try:
        return clone(clusterer).fit_predict(positions)
    except ValueError:
        # Clusterer can't handle this group (e.g. KMeans with
        # n_clusters > n_points). Each point becomes its own cluster.
        return np.arange(len(positions), dtype=np.int64)


class ClusterWithinCell(CompressionAlgorithm):
    """Adaptive clustering of deposits within each detector cell.

    Deposits are grouped by ``cell_id``, then a user-supplied clusterer
    is run on the 3-D positions within each cell. The clusterer decides
    how many sub-clusters to produce. Each cluster is merged into a
    single point: energy-weighted centroid position, summed energy,
    minimum time.

    Parameters
    ----------
    clusterer
        Any scikit-learn-compatible estimator that implements
        ``fit_predict(X) -> labels``. A good starting point is
        ``AgglomerativeClustering(n_clusters=None, distance_threshold=1.0)``
        which merges deposits closer than 1 mm.
    n_jobs : int
        Number of parallel jobs for per-cell clustering. ``1``
        (default) runs sequentially. ``-1`` uses all available cores.
    """

    name = "cluster_within_cell"

    def __init__(self, clusterer, *, n_jobs: int = 1) -> None:
        self.clusterer = clusterer
        self.n_jobs = n_jobs

    def compress(self, shower: Shower) -> CompressionResult:
        if shower.cell_id is None:
            raise ValueError("ClusterWithinCell requires cell_id.")
        unique_cells, cell_inverse = np.unique(shower.cell_id, return_inverse=True)

        # Per-cell clustering (parallelised over cells)
        cell_indices_list = [np.where(cell_inverse == ci)[0] for ci in range(len(unique_cells))]
        cell_positions = [
            np.stack([shower.x[idx], shower.y[idx], shower.z[idx]], axis=1)
            for idx in cell_indices_list
        ]

        local_results = Parallel(n_jobs=self.n_jobs)(
            delayed(_cluster_one_cell)(self.clusterer, pos) for pos in cell_positions
        )

        # Assign globally unique labels from per-cell local labels
        global_labels = np.empty(shower.n_points, dtype=np.int64)
        total_clusters = 0
        for cell_indices, local_labels in zip(cell_indices_list, local_results):
            unique_local = np.unique(local_labels)
            label_remap = np.empty(local_labels.max() + 1, dtype=np.int64)
            for new, old in enumerate(unique_local):
                label_remap[old] = total_clusters + new
            global_labels[cell_indices] = label_remap[local_labels]
            total_clusters += len(unique_local)

        _, inverse = np.unique(global_labels, return_inverse=True)
        n = total_clusters

        e_sum = np.bincount(inverse, weights=shower.E, minlength=n)
        safe_e = np.where(e_sum > 0.0, e_sum, 1.0)
        out_x = np.bincount(inverse, weights=shower.x * shower.E, minlength=n) / safe_e
        out_y = np.bincount(inverse, weights=shower.y * shower.E, minlength=n) / safe_e
        out_z = np.bincount(inverse, weights=shower.z * shower.E, minlength=n) / safe_e

        if shower.t is not None:
            out_t = np.full(n, np.inf, dtype=np.float32)
            np.minimum.at(out_t, inverse, shower.t)
        else:
            out_t = None

        # cell_id per cluster: all deposits in a cluster share the same cell,
        # so take the cell_id of the first deposit in each cluster.
        out_cell_id = np.empty(n, dtype=np.uint64)
        first_index = np.empty(n, dtype=np.int64)
        first_index[:] = -1
        for i, label in enumerate(inverse):
            if first_index[label] < 0:
                first_index[label] = i
        out_cell_id = shower.cell_id[first_index]

        out = Shower(
            shower_id=shower.shower_id,
            x=out_x.astype(np.float32),
            y=out_y.astype(np.float32),
            z=out_z.astype(np.float32),
            E=e_sum.astype(np.float32),
            t=out_t,
            cell_id=out_cell_id,
            primary=shower.primary,
            metadata={**shower.metadata, "algorithm": self.name},
        )

        return CompressionResult(
            shower=out,
            algorithm=self.name,
            stats={
                "n_points_before": shower.n_points,
                "n_points_after": out.n_points,
                "compression_ratio": out.n_points / max(shower.n_points, 1),
                "energy_before": shower.total_energy,
                "energy_after": out.total_energy,
            },
        )
