from __future__ import annotations

import numpy as np

from step2point.algorithms.base import CompressionAlgorithm
from step2point.core.results import CompressionResult
from step2point.core.shower import Shower


class GreedyAgglomerativeClustering(CompressionAlgorithm):
    """Greedy agglomeration within exact detector cells.

    The algorithm first partitions deposits by exact ``cell_id`` and then,
    within each cell independently, greedily merges the closest valid pair
    of clusters until no further merge is allowed.

    A merge between two clusters is allowed only if:

    - the minimum point-to-point separation between the two clusters is
      smaller than or equal to ``max_link_distance``; and
    - the maximum pairwise distance inside the merged cluster is smaller
      than or equal to ``max_cluster_distance``.

    This keeps the clustering deterministic and prevents the long chained
    clusters that a pure radius-graph connected-components algorithm would
    allow.
    """

    name = "greedy_agglomerative"

    def __init__(self, max_link_distance: float, max_cluster_distance: float | None = None) -> None:
        if max_link_distance <= 0.0:
            raise ValueError("max_link_distance must be positive.")
        if max_cluster_distance is not None and max_cluster_distance <= 0.0:
            raise ValueError("max_cluster_distance must be positive when provided.")
        if max_cluster_distance is not None and max_cluster_distance < max_link_distance:
            raise ValueError("max_cluster_distance must be greater than or equal to max_link_distance.")
        self.max_link_distance = float(max_link_distance)
        self.max_cluster_distance = (
            float(max_cluster_distance) if max_cluster_distance is not None else float(max_link_distance)
        )

    @staticmethod
    def _squared_distance_matrix(xyz: np.ndarray) -> np.ndarray:
        diff = xyz[:, None, :] - xyz[None, :, :]
        return np.sum(diff * diff, axis=2, dtype=np.float64)

    def _cluster_single_cell(self, xyz: np.ndarray) -> np.ndarray:
        n_points = xyz.shape[0]
        if n_points <= 1:
            return np.zeros(n_points, dtype=np.int64)

        distance2 = self._squared_distance_matrix(xyz.astype(np.float64, copy=False))
        max_distance2 = self.max_link_distance * self.max_link_distance
        max_cluster_distance2 = self.max_cluster_distance * self.max_cluster_distance

        clusters: list[np.ndarray] = [np.array([idx], dtype=np.int64) for idx in range(n_points)]
        while True:
            best_pair: tuple[int, int] | None = None
            best_link_distance2 = np.inf

            for left_idx in range(len(clusters) - 1):
                left_members = clusters[left_idx]
                for right_idx in range(left_idx + 1, len(clusters)):
                    right_members = clusters[right_idx]
                    link_distance2 = float(np.min(distance2[np.ix_(left_members, right_members)]))
                    if link_distance2 > max_distance2:
                        continue
                    merged_members = np.concatenate([left_members, right_members])
                    merged_extent2 = float(np.max(distance2[np.ix_(merged_members, merged_members)]))
                    if merged_extent2 > max_cluster_distance2:
                        continue
                    if link_distance2 < best_link_distance2:
                        best_link_distance2 = link_distance2
                        best_pair = (left_idx, right_idx)

            if best_pair is None:
                break

            left_idx, right_idx = best_pair
            clusters[left_idx] = np.concatenate([clusters[left_idx], clusters[right_idx]])
            del clusters[right_idx]

        labels = np.empty(n_points, dtype=np.int64)
        for label, members in enumerate(clusters):
            labels[members] = label
        return labels

    def compress(self, shower: Shower) -> CompressionResult:
        if shower.cell_id is None:
            raise ValueError("GreedyAgglomerativeClustering requires cell_id.")
        if shower.n_points == 0:
            out = Shower(
                shower_id=shower.shower_id,
                x=np.empty(0, dtype=np.float32),
                y=np.empty(0, dtype=np.float32),
                z=np.empty(0, dtype=np.float32),
                E=np.empty(0, dtype=np.float32),
                t=np.empty(0, dtype=np.float32) if shower.t is not None else None,
                cell_id=np.empty(0, dtype=np.uint64),
                primary=shower.primary,
                metadata={**shower.metadata, "algorithm": self.name},
            )
            return CompressionResult(
                shower=out,
                algorithm=self.name,
                parameters={
                    "max_link_distance": self.max_link_distance,
                    "max_cluster_distance": self.max_cluster_distance,
                },
                stats={
                    "n_points_before": 0,
                    "n_points_after": 0,
                    "compression_ratio": 0.0,
                    "energy_before": 0.0,
                    "energy_after": 0.0,
                },
                debug_data={"cluster_label": np.empty(0, dtype=np.int64)},
            )

        labels = np.full(shower.n_points, -1, dtype=np.int64)
        cell_ids = np.asarray(shower.cell_id, dtype=np.uint64)
        xyz = np.stack([shower.x, shower.y, shower.z], axis=1).astype(np.float64, copy=False)
        unique_cells, inverse = np.unique(cell_ids, return_inverse=True)
        next_label = 0
        for cell_index in range(len(unique_cells)):
            local_indices = np.where(inverse == cell_index)[0]
            local_labels = self._cluster_single_cell(xyz[local_indices])
            labels[local_indices] = local_labels + next_label
            next_label += int(np.max(local_labels)) + 1 if local_labels.size else 0

        _, cluster_inverse = np.unique(labels, return_inverse=True)
        n_clusters = int(np.max(cluster_inverse)) + 1 if cluster_inverse.size else 0
        energy = shower.E.astype(np.float64)
        energy_sum = np.bincount(cluster_inverse, weights=energy, minlength=n_clusters)
        safe_energy_sum = np.where(energy_sum > 0.0, energy_sum, 1.0)
        out_x = np.bincount(cluster_inverse, weights=shower.x.astype(np.float64) * energy, minlength=n_clusters) / safe_energy_sum
        out_y = np.bincount(cluster_inverse, weights=shower.y.astype(np.float64) * energy, minlength=n_clusters) / safe_energy_sum
        out_z = np.bincount(cluster_inverse, weights=shower.z.astype(np.float64) * energy, minlength=n_clusters) / safe_energy_sum

        out_t = None
        if shower.t is not None:
            out_t = np.bincount(
                cluster_inverse,
                weights=shower.t.astype(np.float64) * energy,
                minlength=n_clusters,
            ) / safe_energy_sum

        out_cell_id = np.empty(n_clusters, dtype=np.uint64)
        for cluster_idx in range(n_clusters):
            member_index = int(np.flatnonzero(cluster_inverse == cluster_idx)[0])
            out_cell_id[cluster_idx] = cell_ids[member_index]

        out = Shower(
            shower_id=shower.shower_id,
            x=out_x.astype(np.float32),
            y=out_y.astype(np.float32),
            z=out_z.astype(np.float32),
            E=energy_sum.astype(np.float32),
            t=None if out_t is None else out_t.astype(np.float32),
            cell_id=out_cell_id,
            primary=shower.primary,
            metadata={**shower.metadata, "algorithm": self.name},
        )
        return CompressionResult(
            shower=out,
            algorithm=self.name,
            parameters={
                "max_link_distance": self.max_link_distance,
                "max_cluster_distance": self.max_cluster_distance,
            },
            stats={
                "n_points_before": shower.n_points,
                "n_points_after": out.n_points,
                "compression_ratio": out.n_points / max(shower.n_points, 1),
                "energy_before": shower.total_energy,
                "energy_after": out.total_energy,
            },
            debug_data={"cluster_label": labels.astype(np.int64, copy=False)},
        )
