"""DBSCAN density-based clustering of calorimeter step deposits."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.neighbors import NearestNeighbors

from step2point.algorithms.base import CompressionAlgorithm
from step2point.core.results import CompressionResult
from step2point.core.shower import Shower
from step2point.geometry.dd4hep.bitfield import extract_field


class DBSCANClustering(CompressionAlgorithm):
    """Density-based clustering of calorimeter step deposits using DBSCAN.

    Deposits are partitioned by the configured merge scope and clustered
    within each partition using DBSCAN on scaled `(x, y, z)` features,
    optionally including time `(t)`. Each cluster is merged into a single
    point: energy-weighted centroid position, summed energy, minimum time,
    and an approximate representative ``cell_id`` chosen from the cluster
    member closest to the cluster barycentre.
    """

    name = "dbscan"

    def __init__(
        self,
        eps: float = 1.0,
        min_samples: int = 3,
        xy_scale: float = 5.0,
        t_scale: float = 1.0,
        use_time: bool = False,
        outlier_policy: str = "nearest_cluster",
        merge_scope: str = "system_layer",
        cell_id_encoding: str | tuple[str, ...] | None = None,
        algorithm: str = "auto",
        n_jobs: int = -1,
    ) -> None:
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.xy_scale = float(xy_scale)
        self.t_scale = float(t_scale)
        self.use_time = use_time
        self.outlier_policy = outlier_policy
        self.merge_scope = merge_scope
        self.algorithm = algorithm
        self.n_jobs = n_jobs
        valid_merge_scopes = {"none", "layer", "system_layer", "cell_id", "cell_id_neighbour"}
        valid_outlier_policies = {"nearest_cluster", "standalone"}
        if self.merge_scope not in valid_merge_scopes:
            raise ValueError(f"Unsupported merge_scope {merge_scope!r}. Expected one of {sorted(valid_merge_scopes)}.")
        if self.outlier_policy not in valid_outlier_policies:
            raise ValueError(
                f"Unsupported outlier_policy {outlier_policy!r}. Expected one of {sorted(valid_outlier_policies)}."
            )

        self.cell_id_encoding: tuple[str, ...] | None
        if self.merge_scope in {"layer", "system_layer", "cell_id_neighbour"} and cell_id_encoding is None:
            raise ValueError(
                "DBSCANClustering assumes a cell_id can be decoded to define the unmergeable points. "
                "Provide cell_id_encoding directly or configure it from compact XML plus collection name(s)."
            )
        if cell_id_encoding is None:
            self.cell_id_encoding = None
        elif isinstance(cell_id_encoding, str):
            self.cell_id_encoding = (cell_id_encoding,)
        else:
            self.cell_id_encoding = tuple(cell_id_encoding)
        if self.cell_id_encoding is not None:
            for encoding in self.cell_id_encoding:
                extract_field(np.array([0], dtype=np.uint64), encoding, "layer")
                extract_field(np.array([0], dtype=np.uint64), encoding, "system")

    @staticmethod
    def _encoding_has_field(encoding: str, field_name: str) -> bool:
        try:
            extract_field(np.array([0], dtype=np.uint64), encoding, field_name)
            return True
        except ValueError:
            return False

    def _decoded_field(self, shower: Shower, field_name: str) -> np.ndarray:
        if shower.cell_id is None:
            raise ValueError("DBSCANClustering requires cell_id for field decoding.")
        if self.cell_id_encoding is None:
            raise ValueError("DBSCANClustering has no cell_id_encoding configured for decoded merge scopes.")
        cell_ids = np.asarray(shower.cell_id, dtype=np.uint64)
        if len(self.cell_id_encoding) == 1:
            return extract_field(cell_ids, self.cell_id_encoding[0], field_name)

        subdetectors = shower.metadata.get("subdetector")
        if subdetectors is None:
            raise ValueError(
                "Multiple cell_id encodings were provided, but shower.metadata['subdetector'] is absent."
            )
        subdetectors = np.asarray(subdetectors, dtype=np.int64)
        decoded = np.empty(cell_ids.shape[0], dtype=np.int64)
        unique_subdetectors = np.unique(subdetectors)
        for subdetector in unique_subdetectors:
            if subdetector < 0 or subdetector >= len(self.cell_id_encoding):
                raise ValueError(
                    f"Subdetector index {subdetector} is outside the available cell_id encodings "
                    f"(n={len(self.cell_id_encoding)})."
                )
            mask = subdetectors == subdetector
            decoded[mask] = extract_field(cell_ids[mask], self.cell_id_encoding[int(subdetector)], field_name)
        return decoded

    def _decoded_neighbour_axis_field(self, shower: Shower) -> np.ndarray:
        if shower.cell_id is None:
            raise ValueError("DBSCANClustering requires cell_id for neighbour decoding.")
        if self.cell_id_encoding is None:
            raise ValueError("DBSCANClustering has no cell_id_encoding configured for neighbour decoding.")
        cell_ids = np.asarray(shower.cell_id, dtype=np.uint64)
        if len(self.cell_id_encoding) == 1:
            encoding = self.cell_id_encoding[0]
            field_name = "y" if self._encoding_has_field(encoding, "y") else "z"
            return extract_field(cell_ids, encoding, field_name)

        subdetectors = shower.metadata.get("subdetector")
        if subdetectors is None:
            raise ValueError(
                "Multiple cell_id encodings were provided, but shower.metadata['subdetector'] is absent."
            )
        subdetectors = np.asarray(subdetectors, dtype=np.int64)
        decoded = np.empty(cell_ids.shape[0], dtype=np.int64)
        unique_subdetectors = np.unique(subdetectors)
        for subdetector in unique_subdetectors:
            if subdetector < 0 or subdetector >= len(self.cell_id_encoding):
                raise ValueError(
                    f"Subdetector index {subdetector} is outside the available cell_id encodings "
                    f"(n={len(self.cell_id_encoding)})."
                )
            encoding = self.cell_id_encoding[int(subdetector)]
            field_name = "y" if self._encoding_has_field(encoding, "y") else "z"
            mask = subdetectors == subdetector
            decoded[mask] = extract_field(cell_ids[mask], encoding, field_name)
        return decoded

    @staticmethod
    def _connected_components_from_cell_neighbours(
        systems: np.ndarray,
        layers: np.ndarray,
        x_bins: np.ndarray,
        secondary_bins: np.ndarray,
    ) -> np.ndarray:
        key_dtype = np.dtype([("system", np.int64), ("layer", np.int64), ("x", np.int64), ("second", np.int64)])
        keys = np.empty(len(systems), dtype=key_dtype)
        keys["system"] = systems
        keys["layer"] = layers
        keys["x"] = x_bins
        keys["second"] = secondary_bins
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        key_to_index = {
            (
                int(unique_keys["system"][idx]),
                int(unique_keys["layer"][idx]),
                int(unique_keys["x"][idx]),
                int(unique_keys["second"][idx]),
            ): idx
            for idx in range(len(unique_keys))
        }
        component_by_unique = np.full(len(unique_keys), -1, dtype=np.int64)
        current_component = 0
        for start_idx in range(len(unique_keys)):
            if component_by_unique[start_idx] != -1:
                continue
            stack = [start_idx]
            component_by_unique[start_idx] = current_component
            while stack:
                idx = stack.pop()
                system = int(unique_keys["system"][idx])
                layer = int(unique_keys["layer"][idx])
                x_bin = int(unique_keys["x"][idx])
                second_bin = int(unique_keys["second"][idx])
                neighbours = (
                    (system, layer, x_bin - 1, second_bin),
                    (system, layer, x_bin + 1, second_bin),
                    (system, layer, x_bin, second_bin - 1),
                    (system, layer, x_bin, second_bin + 1),
                )
                for neighbour in neighbours:
                    neighbour_idx = key_to_index.get(neighbour)
                    if neighbour_idx is None or component_by_unique[neighbour_idx] != -1:
                        continue
                    component_by_unique[neighbour_idx] = current_component
                    stack.append(neighbour_idx)
            current_component += 1
        return component_by_unique[inverse]

    def _partition_indices(self, shower: Shower) -> list[np.ndarray]:
        if shower.n_points == 0:
            return []
        if self.merge_scope == "none":
            return [np.arange(shower.n_points, dtype=np.int64)]
        if shower.cell_id is None:
            raise ValueError(f"DBSCANClustering with merge_scope={self.merge_scope!r} requires cell_id.")
        if self.merge_scope == "cell_id":
            _, inverse = np.unique(np.asarray(shower.cell_id, dtype=np.uint64), return_inverse=True)
        elif self.merge_scope == "cell_id_neighbour":
            inverse = self._connected_components_from_cell_neighbours(
                self._decoded_field(shower, "system"),
                self._decoded_field(shower, "layer"),
                self._decoded_field(shower, "x"),
                self._decoded_neighbour_axis_field(shower),
            )
        elif self.merge_scope == "layer":
            _, inverse = np.unique(self._decoded_field(shower, "layer"), return_inverse=True)
        else:
            systems = self._decoded_field(shower, "system")
            layers = self._decoded_field(shower, "layer")
            key_dtype = np.dtype([("system", np.int64), ("layer", np.int64)])
            keys = np.empty(shower.n_points, dtype=key_dtype)
            keys["system"] = systems
            keys["layer"] = layers
            _, inverse = np.unique(keys, return_inverse=True)
        return [np.where(inverse == idx)[0] for idx in range(int(np.max(inverse)) + 1)]

    def _prepare_features(self, shower: Shower, global_mask: np.ndarray) -> np.ndarray:
        xyz = np.stack([shower.x[global_mask], shower.y[global_mask], shower.z[global_mask]], axis=1).astype(np.float32)
        xyz_scaled = xyz / self.xy_scale
        if self.use_time and shower.t is not None:
            t_slice = shower.t[global_mask].astype(np.float32)
            t_median = np.median(t_slice)
            t_scaled = ((t_slice - t_median) / self.t_scale).reshape(-1, 1)
            return np.hstack([xyz_scaled, t_scaled]).astype(np.float32)
        return xyz_scaled

    def _fit_predict_partition(self, features: np.ndarray) -> np.ndarray:
        model = SklearnDBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            algorithm=self.algorithm,
            n_jobs=self.n_jobs,
        )
        return model.fit_predict(features)

    def compress(self, shower: Shower) -> CompressionResult:
        if self.use_time and shower.t is None:
            raise ValueError("use_time=True but shower has no time data.")

        labels = np.full(shower.n_points, -1, dtype=np.int64)
        total_clusters = 0

        for global_mask in self._partition_indices(shower):
            n_slice = len(global_mask)
            if n_slice < max(self.min_samples, 2):
                labels[global_mask] = np.arange(total_clusters, total_clusters + n_slice)
                total_clusters += n_slice
                continue

            features = self._prepare_features(shower, global_mask)
            predicted = self._fit_predict_partition(features)

            n_new = len(set(predicted)) - (1 if -1 in predicted else 0)
            is_cluster = predicted >= 0
            is_noise = predicted == -1

            if n_new > 0 and np.any(is_cluster):
                predicted[is_cluster] += total_clusters
                total_clusters += n_new
                if np.any(is_noise):
                    if self.outlier_policy == "nearest_cluster":
                        nn = NearestNeighbors(n_neighbors=1, n_jobs=self.n_jobs)
                        nn.fit(features[is_cluster])
                        _, idx = nn.kneighbors(features[is_noise])
                        predicted[is_noise] = predicted[is_cluster][idx.ravel()]
                    else:
                        n_noise = int(np.count_nonzero(is_noise))
                        predicted[is_noise] = np.arange(total_clusters, total_clusters + n_noise, dtype=np.int64)
                        total_clusters += n_noise
            elif np.any(is_noise):
                if self.outlier_policy == "nearest_cluster":
                    predicted[is_noise] = total_clusters
                    total_clusters += 1
                else:
                    n_noise = int(np.count_nonzero(is_noise))
                    predicted[is_noise] = np.arange(total_clusters, total_clusters + n_noise, dtype=np.int64)
                    total_clusters += n_noise

            labels[global_mask] = predicted

        keep = labels >= 0
        kept_labels = labels[keep]
        if kept_labels.size == 0:
            out = Shower(
                shower_id=shower.shower_id,
                x=np.empty(0, dtype=np.float32),
                y=np.empty(0, dtype=np.float32),
                z=np.empty(0, dtype=np.float32),
                E=np.empty(0, dtype=np.float32),
                t=np.empty(0, dtype=np.float32) if shower.t is not None else None,
                primary=shower.primary,
                metadata={**shower.metadata, "algorithm": self.name},
            )
        else:
            kept_x = shower.x[keep].astype(np.float64)
            kept_y = shower.y[keep].astype(np.float64)
            kept_z = shower.z[keep].astype(np.float64)
            kept_E = shower.E[keep].astype(np.float64)
            kept_cell_id = None if shower.cell_id is None else np.asarray(shower.cell_id[keep], dtype=np.uint64)
            _, inverse = np.unique(kept_labels, return_inverse=True)
            n = len(_)
            e_sum = np.bincount(inverse, weights=kept_E, minlength=n)
            safe_e = np.where(e_sum > 0.0, e_sum, 1.0)
            out_x = np.bincount(inverse, weights=kept_x * kept_E, minlength=n) / safe_e
            out_y = np.bincount(inverse, weights=kept_y * kept_E, minlength=n) / safe_e
            out_z = np.bincount(inverse, weights=kept_z * kept_E, minlength=n) / safe_e

            if shower.t is not None:
                kept_t = shower.t[keep].astype(np.float64)
                out_t = np.full(n, np.inf)
                np.minimum.at(out_t, inverse, kept_t)
                out_t = out_t.astype(np.float32)
            else:
                out_t = None

            if kept_cell_id is not None:
                out_cell_id = np.empty(n, dtype=np.uint64)
                for cluster_idx in range(n):
                    cluster_mask = inverse == cluster_idx
                    cluster_dx = kept_x[cluster_mask] - out_x[cluster_idx]
                    cluster_dy = kept_y[cluster_mask] - out_y[cluster_idx]
                    cluster_dz = kept_z[cluster_mask] - out_z[cluster_idx]
                    nearest_index = int(np.argmin(cluster_dx * cluster_dx + cluster_dy * cluster_dy + cluster_dz * cluster_dz))
                    out_cell_id[cluster_idx] = kept_cell_id[cluster_mask][nearest_index]
            else:
                out_cell_id = None

            out = Shower(
                shower_id=shower.shower_id,
                x=out_x.astype(np.float32),
                y=out_y.astype(np.float32),
                z=out_z.astype(np.float32),
                E=e_sum.astype(np.float32),
                t=out_t,
                cell_id=out_cell_id,
                primary=shower.primary,
                metadata={**shower.metadata, "algorithm": self.name, "approximate_cell_id": out_cell_id is not None},
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
            debug_data={"cluster_label": labels.astype(np.int64, copy=False)},
        )
