"""HDBSCAN density-based clustering of calorimeter step deposits.

This algorithm clusters deposits using HDBSCAN within each (subdetector,
layer) partition. Features are scaled x, y coordinates and, when available,
time relative to the layer median. Each cluster is merged into a single
point: energy-weighted centroid position, summed energy, minimum time.


Requires ``scikit-learn`` (install via ``pip install step2point[hdbscan]``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from step2point.algorithms.base import CompressionAlgorithm
from step2point.core.results import CompressionResult
from step2point.core.shower import Shower


def _default_layer_extractor(cell_ids: np.ndarray) -> np.ndarray:
    """Extract layer number from cell_id using the ODD bit layout.

    The ODD (Open Data Detector) encodes the layer in 9 bits starting at
    bit 19 of the 64-bit cell ID.
    """
    return (cell_ids.astype(np.int64) >> 19) & 0x1FF



class HDBSCANClustering(CompressionAlgorithm):
    """Density-based clustering of calorimeter step deposits.

    Deposits are partitioned by (subdetector, layer) and clustered within
    each partition using HDBSCAN on scaled (x, y) features, optionally
    including time (t) when available.  Each cluster is then merged into
    a single point: energy-weighted centroid position, summed energy,
    minimum time.

    Parameters
    ----------
    min_cluster_size : int
        HDBSCAN ``min_cluster_size`` parameter.
    min_samples : int
        HDBSCAN ``min_samples`` parameter.
    cluster_selection_epsilon : float
        HDBSCAN builds a hierarchy of clusters at different density
        levels and by default (epsilon=0) picks the most persistent ones,
        which can produce many small, high-density clusters.  When
        epsilon > 0, clusters separated by a distance below this threshold
        are merged rather than split, producing fewer, larger clusters.
        A small value (e.g. 0.5 - 1.0 in scaled feature space) prevents
        over-fragmenting dense shower cores while still separating
        genuinely distinct deposits.
    low_energy_deposits_handler : str
        Strategy for low-energy deposits that HDBSCAN labels as noise:
        ``"nn"``: reassign to the nearest cluster (default, energy-conserving).
        ``"singleton"``: each deposit becomes its own cluster.
        ``"layer"``: all unclustered deposits in a layer are bundled into one cluster.
        ``"drop"``: discard (loses energy).
    xy_scale : float
        Divide x, y coordinates by this value before clustering (mm).
        Normalises spatial distances so that 1.0 in scaled space
        corresponds to roughly one cell width, putting spatial and
        temporal features on comparable footing.  The value is
        detector-specific (default 5.0 mm matches ODD calorimeter cells).
    t_scale : float
        Divide (t - layer median) by this value before clustering (ns).
        Normalises the temporal dimension so it contributes meaningfully
        alongside the scaled spatial features.  Only used when time is
        present in the input shower.
    layer_extractor : callable, str, or None
        How to extract layer IDs from cell IDs.  Can be a callable
        ``f(cell_ids: ndarray) -> ndarray``, a DD4hep ID encoding string
        (e.g. ``"system:8,barrel:3,layer:19:9"``), or ``None`` to use
        the ODD default ``(cell_id >> 19) & 0x1FF``.
    algorithm : str
        HDBSCAN tree-building algorithm: ``"auto"``, ``"brute"``,
        ``"kd_tree"``, or ``"ball_tree"`` (default ``"auto"``).
        ``"auto"`` lets sklearn choose based on input size.  Fixing
        the algorithm (e.g. ``"kd_tree"``) can improve cross-platform
        reproducibility by avoiding dispatch differences.
    n_jobs : int
        Number of parallel jobs for HDBSCAN and nearest-neighbour queries.
        ``-1`` uses all cores (default).  ``1`` forces single-threaded
        execution, which improves reproducibility across runs.
    """

    name = "hdbscan_clustering"

    def __init__(
        self,
        min_cluster_size: int,
        min_samples: int,
        cluster_selection_epsilon: float = 0.0,
        low_energy_deposits_handler: str = "nn",
        xy_scale: float = 5.0,
        t_scale: float = 1.0,
        layer_extractor: Callable[[np.ndarray], np.ndarray] | str | None = None,
        algorithm: str = "auto",
        n_jobs: int = -1,
    ) -> None:
        if low_energy_deposits_handler not in {"drop", "singleton", "layer", "nn"}:
            raise ValueError(
                f"low_energy_deposits_handler must be 'drop', 'singleton', 'layer', or 'nn', got '{low_energy_deposits_handler}'."
            )
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.low_energy_deposits_handler = low_energy_deposits_handler
        self.xy_scale = xy_scale
        self.t_scale = t_scale
        self.algorithm = algorithm
        self.n_jobs = n_jobs
        if isinstance(layer_extractor, str):
            from step2point.geometry.dd4hep.bitfield import extract_field

            encoding = layer_extractor
            extract_field(np.array([0], dtype=np.uint64), encoding)
            self.layer_extractor = lambda cell_ids: extract_field(cell_ids, encoding)
        else:
            self.layer_extractor = layer_extractor or _default_layer_extractor

    @staticmethod
    def _import_sklearn():
        try:
            from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
            from sklearn.neighbors import NearestNeighbors
        except ImportError as exc:
            raise ImportError(
                "HDBSCANClustering requires scikit-learn. "
                "Install it with: pip install step2point[hdbscan]"
            ) from exc
        return SklearnHDBSCAN, NearestNeighbors

    def compress(self, shower: Shower) -> CompressionResult:
        if shower.cell_id is None:
            raise ValueError("HDBSCANClustering requires cell_id for layer extraction.")

        SklearnHDBSCAN, NearestNeighbors = self._import_sklearn()

        layers = self.layer_extractor(shower.cell_id)
        subdetectors = shower.metadata.get("subdetector")
        if subdetectors is None:
            subdetectors = np.zeros(shower.n_points, dtype=np.uint8)
        subdetectors = np.asarray(subdetectors)

        labels = np.full(shower.n_points, -1, dtype=np.int64)
        total_clusters = 0

        for subdet in np.unique(subdetectors):
            subdet_mask = subdetectors == subdet
            layers_sub = layers[subdet_mask]

            for layer in np.unique(layers_sub):
                layer_mask_local = layers_sub == layer
                global_mask = np.where(subdet_mask)[0][layer_mask_local]

                n_slice = len(global_mask)
                if n_slice < max(self.min_samples, 2):
                    if self.low_energy_deposits_handler != "drop":
                        labels[global_mask] = np.arange(
                            total_clusters, total_clusters + n_slice
                        )
                        total_clusters += n_slice
                    continue

                xy = np.stack([shower.x[global_mask], shower.y[global_mask]], axis=1).astype(np.float32)
                xy_scaled = xy / self.xy_scale

                if shower.t is not None:
                    t_layer = shower.t[global_mask].astype(np.float32)
                    t_median = np.median(t_layer)
                    t_scaled = ((t_layer - t_median) / self.t_scale).reshape(-1, 1)
                    features = np.hstack([xy_scaled, t_scaled]).astype(np.float32)
                else:
                    features = xy_scaled

                model = SklearnHDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    cluster_selection_epsilon=self.cluster_selection_epsilon,
                    algorithm=self.algorithm,
                    n_jobs=self.n_jobs,
                    copy=False,
                )
                predicted = model.fit_predict(features)

                n_new = len(set(predicted)) - (1 if -1 in predicted else 0)
                is_cluster = predicted >= 0
                is_noise = predicted == -1

                if n_new > 0 and np.any(is_cluster):
                    predicted[is_cluster] += total_clusters
                    total_clusters += n_new
                    if np.any(is_noise):
                        if self.low_energy_deposits_handler == "nn":
                            nn = NearestNeighbors(n_neighbors=1, n_jobs=self.n_jobs)
                            nn.fit(features[is_cluster])
                            _, idx = nn.kneighbors(features[is_noise])
                            predicted[is_noise] = predicted[is_cluster][idx.ravel()]
                        elif self.low_energy_deposits_handler == "singleton":
                            noise_idx = np.where(is_noise)[0]
                            n_noise = len(noise_idx)
                            predicted[noise_idx] = np.arange(
                                total_clusters, total_clusters + n_noise
                            )
                            total_clusters += n_noise
                        elif self.low_energy_deposits_handler == "layer":
                            predicted[is_noise] = total_clusters
                            total_clusters += 1
                        # drop: leave as -1
                elif np.any(is_noise):
                    if self.low_energy_deposits_handler in ("nn", "layer"):
                        predicted[is_noise] = total_clusters
                        total_clusters += 1
                    elif self.low_energy_deposits_handler == "singleton":
                        noise_idx = np.where(is_noise)[0]
                        n_noise = len(noise_idx)
                        predicted[noise_idx] = np.arange(
                            total_clusters, total_clusters + n_noise
                        )
                        total_clusters += n_noise
                    # drop: leave as -1

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

            out = Shower(
                shower_id=shower.shower_id,
                x=out_x.astype(np.float32),
                y=out_y.astype(np.float32),
                z=out_z.astype(np.float32),
                E=e_sum.astype(np.float32),
                t=out_t,
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
