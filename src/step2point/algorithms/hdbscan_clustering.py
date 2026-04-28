"""HDBSCAN density-based clustering of calorimeter step deposits.

This algorithm clusters deposits using HDBSCAN within each decoded
``(system, layer)`` partition. Features are scaled x, y coordinates and,
if enabled and available, time relative to the layer median. Each cluster
is merged into a single point: energy-weighted centroid position, summed
energy, minimum time.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
from sklearn.neighbors import NearestNeighbors

from step2point.algorithms.base import CompressionAlgorithm
from step2point.core.results import CompressionResult
from step2point.core.shower import Shower
from step2point.geometry.dd4hep.bitfield import extract_field


class HDBSCANClustering(CompressionAlgorithm):
    """Density-based clustering of calorimeter step deposits.

    Deposits are partitioned by decoded ``(system, layer)`` and clustered
    within each partition using HDBSCAN on scaled (x, y) features,
    optionally including time (t) (if ``use_time`` is True). Each cluster
    is then merged into a single point: energy-weighted centroid position,
    summed energy, minimum time.

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
    xy_scale : float
        Divide x, y coordinates by this value before clustering (mm).
        Normalises spatial distances so that 1.0 in scaled space
        corresponds to roughly one cell width.  When ``use_time`` is
        True, this also ensures spatial and temporal features are on
        comparable magnitudes.  The value is detector-specific
        (default 5.0 mm matches ODD calorimeter cells).
    t_scale : float
        Divide (t - layer median) by this value before clustering (ns).
        Normalises the temporal dimension so it contributes meaningfully
        alongside the scaled spatial features.  Only used when time is
        present and ``use_time`` is True.
    use_time : bool
        Whether to include time as a clustering feature (default False).
        When True, time must be present in the input shower or a
        ``ValueError`` is raised.
    cell_id_encoding : str or tuple[str, ...]
        Cell-ID encoding string(s) used to decode fields such as
        ``system`` and ``layer`` from each deposit `cell_id`. A single
        string is used for all points. A tuple provides one encoding per
        input collection / system slot, matched against
        ``shower.metadata["subdetector"]``.
    algorithm : str
        Internal neighbour-search method used by scikit-learn's HDBSCAN:
        `"auto"` (default), `"brute"`, `"kd_tree"`, or `"ball_tree"`
        (default: `"auto"`). Use `"brute"` for the most reproducible
        reference outputs across machines. `"auto"` may choose different
        methods depending on the environment, which can slightly change
        cluster boundaries and therefore the compressed output.
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
        xy_scale: float = 5.0,
        t_scale: float = 1.0,
        use_time: bool = False,
        cell_id_encoding: str | tuple[str, ...] | None = None,
        algorithm: str = "auto",
        n_jobs: int = -1,
    ) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.xy_scale = xy_scale
        self.t_scale = t_scale
        self.use_time = use_time
        self.algorithm = algorithm
        self.n_jobs = n_jobs
        if cell_id_encoding is None:
            raise ValueError(
                "HDBSCANClustering assumes a cell_id can be decoded to define the unmergeable points. "
                "Provide cell_id_encoding directly or configure it from compact XML plus collection name(s)."
            )
        if isinstance(cell_id_encoding, str):
            self.cell_id_encoding = (cell_id_encoding,)
        else:
            self.cell_id_encoding = tuple(cell_id_encoding)
        for encoding in self.cell_id_encoding:
            extract_field(np.array([0], dtype=np.uint64), encoding, "layer")
            extract_field(np.array([0], dtype=np.uint64), encoding, "system")

    def _decoded_field(self, shower: Shower, field_name: str) -> np.ndarray:
        if shower.cell_id is None:
            raise ValueError("HDBSCANClustering requires cell_id for field decoding.")
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
            decoded[mask] = extract_field(
                cell_ids[mask],
                self.cell_id_encoding[int(subdetector)],
                field_name,
            )
        return decoded

    def compress(self, shower: Shower) -> CompressionResult:
        if shower.cell_id is None:
            raise ValueError("HDBSCANClustering requires cell_id to decode layer boundaries.")
        if self.use_time and shower.t is None:
            raise ValueError("use_time=True but shower has no time data.")

        layers = self._decoded_field(shower, "layer")
        systems = self._decoded_field(shower, "system")

        labels = np.full(shower.n_points, -1, dtype=np.int64)
        total_clusters = 0

        for system in np.unique(systems):
            system_mask = systems == system
            layers_sub = layers[system_mask]

            for layer in np.unique(layers_sub):
                layer_mask_local = layers_sub == layer
                global_mask = np.where(system_mask)[0][layer_mask_local]

                n_slice = len(global_mask)
                if n_slice < max(self.min_samples, 2):
                    # Too few points for HDBSCAN - each becomes its own cluster
                    labels[global_mask] = np.arange(
                        total_clusters, total_clusters + n_slice
                    )
                    total_clusters += n_slice
                    continue

                xy = np.stack([shower.x[global_mask], shower.y[global_mask]], axis=1).astype(np.float32)
                xy_scaled = xy / self.xy_scale

                if self.use_time and shower.t is not None:
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
                        # Reassign noise to nearest cluster (energy-conserving)
                        nn = NearestNeighbors(n_neighbors=1, n_jobs=self.n_jobs)
                        nn.fit(features[is_cluster])
                        _, idx = nn.kneighbors(features[is_noise])
                        predicted[is_noise] = predicted[is_cluster][idx.ravel()]
                elif np.any(is_noise):
                    # No clusters found - bundle all noise into one cluster
                    predicted[is_noise] = total_clusters
                    total_clusters += 1

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
            debug_data={"cluster_label": labels.astype(np.int64, copy=False)},
        )
