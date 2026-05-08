from __future__ import annotations

import numpy as np

from step2point.algorithms.base import CompressionAlgorithm
from step2point.core.results import CompressionResult
from step2point.core.shower import Shower


class IdentityCompression(CompressionAlgorithm):
    """Identity compression.

    No change of the shower. Serves as an example.
    """

    name = "identity"

    def compress(self, shower: Shower) -> CompressionResult:
        return CompressionResult(
            shower=shower,
            algorithm=self.name,
            stats={
                "n_points_before": shower.n_points,
                "n_points_after": shower.n_points,
                "compression_ratio": 1.0,
                "energy_before": shower.total_energy,
                "energy_after": shower.total_energy,
            },
            debug_data={"cluster_label": np.arange(shower.n_points, dtype=np.int64)},
        )
