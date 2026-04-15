from __future__ import annotations

import numpy as np

from step2point.algorithms.base import CompressionAlgorithm
from step2point.core.results import CompressionResult
from step2point.core.shower import Shower
from step2point.cpp_backend import cpp_available, merge_within_cell as cpp_merge_within_cell


class MergeWithinCell(CompressionAlgorithm):
    """Merging within the detector cell.

    All steps that fall within the same detector cell (have the same cell ID)
    are merged, no matter what are the other properties.

    TODO Implement a time window to merge only steps within, and ignore the rest.
    """
    name = "merge_within_cell"

    def __init__(self, backend: str = "auto") -> None:
        if backend not in {"auto", "python", "cpp"}:
            raise ValueError("backend must be 'auto', 'python', or 'cpp'.")
        self.backend = backend

    def compress(self, shower: Shower) -> CompressionResult:
        if self.backend == "cpp":
            return cpp_merge_within_cell(shower)
        if self.backend == "auto" and cpp_available():
            return cpp_merge_within_cell(shower)
        return self._compress_python(shower)

    def _compress_python(self, shower: Shower) -> CompressionResult:
        if shower.cell_id is None:
            raise ValueError("MergeWithinCell requires cell_id.")
        unique_cells, inverse = np.unique(shower.cell_id, return_inverse=True)
        n = len(unique_cells)
        e_sum = np.bincount(inverse, weights=shower.E, minlength=n)
        safe_e = np.where(e_sum > 0.0, e_sum, 1.0)
        x = np.bincount(inverse, weights=shower.x * shower.E, minlength=n) / safe_e
        y = np.bincount(inverse, weights=shower.y * shower.E, minlength=n) / safe_e
        z = np.bincount(inverse, weights=shower.z * shower.E, minlength=n) / safe_e
        t = None
        if shower.t is not None:
            t = np.bincount(inverse, weights=shower.t * shower.E, minlength=n) / safe_e
        out = Shower(
            shower_id=shower.shower_id,
            x=x.astype(np.float32),
            y=y.astype(np.float32),
            z=z.astype(np.float32),
            E=e_sum.astype(np.float32),
            t=None if t is None else t.astype(np.float32),
            cell_id=unique_cells.astype(np.uint64),
            primary=shower.primary,
            metadata={**shower.metadata, "algorithm": self.name, "backend": "python"},
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
