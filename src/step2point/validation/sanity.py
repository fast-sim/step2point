from __future__ import annotations

import numpy as np

from step2point.core.results import ValidationResult
from step2point.validation.base import Validator


class ShowerSanityValidator(Validator):
    """Basic structural sanity checks for a shower-like object."""

    name = "shower_sanity"

    def run(self, before, after) -> ValidationResult:
        lengths = {"x": len(after.x), "y": len(after.y), "z": len(after.z), "E": len(after.E)}
        same_length = len(set(lengths.values())) == 1
        finite = bool(
            np.isfinite(after.x).all()
            and np.isfinite(after.y).all()
            and np.isfinite(after.z).all()
            and np.isfinite(after.E).all()
        )
        non_negative_energy = bool((after.E >= 0.0).all())
        has_points = len(after.E) > 0
        cell_id_consistent = True
        if after.cell_id is not None:
            cell_id_consistent = len(after.cell_id) == len(after.E)
        passed = all([same_length, finite, non_negative_energy, has_points, cell_id_consistent])
        return ValidationResult(
            self.name,
            {
                "same_length": same_length,
                "finite": finite,
                "non_negative_energy": non_negative_energy,
                "has_points": has_points,
                "cell_id_consistent": cell_id_consistent,
                "passed": passed,
            },
        )
