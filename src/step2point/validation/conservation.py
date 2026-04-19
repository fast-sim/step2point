from __future__ import annotations

from step2point.core.results import ValidationResult
from step2point.metrics.energy import aggregate_cell_energy, energy_ratio
from step2point.validation.base import Validator


class EnergyConservationValidator(Validator):
    name = "energy_conservation"

    def run(self, before, after) -> ValidationResult:
        ratio = energy_ratio(before, after)
        return ValidationResult(self.name, {"energy_ratio": ratio, "passed": abs(ratio - 1.0) < 1e-9})


class CellCountRatioValidator(Validator):
    name = "cell_count_ratio"

    def run(self, before, after) -> ValidationResult:
        if before.cell_id is None or after.cell_id is None:
            return ValidationResult(self.name, {"cell_count_ratio": float("nan")})
        _, e_pre = aggregate_cell_energy(before)
        _, e_post = aggregate_cell_energy(after)
        ratio = len(e_post) / len(e_pre) if len(e_pre) else float("nan")
        return ValidationResult(self.name, {"cell_count_ratio": ratio})
