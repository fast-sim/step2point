from __future__ import annotations

from dataclasses import dataclass, field

from step2point.core.reader_base import ShowerReader
from step2point.core.results import ValidationResult


@dataclass
class PipelineReport:
    compression_stats: list[dict] = field(default_factory=list)
    validation_results: list[dict] = field(default_factory=list)


class Pipeline:
    """Simple per-shower execution pipeline.

    The pipeline is intentionally event-wise so the same public contract can be
    reused later in streaming or framework-driven processing.
    """

    def __init__(self, reader: ShowerReader, algorithm, validators=None):
        self.reader = reader
        self.algorithm = algorithm
        self.validators = validators or []

    def run(self, limit: int | None = None) -> PipelineReport:
        report = PipelineReport()
        for i, shower in enumerate(self.reader.iter_showers()):
            if limit is not None and i >= limit:
                break
            result = self.algorithm.compress(shower)
            report.compression_stats.append(result.stats)
            for validator in self.validators:
                vr: ValidationResult = validator.run(shower, result.shower)
                row = {"validator": vr.name, "shower_id": shower.shower_id, **vr.metrics}
                report.validation_results.append(row)
        return report
