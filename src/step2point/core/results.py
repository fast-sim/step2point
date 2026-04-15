from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from step2point.core.shower import Shower


@dataclass(slots=True)
class CompressionResult:
    shower: Shower
    algorithm: str
    parameters: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationResult:
    name: str
    metrics: dict[str, Any]
