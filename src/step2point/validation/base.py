from __future__ import annotations

from abc import ABC, abstractmethod

from step2point.core.results import ValidationResult


class Validator(ABC):
    name = "base"

    @abstractmethod
    def run(self, before, after) -> ValidationResult:
        raise NotImplementedError
