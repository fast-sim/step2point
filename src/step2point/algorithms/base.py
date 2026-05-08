from __future__ import annotations

from abc import ABC, abstractmethod

from step2point.core.results import CompressionResult
from step2point.core.shower import Shower


class CompressionAlgorithm(ABC):
    """Base algorithm for per-shower compression kernels.

    Implementations should be stateless apart from their configuration
    and must return a new shower rather than mutating the input.
    """

    name = "base"

    @abstractmethod
    def compress(self, shower: Shower) -> CompressionResult:
        raise NotImplementedError
