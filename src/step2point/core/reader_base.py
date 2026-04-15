from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from step2point.core.shower import Shower


class ShowerReader(ABC):
    """Language-neutral reader contract.

    Readers translate external formats such as HDF5 or EDM4hep ROOT into the
    canonical per-shower representation used by the library.
    """

    @abstractmethod
    def iter_showers(self) -> Iterator[Shower]:
        """Yield one canonical Shower at a time."""
        raise NotImplementedError
