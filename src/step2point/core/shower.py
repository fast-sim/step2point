from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class Shower:
    """Canonical per-shower container.

    The class is intentionally a thin holder of array-like quantities so it can
    later be mapped cleanly to a C++ struct/class without redesigning the
    library API.
    """

    shower_id: int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    E: np.ndarray
    t: np.ndarray | None = None
    cell_id: np.ndarray | None = None
    pdg: np.ndarray | None = None
    track_id: np.ndarray | None = None
    primary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.ascontiguousarray(self.x, dtype=np.float32)
        self.y = np.ascontiguousarray(self.y, dtype=np.float32)
        self.z = np.ascontiguousarray(self.z, dtype=np.float32)
        self.E = np.ascontiguousarray(self.E, dtype=np.float32)
        if self.t is not None:
            self.t = np.ascontiguousarray(self.t, dtype=np.float32)
        if self.cell_id is not None:
            self.cell_id = np.ascontiguousarray(self.cell_id, dtype=np.uint64)
        if self.pdg is not None:
            self.pdg = np.ascontiguousarray(self.pdg, dtype=np.int32)
        if self.track_id is not None:
            self.track_id = np.ascontiguousarray(self.track_id, dtype=np.int32)

        n = len(self.E)
        for name in ("x", "y", "z"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"{name} length {len(arr)} does not match E length {n}")
        for name in ("t", "cell_id", "pdg", "track_id"):
            arr = getattr(self, name)
            if arr is not None and len(arr) != n:
                raise ValueError(f"{name} length {len(arr)} does not match E length {n}")
        if np.any(self.E < 0):
            raise ValueError("Deposited energies must be non-negative.")

    @property
    def n_points(self) -> int:
        return len(self.E)

    @property
    def total_energy(self) -> float:
        return float(np.sum(self.E, dtype=np.float64))

    def to_xyzE(self) -> np.ndarray:
        return np.stack([self.x, self.y, self.z, self.E], axis=1)

    def copy(self) -> "Shower":
        return Shower(
            shower_id=self.shower_id,
            x=self.x.copy(),
            y=self.y.copy(),
            z=self.z.copy(),
            E=self.E.copy(),
            t=None if self.t is None else self.t.copy(),
            cell_id=None if self.cell_id is None else self.cell_id.copy(),
            pdg=None if self.pdg is None else self.pdg.copy(),
            track_id=None if self.track_id is None else self.track_id.copy(),
            primary=dict(self.primary),
            metadata=dict(self.metadata),
        )
