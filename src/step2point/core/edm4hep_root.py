"""Direct EDM4hep ROOT reader aligned with the current step2point repository.

The uploaded step2point repository reads EDM4hep ROOT files through
`podio.root_io.Reader` and iterates over the `events` collection, extracting
MCParticles plus calorimeter-hit contributions from the configured
SimCalorimeterHit collections. This module follows that same approach and
returns canonical :class:`~step2point.core.shower.Shower` objects directly,
without converting to HDF5 first.

This reader is optional because it requires the Key4hep/PODIO Python stack
(`podio`, usually also `edm4hep`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from step2point.core.shower import Shower

DEFAULT_COLLECTIONS = (
    "ECalBarrelCollection",
    "ECalEndcapCollection",
    "HCalBarrelCollection",
    "HCalEndcapCollection",
)


@dataclass(slots=True)
class EDM4hepRootReader:
    input_path: str
    collections: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_COLLECTIONS))
    shower_limit: int | None = None
    include_primary: bool = True
    include_pdg: bool = True
    include_track_id: bool = True

    def _import_podio(self):
        try:
            from podio import root_io  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("EDM4hepRootReader requires podio.root_io. Use a Key4hep/PODIO environment.") from exc
        return root_io

    def iter_showers(self) -> Iterator[Shower]:
        root_io = self._import_podio()
        reader = root_io.Reader(self.input_path)
        events = reader.get("events")

        for iev, event in enumerate(events):
            if self.shower_limit is not None and iev >= self.shower_limit:
                break

            x: list[float] = []
            y: list[float] = []
            z: list[float] = []
            energy: list[float] = []
            time: list[float] = []
            cell_id: list[int] = []
            pdg: list[int] = []
            track_id: list[int] = []
            subdetector: list[int] = []

            primary: dict[str, object] = {}
            if self.include_primary:
                try:
                    particles = event.get("MCParticles")
                except Exception:  # pragma: no cover - backend-specific
                    particles = []
                for p in particles:
                    if not p.isCreatedInSimulation():
                        v = p.getVertex()
                        m = p.getMomentum()
                        primary = {
                            "pdg": int(p.getPDG()),
                            "vertex": (float(v.x), float(v.y), float(v.z)),
                            "momentum": (float(m.x), float(m.y), float(m.z)),
                        }
                        break

            for icol, col_name in enumerate(self.collections):
                hits = event.get(col_name)
                if not hits:
                    continue
                for hit in hits:
                    cid = int(hit.getCellID())
                    for contrib in hit.getContributions():
                        pos = contrib.getStepPosition()
                        x.append(float(pos.x))
                        y.append(float(pos.y))
                        z.append(float(pos.z))
                        energy.append(float(contrib.getEnergy()))
                        time.append(float(contrib.getTime()))
                        cell_id.append(cid)
                        subdetector.append(icol)
                        if self.include_pdg:
                            pdg.append(int(contrib.getPDG()))
                        if self.include_track_id:
                            track_id.append(int(contrib.getParticle().getObjectID().index))

            yield Shower(
                shower_id=iev,
                x=np.asarray(x, dtype=np.float32),
                y=np.asarray(y, dtype=np.float32),
                z=np.asarray(z, dtype=np.float32),
                E=np.asarray(energy, dtype=np.float32),
                t=np.asarray(time, dtype=np.float32),
                cell_id=np.asarray(cell_id, dtype=np.uint64),
                pdg=np.asarray(pdg, dtype=np.int32) if self.include_pdg else None,
                track_id=np.asarray(track_id, dtype=np.int32) if self.include_track_id else None,
                primary=primary,
                metadata={
                    "source": "edm4hep.root",
                    "collections": list(self.collections),
                    "subdetector": np.asarray(subdetector, dtype=np.uint8),
                },
            )
