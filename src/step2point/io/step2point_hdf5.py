from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import h5py
import numpy as np

from step2point.core.reader_base import ShowerReader
from step2point.core.shower import Shower


@dataclass
class Step2PointHDF5Reader(ShowerReader):
    input_path: str
    shower_limit: int | None = None

    def iter_showers(self) -> Iterator[Shower]:
        with h5py.File(self.input_path, "r") as h5:
            event_ids = np.asarray(h5["steps"]["event_id"], dtype=np.int32)
            energy = np.asarray(h5["steps"]["energy"], dtype=np.float32)
            position = np.asarray(h5["steps"]["position"], dtype=np.float32)
            time = np.asarray(h5["steps"]["time"], dtype=np.float32)
            cell_id = np.asarray(h5["steps"]["cell_id"], dtype=np.uint64)
            pdg = np.asarray(h5["steps"]["pdg"], dtype=np.int32)
            unique_ids = np.unique(event_ids)
            if self.shower_limit is not None:
                unique_ids = unique_ids[: self.shower_limit]

            primary_map: dict[int, dict] = {}
            if "primary" in h5 and "event_id" in h5["primary"]:
                p_evt = np.asarray(h5["primary"]["event_id"], dtype=np.int32)
                p_pdg = np.asarray(h5["primary"]["pdg"], dtype=np.int32)
                p_vertex = np.asarray(h5["primary"]["vertex"], dtype=np.float32)
                p_mom = np.asarray(h5["primary"]["momentum"], dtype=np.float32)
                for i, eid in enumerate(p_evt):
                    primary_map[int(eid)] = {
                        "pdg": int(p_pdg[i]),
                        "vertex": tuple(map(float, p_vertex[i])),
                        "momentum": tuple(map(float, p_mom[i])),
                    }

            for shower_id in unique_ids:
                mask = event_ids == shower_id
                yield Shower(
                    shower_id=int(shower_id),
                    x=position[mask, 0],
                    y=position[mask, 1],
                    z=position[mask, 2],
                    E=energy[mask],
                    t=time[mask],
                    cell_id=cell_id[mask],
                    pdg=pdg[mask],
                    primary=primary_map.get(int(shower_id), {}),
                    metadata={"source": "hdf5"},
                )
