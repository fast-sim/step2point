from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from step2point.core.shower import Shower


def write_step2point_hdf5(
    showers: Iterable[Shower],
    output_path: str | Path,
    *,
    algorithm: str | None = None,
    source_input: str | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    event_id: list[np.ndarray] = []
    energy: list[np.ndarray] = []
    position: list[np.ndarray] = []
    time: list[np.ndarray] = []
    cell_id: list[np.ndarray] = []
    pdg: list[np.ndarray] = []
    track_id: list[np.ndarray] = []

    primary_event_id: list[int] = []
    primary_pdg: list[int] = []
    primary_vertex: list[tuple[float, float, float]] = []
    primary_momentum: list[tuple[float, float, float]] = []

    have_time = False
    have_cell_id = False
    have_pdg = False
    have_track_id = False

    for shower in showers:
        n = shower.n_points
        event_id.append(np.full(n, shower.shower_id, dtype=np.int32))
        energy.append(np.asarray(shower.E, dtype=np.float32))
        position.append(np.stack([shower.x, shower.y, shower.z], axis=1).astype(np.float32))

        if shower.t is not None:
            have_time = True
            time.append(np.asarray(shower.t, dtype=np.float32))
        if shower.cell_id is not None:
            have_cell_id = True
            cell_id.append(np.asarray(shower.cell_id, dtype=np.uint64))
        if shower.pdg is not None:
            have_pdg = True
            pdg.append(np.asarray(shower.pdg, dtype=np.int32))
        if shower.track_id is not None:
            have_track_id = True
            track_id.append(np.asarray(shower.track_id, dtype=np.int32))

        if shower.primary:
            primary_event_id.append(int(shower.shower_id))
            primary_pdg.append(int(shower.primary.get("pdg", 0)))
            primary_vertex.append(tuple(map(float, shower.primary.get("vertex", (0.0, 0.0, 0.0)))))
            primary_momentum.append(tuple(map(float, shower.primary.get("momentum", (0.0, 0.0, 0.0)))))

    with h5py.File(output, "w") as h5:
        if algorithm is not None:
            h5.attrs["algorithm"] = algorithm
        if source_input is not None:
            h5.attrs["source_input"] = source_input

        steps = h5.create_group("steps")
        steps.create_dataset("event_id", data=np.concatenate(event_id) if event_id else np.empty(0, dtype=np.int32))
        steps.create_dataset("energy", data=np.concatenate(energy) if energy else np.empty(0, dtype=np.float32))
        steps.create_dataset(
            "position",
            data=np.concatenate(position) if position else np.empty((0, 3), dtype=np.float32),
        )

        if have_time:
            steps.create_dataset(
                "time",
                data=np.concatenate(time) if time else np.empty(0, dtype=np.float32),
            )
        if have_cell_id:
            steps.create_dataset(
                "cell_id",
                data=np.concatenate(cell_id) if cell_id else np.empty(0, dtype=np.uint64),
            )
        if have_pdg:
            steps.create_dataset(
                "pdg",
                data=np.concatenate(pdg) if pdg else np.empty(0, dtype=np.int32),
            )
        if have_track_id:
            steps.create_dataset(
                "track_id",
                data=np.concatenate(track_id) if track_id else np.empty(0, dtype=np.int32),
            )

        if primary_event_id:
            primary = h5.create_group("primary")
            primary.create_dataset("event_id", data=np.asarray(primary_event_id, dtype=np.int32))
            primary.create_dataset("pdg", data=np.asarray(primary_pdg, dtype=np.int32))
            primary.create_dataset("vertex", data=np.asarray(primary_vertex, dtype=np.float32))
            primary.create_dataset("momentum", data=np.asarray(primary_momentum, dtype=np.float32))

    return output
