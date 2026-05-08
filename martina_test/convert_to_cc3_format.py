"""
Convert a step2point HDF5 file to the format expected by create_cc3_showers.py.

Usage:
    python convert_to_cc3_format.py input.h5 output.h5
"""

import os
import sys

import h5py
import numpy as np
from tqdm import tqdm


def convert(input_path: str):
    print(f"Reading {input_path}...")
    with h5py.File(input_path, "r") as h5:
        # --- primary info ---
        p_evt = np.asarray(h5["primary"]["event_id"], dtype=np.int32)
        p_mom = np.asarray(h5["primary"]["momentum"], dtype=np.float32)  # (N, 3)
        p_vertex = np.asarray(h5["primary"]["vertex"], dtype=np.float32)  # (N, 3)

        # --- steps info ---
        s_evt = np.asarray(h5["steps"]["event_id"], dtype=np.int32)
        s_pos = np.asarray(h5["steps"]["position"], dtype=np.float32)  # (M, 3)
        s_ene = np.asarray(h5["steps"]["energy"], dtype=np.float32)  # (M,)

    unique_ids = np.unique(p_evt)
    n_events = len(unique_ids)
    print(f"Found {n_events} events.")

    # --- compute incident energy per shower from momentum ---
    # E = |p| (massless approximation, valid for photons/electrons at high energy)
    # If you have mass: E = sqrt(mass^2 + |p|^2)
    incident_energies = np.linalg.norm(p_mom, axis=1).astype(np.float32)  # (N,)

    # --- build events array: (N, max_hits, 4) = x, y, z, energy ---
    # first pass: find max hits per event
    max_hits = 0
    for eid in tqdm(unique_ids, desc="Finding max hits"):
        mask = s_evt == eid
        max_hits = max(max_hits, mask.sum())
    print(f"Max hits per event: {max_hits}")

    events = np.zeros((n_events, max_hits, 4), dtype=np.float32)
    for i, eid in enumerate(tqdm(unique_ids, desc="Building events array")):
        mask = s_evt == eid
        n_hits = mask.sum()
        events[i, :n_hits, 0] = s_pos[mask, 0]  # x
        events[i, :n_hits, 1] = s_pos[mask, 1]  # y
        events[i, :n_hits, 2] = s_pos[mask, 2]  # z
        events[i, :n_hits, 3] = s_ene[mask]  # energy

    # --- p_global: unit momentum vector ---
    p_norm = p_mom / np.linalg.norm(p_mom, axis=1, keepdims=True)
    p_global = p_norm.astype(np.float32)

    # --- input_p_global: assumed same as p_global ---
    # (if your data has a separate input direction, replace this)
    input_p_global = p_global.copy()

    # --- input_gun_position: primary vertex ---
    input_gun_position = p_vertex.astype(np.float32)

    print("Writing output_cc3.h5... in ", os.path.dirname(input_path))
    os.makedirs(os.path.dirname(input_path), exist_ok=True)
    with h5py.File(input_path + "output_cc3.h5", "w") as out:
        out.create_dataset("energy", data=incident_energies)
        out.create_dataset("events", data=events)
        out.create_dataset("p_global", data=p_global)
        out.create_dataset("input_p_global", data=input_p_global)
        out.create_dataset("input_gun_position", data=input_gun_position)

    print("Done!")
    print(f"  energy:             {incident_energies.shape}")
    print(f"  events:             {events.shape}")
    print(f"  p_global:           {p_global.shape}")
    print(f"  input_p_global:     {input_p_global.shape}")
    print(f"  input_gun_position: {input_gun_position.shape}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_to_cc3_format.py input.h5 output.h5")
        sys.exit(1)
    convert(sys.argv[1])
