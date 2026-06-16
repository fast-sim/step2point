"""
Convert a CC3 input HDF5 file back to step2point format.

This is the reverse of convert_to_cc3_format.py.

CC3 input layout  (events[:, :, 4]):
  col 0  =  local_x  =  global_z_aligned  (mm, lateral)
  col 1  =  local_y  =  global_x_aligned  (mm, lateral)
  col 2  =  layer_idx + fuzz              (float in [0, n_layers))
  col 3  =  hit energy                    (GeV)

Reverse steps applied here:
  1.  col 2  →  global y (radial, mm) via layer_bottom_pos_global[floor(col2)] + cell_thickness/2
  2.  col 0/1 + alignment shifts  →  global z and global x
  3.  The layer is determined first from col 2 (floor gives layer index).
      Within that layer, each hit is assigned the cell_id of the nearest real
      detector cell by 2-D Euclidean distance in (x_aligned, z_aligned) using
      a per-layer KD-tree built from one reference shower event.  Hits snapped
      to the same cell get the same cell_id, so step2point merge_within_cell
      collapses them into one voxel with summed energy.
  4.  Flatten per-event arrays into the step2point flat (M,) format.

step2point output layout:
  primary/event_id : (N,)    int32   — event indices 0 … N-1
  primary/momentum : (N, 3)  float32 — energy * p_norm_global  [GeV]
  primary/vertex   : (N, 3)  float32 — gun_xyz_pos_global broadcast to all events
  steps/event_id   : (M,)    int32   — per-hit event index
  steps/position   : (M, 3)  float32 — global (x, y, z) in mm  (snapped to ref cell centre)
  steps/energy     : (M,)    float32 — hit energy in GeV
  steps/cell_id    : (M,)    uint64  — real DD4hep cell_id from nearest reference cell

Usage:
    python convert_from_cc3_format.py input_cc3.h5 [output.h5]
        [--reference-dir /path/to/pipeline2_merge_within_cell]
        [--ref-event N]
"""

import argparse
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np
from metadata import Metadata
from scipy.spatial import cKDTree
from tqdm import tqdm

_DEFAULT_REF_DIR = "/eos/user/m/mamozzan/step2point/outputs/pipeline2_merge_within_cell"


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────


def degrees_to_radians(degrees):
    return degrees * math.pi / 180


def compute_alignment_shifts(phi_global_deg, theta_global_deg, metadata):
    """
    Re-derive the per-layer alignment shifts that were subtracted during
    forward conversion.  Returns x_shift (N, L) and z_shift (N, L).
    """
    dist_to_layers = metadata.layer_bottom_pos_global - metadata.gun_xyz_pos_global[1] + metadata.cell_thickness_global / 2
    phi_r = degrees_to_radians(np.array(phi_global_deg).reshape(-1, 1))
    theta_r = degrees_to_radians(np.array(theta_global_deg).reshape(-1, 1))
    r = (dist_to_layers / np.sin(phi_r)) / np.sin(theta_r)
    x_shift = r * np.cos(phi_r) * np.sin(theta_r) - metadata.gun_xyz_pos_global[0]
    z_shift = r * np.cos(theta_r) + metadata.gun_xyz_pos_global[2]
    return x_shift.astype(np.float32), z_shift.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Reference cell lookup
# ─────────────────────────────────────────────────────────────────────────────


def build_reference_tree(
    ref_dir: str,
    layer_centers: np.ndarray,
    layer_bottom: np.ndarray,
    _metadata=None,
    ref_event: int = 0,  # kept for API compat, unused
):
    """
    Load ALL reference events and build one 2-D KD-tree per calorimeter layer
    in GLOBAL (x, z) space.

    Using a single reference event's cells (~45/layer) caused out-of-footprint
    generated hits to pile onto edge cells.  Loading all events gives the full
    detector cell map (~12 000 cells/layer), so every generated hit finds a
    genuinely nearby real cell regardless of shower angle or width.

    Layer assignment of each generated hit is done first (from CC3 col 2),
    then the tree is queried with the hit's computed global (x, z).

    Returns
    -------
    per_layer_trees : list[cKDTree | None]  — one 2-D tree per layer in global (x, z)
    per_layer_pos   : list[ndarray | None]  — (K_l, 3) global (x, y, z) of each ref cell
    per_layer_cids  : list[ndarray | None]  — (K_l,) DD4hep cell_ids
    """
    n_layers = len(layer_centers)

    ref_paths = sorted(Path(ref_dir).glob("file_*/compressed_merge_within_cell.h5"))
    if not ref_paths:
        print(f"[ref] No files found under {ref_dir}; will use Cartesian fallback.")
        return None, None, None

    # Load the first reference file — it contains ~35k events spanning the full
    # phi/theta range, giving ~726k unique cells (full detector coverage).
    # Using all files would load 260M+ hits unnecessarily.
    path = ref_paths[0]
    print(f"[ref] Loading all events from {path} …")
    try:
        with h5py.File(str(path), "r", locking=False) as f:
            pos = np.asarray(f["steps/position"], dtype=np.float32)
            cids = np.asarray(f["steps/cell_id"], dtype=np.uint64)
    except OSError as e:
        print(f"[ref] Could not read {path}: {e}")
        return None, None, None

    print(f"[ref] Loaded {len(cids):,} hits.")

    # Filter to ECAL radial range
    radial = pos[:, 1]
    r_min = float(layer_centers[0]) - 10.0
    r_max = float(layer_centers[-1]) + 10.0
    in_ecal = (radial >= r_min) & (radial <= r_max)
    pos = pos[in_ecal]
    cids = cids[in_ecal]

    # Deduplicate: one representative global position per unique cell_id
    _, idx = np.unique(cids, return_index=True)
    pos = pos[idx]
    cids = cids[idx]

    # Assign each unique cell to its layer
    layer_idx_ref = np.clip(
        np.searchsorted(layer_bottom, pos[:, 1], side="right") - 1,
        0,
        n_layers - 1,
    )

    # Build one 2-D (global x, global z) KD-tree per layer
    per_layer_trees: list = []
    per_layer_pos: list = []
    per_layer_cids: list = []
    for li in range(n_layers):
        lmask = layer_idx_ref == li
        if not lmask.any():
            per_layer_trees.append(None)
            per_layer_pos.append(None)
            per_layer_cids.append(None)
        else:
            xz = np.column_stack([pos[lmask, 0], pos[lmask, 2]]).astype(np.float32)
            per_layer_trees.append(cKDTree(xz))
            per_layer_pos.append(pos[lmask])
            per_layer_cids.append(cids[lmask])

    n_populated = sum(t is not None for t in per_layer_trees)
    print(f"[ref] {len(cids):,} unique cells across {n_populated}/{n_layers} layers (mean {len(cids) // n_populated}/layer).")
    return per_layer_trees, per_layer_pos, per_layer_cids


# ─────────────────────────────────────────────────────────────────────────────
# Main conversion
# ─────────────────────────────────────────────────────────────────────────────


def convert(
    input_path: str,
    output_path: str | None = None,
    ref_dir: str | None = _DEFAULT_REF_DIR,
    ref_event: int = 0,
):
    if output_path is None:
        output_path = input_path.replace(".h5", ".step2point.h5")

    print(f"Input  : {input_path}")
    print(f"Output : {output_path}")

    # ------------------------------------------------------------------ #
    # 1. Read CC3 input file                                               #
    # ------------------------------------------------------------------ #
    with h5py.File(input_path, "r") as f:
        events = f["events"][:]  # (N, max_hits, 4)
        energy_inc = f["energy"][:].reshape(-1)  # (N,)  [GeV]
        p_norm_global = np.array(f["p_norm_global"][:], dtype=np.float32)  # (N, 3)
        phi_global = np.array(f["phi_global"][:], dtype=np.float32)  # (N,)
        theta_global = np.array(f["theta_global"][:], dtype=np.float32)  # (N,)

    n_events = events.shape[0]
    print(f"Events : {n_events}")

    # ------------------------------------------------------------------ #
    # 2. Load metadata                                                     #
    # ------------------------------------------------------------------ #
    metadata = Metadata()
    layer_bottom = metadata.layer_bottom_pos_global  # (n_layers,)
    cell_thick = metadata.cell_thickness_global
    n_layers = len(layer_bottom)
    layer_center = (layer_bottom + cell_thick / 2).astype(np.float32)  # (n_layers,)

    # ------------------------------------------------------------------ #
    # 3. Build reference cell KD-tree (3D Euclidean)                       #
    # ------------------------------------------------------------------ #
    cell_size = 2.0 * metadata.half_cell_size_global  # ≈ 5.09 mm (Cartesian fallback)
    _CELL_OFFSET = 200
    _CELL_N = 500

    per_layer_trees = per_layer_pos = per_layer_cids = None
    if ref_dir and Path(ref_dir).exists():
        per_layer_trees, per_layer_pos, per_layer_cids = build_reference_tree(
            ref_dir, layer_center, layer_bottom, metadata, ref_event=ref_event
        )

    use_ref = per_layer_trees is not None and any(t is not None for t in per_layer_trees)
    if not use_ref:
        print("[cell_id] Using Cartesian grid fallback (no reference data).")

    # ------------------------------------------------------------------ #
    # 4. Pre-compute alignment shifts                                      #
    # ------------------------------------------------------------------ #
    if metadata.aligne:
        print("Computing alignment shifts …")
        x_shifts, z_shifts = compute_alignment_shifts(phi_global, theta_global, metadata)
    else:
        x_shifts = np.zeros((n_events, n_layers), dtype=np.float32)
        z_shifts = np.zeros((n_events, n_layers), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 5. Reconstruct global (x, y, z) and assign cell_ids per event       #
    # ------------------------------------------------------------------ #
    all_positions = []
    all_energies = []
    all_event_ids = []
    all_cell_ids = []

    print("Reconstructing global coordinates …")
    for i in tqdm(range(n_events)):
        ev = events[i]  # (max_hits, 4)
        mask = ev[:, 3] > 0  # valid (non-padded) hits
        if mask.sum() == 0:
            continue

        hits = ev[mask]  # (n_hits, 4)
        z_fuzz = hits[:, 2]
        layer_idx = np.clip(np.floor(z_fuzz).astype(int), 0, n_layers - 1)

        global_y = layer_center[layer_idx].copy()  # (n_hits,) radial depth
        global_x = hits[:, 1] + x_shifts[i, layer_idx]  # undo x alignment
        global_z = hits[:, 0] + z_shifts[i, layer_idx]  # undo z alignment

        if use_ref:
            # ── Per-layer nearest reference cell (global x, z) ──────────
            # Layer is known from CC3 col 2.  Within each layer, snap to
            # the nearest real detector cell using the hit's global (x, z)
            # — computed from local aligned coords + this event's shifts.
            # The reference tree covers the full detector (all events pooled),
            # so every hit finds a genuinely nearby cell regardless of angle.
            cid_ev = np.zeros(len(hits), dtype=np.uint64)
            snap_x = global_x.copy()
            snap_y = global_y.copy()
            snap_z = global_z.copy()

            for li in range(n_layers):
                lmask = layer_idx == li
                if not lmask.any():
                    continue
                tree_l = per_layer_trees[li]
                if tree_l is None:
                    continue
                xz_q = np.column_stack([global_x[lmask], global_z[lmask]])
                _, nn = tree_l.query(xz_q)
                cid_ev[lmask] = per_layer_cids[li][nn]
                snap_x[lmask] = per_layer_pos[li][nn, 0]
                snap_y[lmask] = per_layer_pos[li][nn, 1]
                snap_z[lmask] = per_layer_pos[li][nn, 2]

            global_x = snap_x
            global_y = snap_y
            global_z = snap_z
            all_cell_ids.append(cid_ev)

        else:
            # ── Cartesian fallback ─────────────────────────────────────
            x_bin = np.round(global_x / cell_size).astype(np.int64)
            z_bin = np.round(global_z / cell_size).astype(np.int64)
            global_x = (x_bin * cell_size).astype(np.float32)
            global_z = (z_bin * cell_size).astype(np.float32)
            cid = (
                layer_idx.astype(np.int64) * _CELL_N * _CELL_N + (x_bin + _CELL_OFFSET) * _CELL_N + (z_bin + _CELL_OFFSET)
            ).astype(np.uint64)
            all_cell_ids.append(cid)

        pos = np.stack([global_x, global_y, global_z], axis=1).astype(np.float32)
        all_positions.append(pos)
        all_energies.append(hits[:, 3].astype(np.float32))
        all_event_ids.append(np.full(mask.sum(), i, dtype=np.int32))

    # ------------------------------------------------------------------ #
    # 6. Write step2point HDF5                                            #
    # ------------------------------------------------------------------ #
    positions = np.concatenate(all_positions, axis=0)  # (M, 3)
    energies = np.concatenate(all_energies, axis=0)  # (M,)
    event_ids = np.concatenate(all_event_ids, axis=0)  # (M,)
    cell_ids = np.concatenate(all_cell_ids, axis=0)  # (M,)
    n_hits_total = len(energies)

    p_mom = (energy_inc[:, None] * p_norm_global).astype(np.float32)
    vertex = np.tile(metadata.gun_xyz_pos_global.astype(np.float32), (n_events, 1))
    prim_ids = np.arange(n_events, dtype=np.int32)
    prim_pdg = np.full(n_events, 22, dtype=np.int32)  # photon

    print(f"Total hits : {n_hits_total}  ({n_hits_total / n_events:.0f} avg/event)")

    # Quick sanity: unique cell_ids per event (first 5)
    for _i in range(min(5, n_events)):
        mask_ev = event_ids == _i
        n_unique = len(np.unique(cell_ids[mask_ev]))
        n_total_ev = mask_ev.sum()
        print(
            f"  event {_i:4d}: {n_total_ev} hits → {n_unique} unique cell_ids"
            f"  (reduction {(1 - n_unique / max(n_total_ev, 1)) * 100:.1f}%)"
        )

    print(f"Writing to {output_path} …")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with h5py.File(output_path, "w", locking=False) as out:
        g_prim = out.create_group("primary")
        g_prim.create_dataset("event_id", data=prim_ids)
        g_prim.create_dataset("pdg", data=prim_pdg)
        g_prim.create_dataset("momentum", data=p_mom)
        g_prim.create_dataset("vertex", data=vertex)

        g_steps = out.create_group("steps")
        g_steps.create_dataset("event_id", data=event_ids)
        g_steps.create_dataset("cell_id", data=cell_ids)
        g_steps.create_dataset(
            "position",
            data=positions,
            chunks=(min(65536, n_hits_total), 3),
        )
        g_steps.create_dataset(
            "energy",
            data=energies,
            chunks=(min(65536, n_hits_total),),
        )

    print("Done.")
    print(f"  primary/event_id : {prim_ids.shape}")
    print(f"  steps/position   : {positions.shape}")
    print(f"  steps/energy     : {energies.shape}")
    print(f"  steps/cell_id    : {cell_ids.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="CC3 HDF5 input file")
    p.add_argument("output", nargs="?", default=None, help="Output step2point HDF5 (default: input with .step2point.h5 suffix)")
    p.add_argument(
        "--reference-dir",
        default=_DEFAULT_REF_DIR,
        help="Directory containing file_*/compressed_merge_within_cell.h5 reference files. "
        "Pass an empty string to disable and use Cartesian fallback.",
    )
    p.add_argument(
        "--ref-event",
        type=int,
        default=0,
        help="Which event index to load from the first reference file (default: 0). "
        "This single event's cells (~1258) form the lookup table; all generated "
        "hits snap to one of these cells so the merged count matches the reference.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t0 = time.time()
    convert(
        args.input,
        args.output,
        ref_dir=args.reference_dir or None,
        ref_event=args.ref_event,
    )
    print(f"--- {time.time() - t0:.1f} s ---")
