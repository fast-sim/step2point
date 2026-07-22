"""
Produce an h5 in the same on-disk format as a DDML ML_FILE (e.g.
DDML/models/input_cc3_file_0_ddml_identity.h5: energy, events, layer_counts,
n_points, p_norm_global/local, theta_global/local, phi_global/local) directly
from a raw edm4hep.root file, applying the global->local coordinate rotation
(convert_to_cc3_format.py's global_to_local_points), the layer binning needed
to populate the layer_id column (split_to_layers(), using the real
metadata.layer_bottom_pos_global/cell_thickness_global - reused, not
re-derived), and the per-shower alignment shift (get_alignment_shifts(),
reused from Transform_pointcloud) - without this shift, the box selection
below would be centered on the fixed global origin instead of each shower's
own incident axis, truncating off-axis showers' hits much more aggressively
than the real pipeline does.

Output coordinates are stored *shifted* (not undone), same convention as
convert_to_cc3_format.py's own output - use remove_alignment_shift.py
afterward if you want true global positions for plotting/comparison.

Everything else convert_to_cc3_format.py's apply_transformations() does is
deliberately skipped: no backscatter-radius cut, no energy cut, no
digitize_and_fuzz, no layer-count sorting. Every raw MC step is kept, unmerged.

Pass --no-shift-and-cut to also skip the per-shower alignment shift and the
local-frame box cut, writing out raw (unshifted, unselected) hit positions.

Usage:
    python -m martina_test.edm4hep_to_ddml_minimal \\
        /eos/project/f/fast/edm4hep_frombenchmark/sim-E1261AT600AP180-180_file_0.edm4hep.root \\
        --shower-limit 200 --output outputs/local_frame/file_0/ddml_minimal.h5
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from martina_test.convert_to_cc3_format import Transform_pointcloud, split_to_layers
from martina_test.metadata import Metadata
from step2point.core.shower import Shower
from step2point.geometry.dd4hep.bitfield import decode_dd4hep_cell_id
from step2point.geometry.dd4hep.factory_geometry import barrel_cell_center, build_barrel_layout_from_collection
from step2point.io import EDM4hepRootReader

DEFAULT_COLLECTIONS = ("EcalBarrelCollection",)

# EDM4hep MCParticle.simulatorStatus bit 0: set if the particle was created
# during simulation (a secondary); unset for an original generator/gun
# particle - same bit podio's MCParticle.isCreatedInSimulation() checks.
_CREATED_IN_SIMULATION_BIT = 1


def iter_showers_uproot(input_path, collections, shower_limit=None):
    """Yield Shower objects straight from each calorimeter collection's own
    SimCalorimeterHit fields (``<col>.position.x/y/z``, ``.energy``,
    ``.cellID``), read directly via uproot - no contribution/step loop, no
    merging computed by us. One row per struck cell per event, exactly as
    Geant4's own sensitive detector already built it: energy summed over
    that cell's raw MC-step contributions, position placed by the
    sensitive detector (not the raw steps' energy centroid).

    Verified (2026-07-16): this ``.energy`` equals the sum of the same hit's
    raw contribution energies to float precision, but ``.position`` is NOT
    the energy-weighted centroid of those contributions - it's a fixed
    function of ``cellID`` alone (0 mismatches checked across 5,642 repeat
    cellIDs across events), i.e. the segmentation's geometric cell center.
    So this path is directly comparable to, but not expected to numerically
    match, this script's own ``--algorithm merge_within_cell
    --position-mode weighted`` computed from raw contributions
    (--position-mode center is the one that should agree).
    """
    import awkward as ak
    import uproot

    f = uproot.open(input_path)
    tree = f["events"]
    n_events = tree.num_entries
    if shower_limit is not None:
        n_events = min(n_events, shower_limit)

    sim_status = tree["MCParticle/MCParticle.simulatorStatus"].array(entry_stop=n_events)
    mom_x = tree["MCParticle/MCParticle.momentum.x"].array(entry_stop=n_events)
    mom_y = tree["MCParticle/MCParticle.momentum.y"].array(entry_stop=n_events)
    mom_z = tree["MCParticle/MCParticle.momentum.z"].array(entry_stop=n_events)
    is_primary = (sim_status & _CREATED_IN_SIMULATION_BIT) == 0

    per_collection = []
    for col in collections:
        base = f"{col}/{col}."
        per_collection.append(
            {
                "x": tree[base + "position.x"].array(entry_stop=n_events),
                "y": tree[base + "position.y"].array(entry_stop=n_events),
                "z": tree[base + "position.z"].array(entry_stop=n_events),
                "E": tree[base + "energy"].array(entry_stop=n_events),
                "cell_id": tree[base + "cellID"].array(entry_stop=n_events),
            }
        )

    for iev in range(n_events):
        primary_idx = ak.local_index(is_primary[iev])[is_primary[iev]]
        primary = {}
        if len(primary_idx) > 0:
            p0 = int(primary_idx[0])
            primary = {"momentum": (float(mom_x[iev][p0]), float(mom_y[iev][p0]), float(mom_z[iev][p0]))}

        x = np.concatenate([ak.to_numpy(c["x"][iev]) for c in per_collection]) if per_collection else np.array([])
        y = np.concatenate([ak.to_numpy(c["y"][iev]) for c in per_collection]) if per_collection else np.array([])
        z = np.concatenate([ak.to_numpy(c["z"][iev]) for c in per_collection]) if per_collection else np.array([])
        E = np.concatenate([ak.to_numpy(c["E"][iev]) for c in per_collection]) if per_collection else np.array([])
        cell_id = (
            np.concatenate([ak.to_numpy(c["cell_id"][iev]) for c in per_collection])
            if per_collection
            else np.array([], dtype=np.uint64)
        )

        yield Shower(shower_id=iev, x=x, y=y, z=z, E=E, cell_id=cell_id, primary=primary)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to the raw edm4hep.root file")
    parser.add_argument("--collections", nargs="+", default=list(DEFAULT_COLLECTIONS))
    parser.add_argument("--shower-limit", type=int, default=None)
    parser.add_argument("--output", required=True, help="Output h5 path")
    parser.add_argument(
        "--algorithm",
        choices=["identity", "merge_within_cell"],
        default="identity",
        help="merge_within_cell groups by cell_id AFTER the coordinate transform, "
        "alignment shift, and box cut (not before, unlike step2point's own "
        "MergeWithinCell) - i.e. it merges whatever raw steps survive the cut, "
        "grouped by their true detector cell_id.",
    )
    parser.add_argument(
        "--position-mode",
        choices=["weighted", "center"],
        default="weighted",
        help="merge_within_cell only: 'weighted' places the merged point at the "
        "energy-weighted barycenter of its raw steps (matches step2point's real "
        "MergeWithinCell). 'center' places it at the cell's true geometric center "
        "instead, decoded from cell_id via the real dd4hep segmentation (requires "
        "--compact-xml/--collection-name for whatever geometry assigned this "
        "input file's cell_ids).",
    )
    parser.add_argument("--compact-xml", help="Compact detector XML, required for --position-mode center")
    parser.add_argument("--collection-name", help="Readout collection name, required for --position-mode center")
    parser.add_argument(
        "--no-shift-and-cut",
        action="store_true",
        help="Skip the per-shower alignment shift (get_alignment_shifts()) and "
        "the local-frame box cut. Hits are kept in raw (unshifted) global "
        "position and no spatial selection is applied - every raw MC step "
        "that survives layer binning is written out.",
    )
    parser.add_argument(
        "--uproot-reader",
        action="store_true",
        help="Read each collection's own SimCalorimeterHit fields "
        "(position.x/y/z, energy, cellID) directly from the edm4hep.root "
        "branch keys via uproot, instead of iterating raw MC-step "
        "contributions through podio (EDM4hepRootReader). One row per "
        "struck cell per event, already merged by Geant4's own sensitive "
        "detector - no podio dependency, and --algorithm is moot on this "
        "path (cell_ids are already unique, so merge_within_cell is a "
        "no-op grouping).",
    )
    args = parser.parse_args()
    if args.position_mode == "center" and not (args.compact_xml and args.collection_name):
        parser.error("--position-mode center requires --compact-xml and --collection-name")

    metadata = Metadata()
    transform = Transform_pointcloud(metadata)
    n_layers = len(metadata.layer_bottom_pos_global)
    print(f"n_layers = {n_layers}")

    layout = None
    if args.position_mode == "center":
        layout = build_barrel_layout_from_collection(args.compact_xml, args.collection_name)
        if layout.segmentation_type != "CartesianGridXY":
            raise NotImplementedError("--position-mode center currently supports only barrel CartesianGridXY layouts.")

    if args.uproot_reader:
        shower_iter = iter_showers_uproot(args.input, collections=tuple(args.collections), shower_limit=args.shower_limit)
    else:
        reader = EDM4hepRootReader(args.input, collections=tuple(args.collections), shower_limit=args.shower_limit)
        shower_iter = reader.iter_showers()

    per_shower_hits = []  # list of (n_i, 4) arrays: (x_local, y_local, layer_id, energy_MeV)
    per_shower_layer_counts = []
    p_mom = []
    n_empty = 0

    for shower in shower_iter:
        if shower.n_points == 0:
            n_empty += 1
            continue

        # this shower's own incident direction, needed for get_alignment_shifts()
        # below - same formula as the batch phi/theta computation at the end of
        # this function, just done per-shower since the shift must be applied
        # before this shower's own box cut, not after.
        mom = shower.primary["momentum"]
        p_norm = mom / np.linalg.norm(mom)
        phi_i = np.degrees(np.arctan2(p_norm[1], p_norm[0]))
        theta_i = np.degrees(np.arccos(p_norm[2]))
        if args.no_shift_and_cut:
            x_shift = np.zeros((1, n_layers), dtype=np.float64)
            z_shift = np.zeros((1, n_layers), dtype=np.float64)
        else:
            x_shift, z_shift = transform.get_alignment_shifts(
                np.array([phi_i], dtype=np.float64), np.array([theta_i], dtype=np.float64)
            )  # each (1, n_layers)

        # layer binning on raw global Y (unaffected by the X/Z shift below) -
        # computed first, same order as apply_transformations(), since the
        # shift is looked up per-hit by its layer_id.
        layer_ids_raw, _, _ = split_to_layers(
            np.stack([shower.x, shower.y, shower.z, shower.E], axis=1),
            metadata.layer_bottom_pos_global,
            metadata.cell_thickness_global,
            layer_axis=1,
        )
        clipped = np.clip(layer_ids_raw, 0, n_layers - 1)
        x_shifted = shower.x - x_shift[0, clipped]
        z_shifted = shower.z - z_shift[0, clipped]

        # shifted global (x, y, z) -> local frame, same rotation as
        # convert_to_cc3_format.Transform_pointcloud.global_to_local_points:
        # (X, Y, Z) -> (Z, X, Y), so x_local=raw Z, y_local=raw X, z_local=raw Y
        # (the depth/layer axis).
        raw = np.stack([x_shifted, shower.y, z_shifted], axis=1).astype(np.float64)
        local = transform.global_to_local_points(raw.T).T
        x_local, y_local = local[:, 0], local[:, 1]

        # Box selection in local x/y, now on shifted (per-shower-axis-centered)
        # coordinates like apply_transformations()'s own box_selection() call -
        # the one other cut kept, since without it rare far-outlier steps
        # (backscatter, stray secondaries) blow out the spatial range and
        # degrade binning/resolution for everything. Edge set to
        # 249.32670000000002 mm (matching apply_transformations()'s cell_edge)
        # instead of metadata's default 250mm, so the cut lands exactly on a
        # readout cell boundary instead of mid-cell.
        if args.no_shift_and_cut:
            in_box = np.ones(x_local.shape[0], dtype=bool)
        else:
            cell_edge = 249.32670000000002
            in_box = (x_local > -cell_edge) & (x_local < cell_edge) & (y_local > -cell_edge) & (y_local < cell_edge)
        if not np.any(in_box):
            n_empty += 1
            continue

        x_local, y_local = x_local[in_box], y_local[in_box]
        E_boxed = shower.E[in_box]
        layer_ids = layer_ids_raw[in_box]

        if args.algorithm == "merge_within_cell":
            # Merge raw steps that survived the box cut, grouped by their true
            # detector cell_id - same energy-weighted-centroid formula as
            # step2point.algorithms.merge_within_cell.MergeWithinCell._compress_python(),
            # but applied AFTER the transform/shift/box-cut instead of before
            # (on the original, unrestricted raw steps) like the real pipeline
            # does. All steps sharing one cell_id are physically in the same
            # layer, so layer_ids[0] per group is exact, not an approximation.
            cell_id_boxed = shower.cell_id[in_box]
            unique_cells, first_indices, inverse = np.unique(cell_id_boxed, return_index=True, return_inverse=True)
            n_cells = len(unique_cells)
            e_sum = np.bincount(inverse, weights=E_boxed, minlength=n_cells)
            safe_e = np.where(e_sum > 0.0, e_sum, 1.0)

            if args.position_mode == "weighted":
                x_local = np.bincount(inverse, weights=x_local * E_boxed, minlength=n_cells) / safe_e
                y_local = np.bincount(inverse, weights=y_local * E_boxed, minlength=n_cells) / safe_e
            else:
                # True geometric cell center (not the raw steps' centroid), decoded
                # from cell_id via the real dd4hep segmentation - same lookup
                # step2point.algorithms.merge_within_regular_subcell.MergeWithinRegularSubcell
                # uses for its own "center" position_mode. Returned in GLOBAL
                # coords, so it needs the same per-layer shift + rotation the raw
                # hits already went through above, to land in the same local
                # frame as x_local/y_local. The shift is looked up via OUR OWN
                # layer binning (split_to_layers on this center's raw Y), not
                # decode_dd4hep_cell_id's "layer" field, since the two "layer"
                # concepts (dd4hep readout layer vs metadata.layer_bottom_pos_global
                # index) are not guaranteed to use the same numbering.
                centers_global = np.empty((n_cells, 3), dtype=np.float64)
                for g, cid in enumerate(unique_cells):
                    decoded = decode_dd4hep_cell_id(int(cid), layout.cell_id_encoding)
                    centers_global[g] = barrel_cell_center(
                        layout, decoded["layer"], decoded["module"], decoded["x"], decoded["y"]
                    )
                center_layer_ids, _, _ = split_to_layers(
                    np.concatenate([centers_global, np.zeros((n_cells, 1))], axis=1),
                    metadata.layer_bottom_pos_global,
                    metadata.cell_thickness_global,
                    layer_axis=1,
                )
                center_clipped = np.clip(center_layer_ids, 0, n_layers - 1)
                cx_shifted = centers_global[:, 0] - x_shift[0, center_clipped]
                cz_shifted = centers_global[:, 2] - z_shift[0, center_clipped]
                centers_local = transform.global_to_local_points(
                    np.stack([cx_shifted, centers_global[:, 1], cz_shifted], axis=1).T
                ).T
                x_local, y_local = centers_local[:, 0], centers_local[:, 1]

            layer_ids = layer_ids[first_indices]
            E_boxed = e_sum

        # layer_id must be a valid array index (0..n_layers-1) for the fixed-layer-count
        # events format - not a physics/quality cut, a structural requirement. Points
        # below the first layer floor or beyond the last one get layer_ids of -1 or
        # n_layers from split_to_layers()'s searchsorted and can't be written at all
        # (the real pipeline never sees them, since its backscatter-radius cut removes
        # anything before the first layer before split_to_layers() is even called).
        valid = (layer_ids >= 0) & (layer_ids < n_layers)
        counts = np.bincount(layer_ids[valid], minlength=n_layers)[:n_layers].astype(np.float32)
        per_shower_layer_counts.append(counts)

        energy_mev = E_boxed.astype(np.float64) * 1000.0  # GeV -> MeV, DDML convention
        hits = np.stack([x_local, y_local, layer_ids.astype(np.float64), energy_mev], axis=1)[valid].astype(np.float32)
        per_shower_hits.append(hits)

        p_mom.append(mom)

    print(f"Read {len(per_shower_hits)} showers ({n_empty} empty skipped)")

    n_showers = len(per_shower_hits)
    max_hits = max(h.shape[0] for h in per_shower_hits)
    print(f"max_hits (unfiltered, uncut) = {max_hits}")

    events = np.zeros((n_showers, n_layers + max_hits, 4), dtype=np.float32)
    n_points = np.zeros(n_showers, dtype=np.int64)
    layer_counts = np.zeros((n_showers, n_layers), dtype=np.float32)
    for i, (hits, counts) in enumerate(zip(per_shower_hits, per_shower_layer_counts)):
        events[i, :n_layers, :] = counts[:, None]
        events[i, n_layers : n_layers + hits.shape[0], :] = hits
        n_points[i] = hits.shape[0]
        layer_counts[i] = counts

    p_mom = np.asarray(p_mom, dtype=np.float64)
    energy = np.linalg.norm(p_mom, axis=1).astype(np.float32).reshape(-1, 1)
    p_norm_global = (p_mom / np.linalg.norm(p_mom, axis=1, keepdims=True)).astype(np.float32)
    phi_global = np.degrees(np.arctan2(p_norm_global[:, 1], p_norm_global[:, 0])).astype(np.float32)
    theta_global = np.degrees(np.arccos(p_norm_global[:, 2])).astype(np.float32)
    p_norm_local = transform.global_to_local_points(p_norm_global.T.astype(np.float64)).T.astype(np.float32)
    phi_local = np.degrees(np.arctan2(p_norm_local[:, 1], p_norm_local[:, 0])).astype(np.float64)
    theta_local = np.degrees(np.arccos(p_norm_local[:, 2])).astype(np.float64)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with h5py.File(args.output, "w") as hf:
        hf.create_dataset("energy", data=energy)
        hf.create_dataset("events", data=events)
        hf.create_dataset("layer_counts", data=layer_counts)
        hf.create_dataset("n_points", data=n_points)
        hf.create_dataset("p_norm_global", data=p_norm_global)
        hf.create_dataset("p_norm_local", data=p_norm_local)
        hf.create_dataset("phi_global", data=phi_global)
        hf.create_dataset("phi_local", data=phi_local)
        hf.create_dataset("theta_global", data=theta_global)
        hf.create_dataset("theta_local", data=theta_local)
    print(f"Wrote {args.output}")
    print(f"events shape: {events.shape}")


if __name__ == "__main__":
    main()
