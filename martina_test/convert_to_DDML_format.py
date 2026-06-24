"""
Converts an already-built CC3-format shower file (the 'input_cc3_file_N.h5'
produced by convert_to_cc3_format.py's convert()) into DDML format: the
per-layer hit counts are prepended to the 'events' array, so that a consumer
can read the first n_layers (30 for ILD) entries per shower to know how many
real hits follow for each layer (for resizing the variable-length vector).
Each prepended "point" repeats its count across all 4 columns (x, y,
layer_id, energy), so it doesn't matter which column a consumer reads it
from.

The input file's 'events' column 2 (layer id) still carries the
digitize_and_fuzz fuzz (layer_idx + random offset in [0, 1)) needed for CC3's
own training; DDML expects the plain calorimeter layer number, so it is
de-fuzzed back to a clean int before computing layer counts.

Usage:
    python -m martina_test.convert_to_DDML_format <input_cc3_file.h5> [out_file]

Example:
    python -m martina_test.convert_to_DDML_format \\
        outputs/cc3input_merge_within_cell/input_cc3_file_0.h5
"""

import os
import sys

import h5py
import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from martina_test.metadata import Metadata


def convert_to_ddml_format(input_cc3_path: str, out_file: str = None):
    print(f"Reading {input_cc3_path}...")
    with h5py.File(input_cc3_path, "r") as f:
        point_clouds = f["events"][:].astype(np.float32)
        energies = f["energy"][:].astype(np.float32)
        passthrough = {key: f[key][:] for key in f.keys() if key not in ("events", "energy", "n_points")}

    metadata = Metadata()
    n_layers = len(metadata.layer_bottom_pos_global)

    # digitize_and_fuzz leaves column 2 as layer_idx + fuzz (a float in [0, n_layers)),
    # for CC3's own training use. DDML expects the plain calorimeter layer number, so
    # de-fuzz it back to a clean int before anything downstream sees it.
    valid = point_clouds[..., 3] > 0
    point_clouds[..., 2] = np.where(valid, np.floor(point_clouds[..., 2]), point_clouds[..., 2])

    # CC3's energy column is in GeV. DDML's CaloCloudsTwoAngleModel::convertOutput
    # passes this value straight into G4FastHit with no unit conversion, and Geant4's
    # internal energy unit is MeV, so it must already be in MeV by the time it gets here.
    point_clouds[..., 3] *= 1000.0  # GeV -> MeV

    # --- per-layer hit counts ---
    layer_ids = point_clouds[..., 2].astype(int)
    layer_counts = np.zeros((point_clouds.shape[0], n_layers), dtype=np.float32)
    for i in range(point_clouds.shape[0]):
        ids = layer_ids[i][valid[i]]
        ids = ids[(ids >= 0) & (ids < n_layers)]
        layer_counts[i] = np.bincount(ids, minlength=n_layers)[:n_layers]

    # --- prepend the counts as the first n_layers "points" of events ---
    n_shown, max_hits, n_feat = point_clouds.shape
    events_with_counts = np.zeros((n_shown, n_layers + max_hits, n_feat), dtype=np.float32)
    events_with_counts[:, :n_layers, :] = layer_counts[:, :, None]
    events_with_counts[:, n_layers:, :] = point_clouds

    num_points = valid.sum(axis=1)

    if out_file is None:
        in_dir, in_name = os.path.split(input_cc3_path)
        base = in_name[:-3] if in_name.endswith(".h5") else in_name
        # parent dir is named "cc3input_<algo>" (see convert_to_cc3_format.py); pull
        # the algo name out of it so the DDML file is identifiable without the path.
        parent_name = os.path.basename(in_dir)
        algo = parent_name.split("cc3input_", 1)[-1] if "cc3input_" in parent_name else parent_name
        out_file = os.path.join(in_dir, f"{base}_ddml_{algo}.h5")

    print(f"Writing {out_file}...")
    with h5py.File(out_file, "w") as hf:
        hf.create_dataset("energy", data=energies)
        hf.create_dataset("events", data=events_with_counts)
        hf.create_dataset("n_points", data=num_points)
        hf.create_dataset("layer_counts", data=layer_counts)  # also kept standalone for easy checking
        for key, value in passthrough.items():
            hf.create_dataset(key, data=value)

    print("Done.")
    print(f"events shape: {events_with_counts.shape}")
    print(f"  events[:, :{n_layers}, 0]  -> per-layer hit counts")
    print(f"  events[:, {n_layers}:, :]  -> actual hits (x, y, layer_id, energy)")
    print("layer_counts[0]:", layer_counts[0])
    print("sum(layer_counts[0]) vs n_points[0]:", layer_counts[0].sum(), num_points[0])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    input_cc3_path = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else None
    convert_to_ddml_format(input_cc3_path, out_file=out_file)
