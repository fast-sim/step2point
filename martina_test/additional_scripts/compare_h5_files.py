from pathlib import Path

import h5py
import numpy as np

from step2point.io import EDM4hepRootReader, Step2PointHDF5Reader


def build_reader(input_path: str):
    suffixes = Path(input_path).suffixes
    if suffixes[-1:] == [".root"]:
        return EDM4hepRootReader(input_path)
    if suffixes[-1:] in ([".h5"], [".hdf5"]):
        return Step2PointHDF5Reader(input_path)
    raise ValueError(f"Unsupported input file type for '{input_path}'. Expected .root, .h5, or .hdf5.")


def inspect_h5(path, label):
    print(f"\n{'=' * 50}")
    print(f"FILE: {label}")
    print(f"{'=' * 50}")
    with h5py.File(path, "r") as f:

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
                # print a small sample
                try:
                    data = obj[0]
                    print(f"    sample[0] = {data}")
                except Exception:
                    pass

        f.visititems(visit)


if __name__ == "__main__":
    f1 = h5py.File("../photons_h5/p22_th45-135_ph79-109_en5-130_seed0_ip.h5", "r")
    f2 = h5py.File("outputs/pipeline_identity_0/compressed_identity.h5", "r")

    print("=== GUN PARTICLE COMPARISON ===")
    # PDG
    pdg1 = f1["gun_PDG"][:]
    pdg2 = f2["primary/pdg"][:]
    print(f"PDG match: {np.all(pdg1 == pdg2)}")

    # Momentum
    mom1 = f1["gun_p_global"][:]
    mom2 = f2["primary/momentum"][:]
    diff = np.abs(mom1 - mom2.astype(np.float64))
    print(f"Momentum max diff: {diff.max():.6f}  (float32 vs float64 precision)")

    # Vertex
    pos1 = f1["input_gun_position"][:]
    pos2 = f2["primary/vertex"][:]
    diff = np.abs(pos1 - pos2.astype(np.float64))
    print(f"Vertex max diff:   {diff.max():.6f}")

    print("\n=== HIT-LEVEL COMPARISON (per event) ===")
    event_ids = f2["steps/event_id"][:]
    steps_energy = f2["steps/energy"][:]
    steps_pos = f2["steps/position"][:]

    for ev in range(5):  # check first 5 events
        # File 1: remove padded zeros
        ev1 = f1["events"][ev]  # (47983, 7)
        mask = ev1[:, 3] != 0  # energy != 0
        hits1 = ev1[mask]

        # File 2: group by event_id
        ev_mask = event_ids == ev
        hits2_energy = steps_energy[ev_mask]
        hits2_pos = steps_pos[ev_mask]

        total_e1 = hits1[:, 3].sum()
        total_e2 = hits2_energy.sum()

        print(
            f"Event {ev}: "
            f"n_hits_f1={mask.sum():5d}  n_hits_f2={ev_mask.sum():5d}  "
            f"E_f1={total_e1:.4f}  E_f2={total_e2:.4f}  "
            f"dE={abs(total_e1 - total_e2):.4f}"
        )

    f1.close()
    f2.close()
