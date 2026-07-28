"""
Produce a step2point-format h5 (same layout as compressed_<algo>.h5: primary/,
steps/) directly from a raw edm4hep.root file, with per-step positions rotated
into the local frame used by convert_to_cc3_format.py - and nothing else.

This deliberately skips everything convert_to_cc3_format.py's apply_transformations()
does beyond that one rotation: no backscatter-radius cut, no 500x500mm box
selection, no per-event alignment shift, no energy cut, no digitize_and_fuzz,
no layer sorting. Every raw MC step from the file is kept, unmerged (same
guarantee as the "identity" algorithm), just rotated.

Rotation (verified against convert_to_cc3_format.py's global_to_local_points,
which computes rotate_z(rotate_x(points, 90), 90) == (X, Y, Z) -> (Z, X, Y)):
    x_local = raw_global_Z
    y_local = raw_global_X
    z_local = raw_global_Y   (unchanged meaning: still the depth/layer axis)

Usage:
    python -m martina_test.edm4hep_to_local_h5 \\
        /eos/project/f/fast/edm4hep_frombenchmark/sim-E1261AT600AP180-180_file_0.edm4hep.root \\
        --shower-limit 200 --output outputs/local_frame/file_0/raw_local.h5
"""

from __future__ import annotations

import argparse
import os
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step2point.core.shower import Shower
from step2point.io import EDM4hepRootReader, write_step2point_hdf5

DEFAULT_COLLECTIONS = ("EcalBarrelCollection",)


def to_local_frame(shower: Shower) -> Shower:
    x_local = shower.z.copy()
    y_local = shower.x.copy()
    z_local = shower.y.copy()
    return Shower(
        shower_id=shower.shower_id,
        x=x_local,
        y=y_local,
        z=z_local,
        E=shower.E,
        t=shower.t,
        cell_id=shower.cell_id,
        pdg=shower.pdg,
        track_id=shower.track_id,
        primary=shower.primary,
        metadata={**shower.metadata, "frame": "local (rotated only, no shift/cut/merge)"},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to the raw edm4hep.root file")
    parser.add_argument("--collections", nargs="+", default=list(DEFAULT_COLLECTIONS))
    parser.add_argument("--shower-limit", type=int, default=None)
    parser.add_argument("--output", required=True, help="Output h5 path")
    args = parser.parse_args()

    reader = EDM4hepRootReader(args.input, collections=tuple(args.collections), shower_limit=args.shower_limit)
    local_showers = []
    n_empty = 0
    for shower in reader.iter_showers():
        if shower.n_points == 0:
            n_empty += 1
            continue
        local_showers.append(to_local_frame(shower))

    print(f"Read {len(local_showers)} showers ({n_empty} empty skipped)")
    write_step2point_hdf5(local_showers, args.output, algorithm="local_frame_raw", source_input=args.input)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
