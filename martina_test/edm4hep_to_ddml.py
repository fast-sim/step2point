"""
Produce a DDML-format h5 (like DDML/models/input_cc3_file_<seed>_ddml_<algo>.h5)
directly from a raw benchmark edm4hep.root file, in a single script.

This runs the same three stages as submit_job_new.sh / convert_pipeline2_to_ddml.py
(edm4hep.root -> compressed_<algo>.h5 -> input_cc3_file_<seed>.h5 ->
input_cc3_file_<seed>_ddml_<algo>.h5), but does stage 1 in-process via the
step2point library (EDM4hepRootReader + IdentityCompression/MergeWithinCell +
write_step2point_hdf5) instead of shelling out to run_step2point_pipeline.py.

Must be run with the step2point repo root as the working directory (same
requirement as convert_to_cc3_format.py/convert_to_DDML_format.py, whose
output paths are relative to cwd).

Usage:
    python -m martina_test.edm4hep_to_ddml \\
        /eos/project/f/fast/edm4hep_frombenchmark/sim-E1261AT600AP180-180_file_0.edm4hep.root \\
        --algorithm identity --copy-to-ddml-models

    python -m martina_test.edm4hep_to_ddml \\
        /eos/project/f/fast/edm4hep_frombenchmark/sim-E1261AT600AP180-180_file_0.edm4hep.root \\
        --algorithm merge_within_cell
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from martina_test.convert_pipeline2_to_ddml import _cc3_output_path
from martina_test.convert_to_cc3_format import convert as convert_to_cc3
from martina_test.convert_to_DDML_format import convert_to_ddml_format
from step2point.algorithms.identity import IdentityCompression
from step2point.algorithms.merge_within_cell import MergeWithinCell
from step2point.io import EDM4hepRootReader, write_step2point_hdf5

DDML_MODELS_DIR = "/eos/user/m/mamozzan/DDML/models"
# matches DEFAULT_COLLECTIONS["regular"] in DDML/scripts/plot_first_shower.py -
# this benchmark file's barrel ECal hits live in one un-split collection (as
# opposed to the real geometry's four Si/Sc Odd/Even collections).
DEFAULT_COLLECTIONS = ("EcalBarrelCollection",)

ALGORITHMS = {
    "identity": IdentityCompression,
    "merge_within_cell": MergeWithinCell,
}


def _seed_from_filename(input_path: str) -> int:
    m = re.search(r"file_(\d+)\.edm4hep\.root$", os.path.basename(input_path))
    if not m:
        raise ValueError(f"Couldn't parse a file_<seed> from '{input_path}'; pass --seed explicitly.")
    return int(m.group(1))


def edm4hep_to_ddml(
    input_path: str,
    algorithm: str,
    seed: int | None = None,
    collections: tuple[str, ...] = DEFAULT_COLLECTIONS,
    outdir: str | None = None,
    copy_to_ddml_models: bool = False,
    shower_limit: int | None = None,
) -> str:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm}', expected one of {sorted(ALGORITHMS)}")
    seed = _seed_from_filename(input_path) if seed is None else seed
    # compressed_<algo>.h5 must sit under a "file_<seed>" directory - both
    # convert_to_cc3_format.convert() and convert_to_DDML_format.convert_to_ddml_format()
    # parse algo/seed back out of this exact path shape (see their out_file naming logic).
    outdir = outdir or f"outputs/pipeline2_{algorithm}"
    stage_dir = os.path.join(outdir, f"file_{seed}")
    os.makedirs(stage_dir, exist_ok=True)

    print(f"[1/3] Reading {input_path} (collections={list(collections)}) and compressing with '{algorithm}' ...")
    reader = EDM4hepRootReader(input_path, collections=tuple(collections), shower_limit=shower_limit)
    algo = ALGORITHMS[algorithm]()
    compressed_showers = []
    n_points_before = n_points_after = 0
    e_before = e_after = 0.0
    n_empty = 0
    for shower in reader.iter_showers():
        if shower.n_points == 0:
            n_empty += 1
            continue
        result = algo.compress(shower)
        compressed_showers.append(result.shower)
        n_points_before += result.stats["n_points_before"]
        n_points_after += result.stats["n_points_after"]
        e_before += result.stats["energy_before"]
        e_after += result.stats["energy_after"]
    print(
        f"  compressed {len(compressed_showers)} showers ({n_empty} empty events skipped)\n"
        f"  points: {n_points_before} -> {n_points_after}  "
        f"energy: {e_before:.6f} -> {e_after:.6f} GeV (should match - merging only regroups hits)"
    )

    compressed_h5 = os.path.join(stage_dir, f"compressed_{algorithm}.h5")
    write_step2point_hdf5(compressed_showers, compressed_h5, algorithm=algorithm, source_input=input_path)
    print(f"  wrote {compressed_h5}")

    # Mirror convert_to_cc3_format.convert()'s own out_file derivation (it doesn't return the
    # path) via the same helper convert_pipeline2_to_ddml.py uses - convert() derives the algo
    # name from compressed_h5's parent dir (outdir's basename), not from the `algorithm` string,
    # so a naive f"outputs/cc3input_{algorithm}/..." guess here can silently point at a stale
    # file from an unrelated run (e.g. a prior full, non-traced run) instead of this run's output.
    cc3_path = _cc3_output_path(compressed_h5)
    print(f"[2/3] Converting {compressed_h5} -> {cc3_path} ...")
    convert_to_cc3(compressed_h5)
    if not os.path.isfile(cc3_path):
        raise RuntimeError(f"convert_to_cc3_format.convert() finished but expected output {cc3_path} wasn't created.")

    print(f"[3/3] Converting {cc3_path} -> DDML format ...")
    convert_to_ddml_format(cc3_path)
    in_dir, in_name = os.path.split(cc3_path)
    base = in_name[:-3] if in_name.endswith(".h5") else in_name
    cc3_algo = os.path.basename(in_dir).split("cc3input_", 1)[-1]
    ddml_path = os.path.join(in_dir, f"{base}_ddml_{cc3_algo}.h5")

    if copy_to_ddml_models:
        dest = os.path.join(DDML_MODELS_DIR, os.path.basename(ddml_path))
        print(f"Copying {ddml_path} -> {dest} ...")
        shutil.copy2(ddml_path, dest)
        ddml_path = dest

    print(f"Done: {ddml_path}")
    return ddml_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to the raw edm4hep.root file")
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="identity")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Defaults to the file_<seed> parsed from the input filename",
    )
    parser.add_argument("--collections", nargs="+", default=list(DEFAULT_COLLECTIONS))
    parser.add_argument("--outdir", default=None, help="Defaults to outputs/pipeline2_<algorithm>")
    parser.add_argument(
        "--copy-to-ddml-models",
        action="store_true",
        help=f"Also copy the resulting DDML h5 into {DDML_MODELS_DIR}",
    )
    parser.add_argument(
        "--shower-limit",
        type=int,
        default=None,
        help="Only read the first N showers from the input file (for fast, traceable test runs)",
    )
    args = parser.parse_args()
    edm4hep_to_ddml(
        args.input,
        args.algorithm,
        seed=args.seed,
        collections=tuple(args.collections),
        outdir=args.outdir,
        copy_to_ddml_models=args.copy_to_ddml_models,
        shower_limit=args.shower_limit,
    )


if __name__ == "__main__":
    main()
