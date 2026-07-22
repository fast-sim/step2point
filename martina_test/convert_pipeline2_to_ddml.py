"""
Convert a raw pre-step2point edm4hep.root file all the way into DDML format:
run_step2point_pipeline.py (edm4hep.root -> compressed_<algo>.h5) then the
two existing h5 conversion steps (convert_to_cc3_format.convert then
convert_to_DDML_format.convert_to_ddml_format) that would otherwise need to
be run and tracked by hand. Mirrors the exact invocation in
submit_job_new.sh, so e.g. "identity" and "merge_within_cell" runs on the
same raw file are directly comparable (isolates whether a distribution
mismatch comes from step2point's clustering or downstream in DDML).

Usage:
    # by algo name - defaults to file_0, running step2point on RAW_EDM4HEP_TEMPLATE
    # for that seed first if compressed_<algo>.h5 doesn't already exist
    python -m martina_test.convert_pipeline2_to_ddml identity
    python -m martina_test.convert_pipeline2_to_ddml merge_within_cell --seed 3

    # or a full path to an already-produced compressed_<algo>.h5, skipping the
    # step2point stage entirely
    python -m martina_test.convert_pipeline2_to_ddml \\
        /eos/project/f/fast/step2point_files/pipeline2_<algo>/file_<seed>/compressed_<algo>.h5

    # also copy the result into DDML/models/ (what submit_job_new.sh's manual
    # `cp` step does)
    python -m martina_test.convert_pipeline2_to_ddml identity --copy-to-ddml-models

Must be run with the step2point repo root as the working directory (same
requirement as convert_to_cc3_format.py, whose output path is relative to
cwd: outputs/cc3input_<algo>/input_cc3_file_<seed>.h5).
"""

import argparse
import os
import shutil
import subprocess
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from martina_test.convert_to_cc3_format import convert as convert_to_cc3
from martina_test.convert_to_DDML_format import convert_to_ddml_format

DDML_MODELS_DIR = "/eos/user/m/mamozzan/DDML/models"
PIPELINE2_DIR = "/eos/project/f/fast/step2point_files"
RAW_EDM4HEP_TEMPLATE = "/eos/project/f/fast/edm4hep_frombenchmark/sim-E1261AT600AP180-180_file_{seed}.edm4hep.root"
DEFAULT_SEED = 0

# algorithms whose run_step2point_pipeline.py invocation needs no extra args
# beyond --input/--algorithm/--output/--collections (see submit_job_new.sh;
# merge_within_regular_subcell and hdbscan need --compact-xml / HDBSCAN params
# that aren't wired up here)
SIMPLE_ALGORITHMS = {"identity", "merge_within_cell"}


def resolve_compressed_h5_path(algo_or_path: str, seed: int = DEFAULT_SEED) -> str:
    """Accept either a full compressed_<algo>.h5 path or a bare algo name
    (e.g. "identity"), in which case it resolves to
    PIPELINE2_DIR/pipeline2_<algo>/file_<seed>/compressed_<algo>.h5."""
    if algo_or_path.endswith(".h5"):
        return algo_or_path
    algo = algo_or_path
    return os.path.join(PIPELINE2_DIR, f"pipeline2_{algo}", f"file_{seed}", f"compressed_{algo}.h5")


def run_step2point(algo: str, seed: int, compressed_h5_path: str) -> None:
    if algo not in SIMPLE_ALGORITHMS:
        raise ValueError(
            f"Don't know how to invoke run_step2point_pipeline.py for algorithm '{algo}' - only "
            f"{sorted(SIMPLE_ALGORITHMS)} are wired up here (others need extra args, e.g. --compact-xml "
            "for merge_within_regular_subcell, HDBSCAN params for hdbscan). Run it by hand instead, "
            "then pass the resulting compressed_<algo>.h5 path to this script directly."
        )
    raw_edm4hep = RAW_EDM4HEP_TEMPLATE.format(seed=seed)
    if not os.path.isfile(raw_edm4hep):
        raise FileNotFoundError(f"Raw edm4hep input not found: {raw_edm4hep}")

    output_dir = os.path.dirname(compressed_h5_path)
    print(f"[1/3] Running step2point ({algo}) on {raw_edm4hep} -> {output_dir} ...")
    subprocess.run(
        [
            sys.executable,
            "examples/run_step2point_pipeline.py",
            "--input",
            raw_edm4hep,
            "--algorithm",
            algo,
            "--output",
            output_dir,
            "--collections",
            "EcalBarrelCollection",
        ],
        check=True,
    )
    if not os.path.isfile(compressed_h5_path):
        raise RuntimeError(f"run_step2point_pipeline.py finished but expected output {compressed_h5_path} wasn't created.")


def _cc3_output_path(compressed_h5_path: str) -> str:
    """Mirror convert_to_cc3_format.convert()'s own out_file derivation (it
    doesn't return the path), so we know where to find its output without
    re-parsing stdout."""
    seed = int(compressed_h5_path.split("/")[-2].split("_")[1])
    dir_name = compressed_h5_path.split("/")[-3]  # e.g. pipeline2_hdbscan_ms8_mcs40
    if "pipeline2_" in dir_name:
        algo = dir_name.split("pipeline2_")[-1]
    else:
        algo = compressed_h5_path.split("/")[-1].split(".")[0].split("compressed_")[-1]
    return f"outputs/cc3input_{algo}/input_cc3_file_{seed}.h5"


def convert_pipeline2_to_ddml(compressed_h5_path: str, copy_to_ddml_models: bool = False) -> str:
    """Run compressed_<algo>.h5 -> input_cc3_file_<seed>.h5 -> input_cc3_file_<seed>_ddml_<algo>.h5.

    Returns the path to the final DDML-format h5 file.
    """
    if not os.path.isfile(compressed_h5_path):
        raise FileNotFoundError(compressed_h5_path)

    cc3_path = _cc3_output_path(compressed_h5_path)
    if os.path.isfile(cc3_path):
        print(f"[2/3] Skipping cc3 conversion, {cc3_path} already exists ...")
    else:
        print(f"[2/3] Converting {compressed_h5_path} -> {cc3_path} ...")
        convert_to_cc3(compressed_h5_path)
        if not os.path.isfile(cc3_path):
            raise RuntimeError(
                f"convert_to_cc3_format.convert() finished but expected output {cc3_path} wasn't created - "
                "check its out_file naming logic hasn't diverged from _cc3_output_path() above."
            )

    print(f"[3/3] Converting {cc3_path} -> DDML format ...")
    convert_to_ddml_format(cc3_path)

    in_dir, in_name = os.path.split(cc3_path)
    base = in_name[:-3] if in_name.endswith(".h5") else in_name
    algo = os.path.basename(in_dir).split("cc3input_", 1)[-1]
    ddml_path = os.path.join(in_dir, f"{base}_ddml_{algo}.h5")

    if copy_to_ddml_models:
        dest = os.path.join(DDML_MODELS_DIR, os.path.basename(ddml_path))
        print(f"Copying {ddml_path} -> {dest} ...")
        shutil.copy2(ddml_path, dest)
        ddml_path = dest

    print(f"Done: {ddml_path}")
    return ddml_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "algo_or_compressed_h5",
        help="Algo name (e.g. 'identity', 'merge_within_cell') to convert file_<seed> of, "
        "or a full path to a compressed_<algo>.h5 file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Which file_<seed> to convert when given an algo name (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--copy-to-ddml-models",
        action="store_true",
        help=f"Also copy the resulting DDML h5 into {DDML_MODELS_DIR}",
    )
    args = parser.parse_args()
    is_algo_name = not args.algo_or_compressed_h5.endswith(".h5")
    compressed_h5 = resolve_compressed_h5_path(args.algo_or_compressed_h5, seed=args.seed)

    if is_algo_name and os.path.isfile(compressed_h5):
        print(f"[1/3] Skipping step2point, {compressed_h5} already exists ...")
    elif is_algo_name:
        run_step2point(args.algo_or_compressed_h5, args.seed, compressed_h5)

    convert_pipeline2_to_ddml(compressed_h5, copy_to_ddml_models=args.copy_to_ddml_models)


if __name__ == "__main__":
    main()
