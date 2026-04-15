from __future__ import annotations

import argparse
from pathlib import Path

from step2point.algorithms.identity import IdentityCompression
from step2point.algorithms.merge_within_cell import MergeWithinCell
from step2point.core.edm4hep_root import EDM4hepRootReader
from step2point.io.step2point_hdf5 import Step2PointHDF5Reader
from step2point.validation.benchmark_plots import generate_benchmark_plots

ALGORITHMS = {
    "identity": IdentityCompression,
    "merge_within_cell": MergeWithinCell,
}


def build_reader(input_path: str, input_format: str):
    if input_format == "hdf5":
        return Step2PointHDF5Reader(input_path)
    if input_format == "edm4hep_root":
        return EDM4hepRootReader(input_path)
    raise ValueError(f"Unsupported input format: {input_format}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-format", choices=["hdf5", "edm4hep_root"], default="hdf5")  # TODO deduce format from extension
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="merge_within_cell")
    parser.add_argument("--outdir", default="outputs/plots")
    args = parser.parse_args()

    reader = build_reader(args.input, args.input_format)
    algorithm = ALGORITHMS[args.algorithm]()
    pairs = []
    for shower in reader.iter_showers():
        result = algorithm.compress(shower)
        pairs.append((shower, result.shower))
    generate_benchmark_plots(pairs, Path(args.outdir))


if __name__ == "__main__":
    main()
