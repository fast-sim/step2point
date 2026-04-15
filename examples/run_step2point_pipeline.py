from __future__ import annotations

import argparse
from pathlib import Path

from step2point.algorithms.identity import IdentityCompression
from step2point.algorithms.merge_within_cell import MergeWithinCell
from step2point.core.pipeline import Pipeline
from step2point.io.step2point_hdf5 import Step2PointHDF5Reader
from step2point.validation.conservation import CellCountRatioValidator, EnergyConservationValidator
from step2point.validation.profiles import ShowerMomentsValidator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--algorithm", choices=["identity", "merge_within_cell"], default="identity")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    reader = Step2PointHDF5Reader(args.input) # TODO deduce format from extension and execute appropriate file reader
    algorithm = IdentityCompression() if args.algorithm == "identity" else MergeWithinCell()
    validators = [EnergyConservationValidator(), CellCountRatioValidator(), ShowerMomentsValidator()]
    report = Pipeline(reader, algorithm, validators).run()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"compression_summary_{arg.algorithm}.txt").write_text(
        f"compression_stats={len(report.compression_stats)}validation_results={len(report.validation_results)}"
    )


if __name__ == "__main__":
    main()
