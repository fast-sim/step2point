from __future__ import annotations

import argparse
from pathlib import Path

from step2point.algorithms.identity import IdentityCompression
from step2point.algorithms.merge_within_cell import MergeWithinCell
from step2point.io import EDM4hepRootReader, Step2PointHDF5Reader, write_step2point_hdf5
from step2point.validation.conservation import CellCountRatioValidator, EnergyConservationValidator
from step2point.validation.profiles import ShowerMomentsValidator

DEFAULT_ROOT_COLLECTIONS = (
    "ECalBarrelCollection",
    "ECalEndcapCollection",
    "HCalBarrelCollection",
    "HCalEndcapCollection",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--collections",
        nargs="+",
        default=list(DEFAULT_ROOT_COLLECTIONS),
        help="EDM4hep SimCalorimeterHit collection names for ROOT input.",
    )
    parser.add_argument("--algorithm", choices=["identity", "merge_within_cell"], default="identity")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def build_reader(input_path: str):
    suffixes = Path(input_path).suffixes
    if suffixes[-1:] == [".root"]:
        return EDM4hepRootReader(input_path)
    if suffixes[-1:] in ([".h5"], [".hdf5"]):
        return Step2PointHDF5Reader(input_path)
    raise ValueError(f"Unsupported input file type for '{input_path}'. Expected .root, .h5, or .hdf5.")


def parse_collections(collections: list[str] | None) -> tuple[str, ...] | None:
    if not collections:
        return None
    return tuple(collections)


def main():
    args = parse_args()
    reader = build_reader(args.input)
    if isinstance(reader, EDM4hepRootReader):
        reader.collections = parse_collections(args.collections)
    algorithm = IdentityCompression() if args.algorithm == "identity" else MergeWithinCell()
    validators = [EnergyConservationValidator(), CellCountRatioValidator(), ShowerMomentsValidator()]

    compression_stats: list[dict] = []
    validation_results: list[dict] = []
    compressed_showers = []
    for shower in reader.iter_showers():
        result = algorithm.compress(shower)
        compressed_showers.append(result.shower)
        compression_stats.append(result.stats)
        for validator in validators:
            vr = validator.run(shower, result.shower)
            validation_results.append({"validator": vr.name, "shower_id": shower.shower_id, **vr.metrics})

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    output_h5 = write_step2point_hdf5(
        compressed_showers,
        outdir / f"compressed_{args.algorithm}.h5",
        algorithm=args.algorithm,
        source_input=args.input,
    )
    (outdir / f"compression_summary_{args.algorithm}.txt").write_text(
        f"compression_stats={len(compression_stats)}\n"
        f"validation_results={len(validation_results)}\n"
        f"output_hdf5={output_h5.name}\n"
    )
    print(f"wrote {output_h5}")
    print(f"wrote {outdir / f'compression_summary_{args.algorithm}.txt'}")


if __name__ == "__main__":
    main()
