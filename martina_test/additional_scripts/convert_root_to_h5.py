"""
Convert edm4hep ROOT files directly into step2point HDF5 format.

This bypasses the step2point pipeline entirely (no compression algorithm,
no validators) -- it just reads the raw SimCalorimeterHit contributions via
EDM4hepRootReader and writes them out with write_step2point_hdf5, so the
result is a plain, uncompressed step2point-format HDF5 file.

Usage:
    # single file
    python martina_test/convert_root_to_h5.py --input /path/to/file.root --output outputs/raw_h5/file_0.h5

    # loop over seeds using the default benchmark filename template
    python martina_test/convert_root_to_h5.py --seeds 0 1 2 3 --output-dir outputs/raw_h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from step2point.io import EDM4hepRootReader, write_step2point_hdf5

DEFAULT_INPUT_TEMPLATE = "/eos/project/f/fast/edm4hep_frombenchmark/sim-E1261AT600AP180-180_file_{seed}.edm4hep.root"
DEFAULT_COLLECTIONS = ("EcalBarrelCollection",)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", help="Path to a single edm4hep ROOT file. Overrides --seeds.")
    parser.add_argument("--output", help="Output HDF5 path, required when using --input.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Seeds to convert using --input-template.")
    parser.add_argument(
        "--input-template",
        default=DEFAULT_INPUT_TEMPLATE,
        help="Filename template with a {seed} placeholder, used with --seeds.",
    )
    parser.add_argument("--output-dir", default="outputs/raw_h5", help="Output directory, used with --seeds.")
    parser.add_argument(
        "--collections",
        nargs="+",
        default=list(DEFAULT_COLLECTIONS),
        help="EDM4hep SimCalorimeterHit collection names to read.",
    )
    parser.add_argument("--shower-limit", type=int, default=None, help="Optional cap on number of showers read.")
    return parser.parse_args()


def convert(input_path: str, output_path: str | Path, collections: list[str], shower_limit: int | None = None) -> None:
    reader = EDM4hepRootReader(str(input_path), collections=tuple(collections), shower_limit=shower_limit)
    showers = list(reader.iter_showers())
    output = write_step2point_hdf5(showers, output_path, algorithm="raw", source_input=str(input_path))
    print(f"wrote {output} ({len(showers)} showers)")


def main():
    args = parse_args()

    if args.input:
        if not args.output:
            raise ValueError("--output is required when using --input.")
        convert(args.input, args.output, args.collections, args.shower_limit)
        return

    if not args.seeds:
        raise ValueError("Provide either --input/--output or --seeds.")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        input_path = args.input_template.format(seed=seed)
        output_path = outdir / f"file_{seed}.h5"
        convert(input_path, output_path, args.collections, args.shower_limit)


if __name__ == "__main__":
    main()
