from __future__ import annotations

import argparse
from pathlib import Path

from step2point.core.edm4hep_root import EDM4hepRootReader
from step2point.io.step2point_hdf5 import Step2PointHDF5Reader
from step2point.metrics.spatial import estimate_shower_axis, longitudinal_radial_phi
from step2point.validation.benchmark_plots import generate_observables_matrix
from step2point.vis import (
    plot_shower_distributions,
    plot_shower_overview,
    plot_shower_projections,
)


def build_reader(input_path: str):
    suffixes = Path(input_path).suffixes
    if suffixes[-1:] == [".root"]:
        return EDM4hepRootReader(input_path)
    if suffixes[-1:] in ([".h5"], [".hdf5"]):
        return Step2PointHDF5Reader(input_path)
    raise ValueError(f"Unsupported input file type for '{input_path}'. Expected .root, .h5, or .hdf5.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate shower inspection plots. PCA is the default axis unless --axis is given."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--shower-index", type=int, help="Optional shower index for single-shower plots.")
    parser.add_argument("--outdir", default="outputs/inspect")
    parser.add_argument(
        "--axis",
        type=float,
        nargs=3,
        help="Optional manual shower axis override: ax ay az. If omitted, PCA is used.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    reader = build_reader(args.input)
    showers = list(reader.iter_showers())

    if args.shower_index is not None:
        if args.shower_index < 0 or args.shower_index >= len(showers):
            raise ValueError(f"--shower-index {args.shower_index} is out of range for {len(showers)} showers.")
        shower = showers[args.shower_index]
        centroid, axis = estimate_shower_axis(shower, axis_override=args.axis)
        long_centroid, _, _ = longitudinal_radial_phi(
            shower,
            centroid=centroid,
            axis=axis,
            longitudinal_origin="centroid",
        )
        long_first, _, _ = longitudinal_radial_phi(
            shower,
            centroid=centroid,
            axis=axis,
            longitudinal_origin="first_deposit",
        )
        print(
            f"shower_id={shower.shower_id} first_point_longitudinal_centroid_origin={float(long_centroid[0]):.3f} mm "
            f"first_point_longitudinal_first_deposit_origin={float(long_first[0]):.3f} mm "
            f"first_point_xyz=({float(shower.x[0]):.3f}, {float(shower.y[0]):.3f}, {float(shower.z[0]):.3f}) mm"
        )
        plot_shower_projections(shower, outdir / f"shower_{shower.shower_id}_projections.png")
        plot_shower_distributions(shower, outdir / f"shower_{shower.shower_id}_distributions.png")
        plot_shower_overview(shower, outdir / f"shower_{shower.shower_id}_overview.png", axis_override=args.axis)

    generate_observables_matrix(
        showers,
        outdir / "dataset_observables.png",
        selected_index=args.shower_index,
        axis_override=args.axis,
    )


if __name__ == "__main__":
    main()
