"""End-to-end HDBSCAN example: compress, validate, visualise.

Usage:

    python examples/run_hdbscan.py \
      --input tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV.h5 \
      --outdir outputs/hdbscan_gamma

Optional flags:

    --min-cluster-size 5   (default 5)
    --min-samples 3        (default 3)
    --epsilon 0.0          (default 0.0)
    --noise-handle nn      (nn | singleton | layer | drop)
    --shower-index 0       (single-shower plots for this event)
    --limit 3              (process only the first N showers)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from step2point.algorithms.hdbscan_clustering import HDBSCANClustering
from step2point.core.pipeline import Pipeline
from step2point.io import EDM4hepRootReader, Step2PointHDF5Reader
from step2point.validation.benchmark_plots import generate_benchmark_plots, generate_observables_matrix
from step2point.validation.conservation import EnergyConservationValidator
from step2point.validation.profiles import ShowerMomentsValidator
from step2point.validation.sanity import ShowerSanityValidator
from step2point.vis import (
    plot_shower_distributions,
    plot_shower_overview,
    plot_shower_projections,
)

DEFAULT_ROOT_COLLECTIONS = (
    "ECalBarrelCollection",
    "ECalEndcapCollection",
    "HCalBarrelCollection",
    "HCalEndcapCollection",
)


def build_reader(input_path: str, collections: list[str] | None = None, limit: int | None = None):
    suffixes = Path(input_path).suffixes
    if suffixes[-1:] == [".root"]:
        reader = EDM4hepRootReader(input_path, shower_limit=limit)
        if collections:
            reader.collections = tuple(collections)
        return reader
    if suffixes[-1:] in ([".h5"], [".hdf5"]):
        return Step2PointHDF5Reader(input_path, shower_limit=limit)
    raise ValueError(f"Unsupported input file type for '{input_path}'. Expected .root, .h5, or .hdf5.")


def main():
    parser = argparse.ArgumentParser(description="Run HDBSCAN compression with full validation and visualisation.")
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--collections",
        nargs="+",
        default=list(DEFAULT_ROOT_COLLECTIONS),
        help="EDM4hep SimCalorimeterHit collection names for ROOT input.",
    )
    parser.add_argument("--outdir", default="outputs/hdbscan")
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--noise-handle", choices=["nn", "singleton", "layer", "drop"], default="nn")
    parser.add_argument("--shower-index", type=int, help="Produce single-shower plots for this event index.")
    parser.add_argument("--limit", type=int, help="Only process the first N showers.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    algorithm = HDBSCANClustering(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.epsilon,
        noise_handle=args.noise_handle,
    )

    # --- 1. Pipeline: compress + validate ---
    reader = build_reader(args.input, args.collections, args.limit)
    validators = [EnergyConservationValidator(), ShowerSanityValidator(), ShowerMomentsValidator()]
    report = Pipeline(reader, algorithm, validators).run()

    print(f"Processed {len(report.compression_stats)} showers\n")
    for i, stats in enumerate(report.compression_stats):
        print(
            f"  shower {i}: "
            f"{stats['n_points_before']} -> {stats['n_points_after']} points "
            f"({stats['compression_ratio']:.3f}x), "
            f"energy {stats['energy_before']:.4f} -> {stats['energy_after']:.4f} GeV"
        )

    print()
    for row in report.validation_results:
        print(f"  [{row['validator']}] shower {row['shower_id']}: {row}")

    # --- 2. Benchmark plots: pre vs post overlays ---
    reader2 = build_reader(args.input, args.collections, args.limit)
    pairs = []
    for shower in reader2.iter_showers():
        result = algorithm.compress(shower)
        pairs.append((shower, result.shower))

    generate_benchmark_plots(pairs, outdir / "benchmark")
    print(f"\nBenchmark plots saved to {outdir / 'benchmark'}/")

    # --- 3. Observables matrix: dataset-level summary ---
    compressed_showers = [post for _, post in pairs]
    generate_observables_matrix(
        compressed_showers,
        outdir / "observables_compressed.png",
        selected_index=args.shower_index,
    )
    print(f"Compressed observables matrix saved to {outdir / 'observables_compressed.png'}")

    original_showers = [pre for pre, _ in pairs]
    generate_observables_matrix(
        original_showers,
        outdir / "observables_original.png",
        selected_index=args.shower_index,
    )
    print(f"Original observables matrix saved to {outdir / 'observables_original.png'}")

    # --- 4. Single-shower visualisation ---
    if args.shower_index is not None:
        if args.shower_index < 0 or args.shower_index >= len(pairs):
            print(f"Warning: --shower-index {args.shower_index} out of range for {len(pairs)} showers, skipping.")
        else:
            pre, post = pairs[args.shower_index]
            sid = pre.shower_id

            for label, shower in [("original", pre), ("compressed", post)]:
                plot_shower_projections(
                    shower, outdir / f"shower_{sid}_{label}_projections.png"
                )
                plot_shower_distributions(
                    shower, outdir / f"shower_{sid}_{label}_distributions.png"
                )
                plot_shower_overview(
                    shower, outdir / f"shower_{sid}_{label}_overview.png"
                )

            print(f"Single-shower plots for shower {sid} saved to {outdir}/")


if __name__ == "__main__":
    main()
