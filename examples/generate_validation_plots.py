from __future__ import annotations

import argparse
from pathlib import Path

from step2point.algorithms.hdbscan_clustering import HDBSCANClustering
from step2point.algorithms.identity import IdentityCompression
from step2point.algorithms.merge_within_cell import MergeWithinCell
from step2point.algorithms.merge_within_regular_subcell import MergeWithinRegularSubcell
from step2point.io import EDM4hepRootReader, Step2PointHDF5Reader
from step2point.validation.benchmark_plots import generate_benchmark_plots

DEFAULT_ROOT_COLLECTIONS = (
    "ECalBarrelCollection",
    "ECalEndcapCollection",
    "HCalBarrelCollection",
    "HCalEndcapCollection",
)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--collections",
        nargs="+",
        default=list(DEFAULT_ROOT_COLLECTIONS),
        help="EDM4hep SimCalorimeterHit collection names for ROOT input.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["identity", "merge_within_cell", "merge_within_regular_subcell", "hdbscan_clustering"],
        default="merge_within_cell",
    )
    parser.add_argument("--min-cluster-size", type=int, default=5, help="HDBSCAN min_cluster_size.")
    parser.add_argument("--min-samples", type=int, default=3, help="HDBSCAN min_samples.")
    parser.add_argument("--epsilon", type=float, default=0.0, help="HDBSCAN cluster_selection_epsilon.")
    parser.add_argument(
        "--low-energy-deposits-handler",
        choices=["nn", "singleton", "layer", "drop"],
        default="nn",
        help="Strategy for low-energy deposits in HDBSCAN.",
    )
    parser.add_argument(
        "--hdbscan-algorithm",
        choices=["auto", "brute", "kd_tree", "ball_tree"],
        default="brute",
        help="HDBSCAN tree-building algorithm."
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs for HDBSCAN (-1 for all cores).")
    parser.add_argument("--compact-xml", help="DD4hep compact XML required by geometry-aware algorithms.")
    parser.add_argument("--collection-name", help="DD4hep readout collection name required by geometry-aware algorithms.")
    parser.add_argument("--grid-x", type=int, default=2, help="Number of regular subdivisions along local cell x.")
    parser.add_argument("--grid-y", type=int, default=2, help="Number of regular subdivisions along local cell y/z.")
    parser.add_argument(
        "--position-mode",
        choices=["weighted", "center"],
        default="weighted",
        help="Output position within each subcell: weighted barycenter or geometric center.",
    )
    parser.add_argument("--outdir", default="outputs/plots")
    args = parser.parse_args()

    reader = build_reader(args.input)
    if isinstance(reader, EDM4hepRootReader):
        reader.collections = parse_collections(args.collections)
    if args.algorithm == "identity":
        algorithm = IdentityCompression()
    elif args.algorithm == "merge_within_cell":
        algorithm = MergeWithinCell()
    elif args.algorithm == "hdbscan_clustering":
        algorithm = HDBSCANClustering(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_epsilon=args.epsilon,
            low_energy_deposits_handler=args.low_energy_deposits_handler,
            algorithm=args.hdbscan_algorithm,
            n_jobs=args.n_jobs,
        )
    else:
        if args.compact_xml is None or args.collection_name is None:
            raise ValueError("--compact-xml and --collection-name are required for merge_within_regular_subcell.")
        algorithm = MergeWithinRegularSubcell(
            x_bins=args.grid_x,
            y_bins=args.grid_y,
            position_mode=args.position_mode,
            compact_xml=args.compact_xml,
            collection_name=args.collection_name,
        )
    pairs = []
    for shower in reader.iter_showers():
        result = algorithm.compress(shower)
        pairs.append((shower, result.shower))
    generate_benchmark_plots(pairs, Path(args.outdir))


if __name__ == "__main__":
    main()
