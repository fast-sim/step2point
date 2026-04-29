from __future__ import annotations

import argparse
from pathlib import Path

from step2point.algorithms.hdbscan_clustering import HDBSCANClustering
from step2point.algorithms.identity import IdentityCompression
from step2point.algorithms.merge_within_cell import MergeWithinCell
from step2point.algorithms.merge_within_regular_subcell import MergeWithinRegularSubcell
from step2point.geometry.dd4hep.factory_geometry import get_dd4hep_cell_id_encoding
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


def load_showers(input_path: str, collections: tuple[str, ...] | None):
    reader = build_reader(input_path)
    if isinstance(reader, EDM4hepRootReader):
        reader.collections = collections
    return list(reader.iter_showers())


def pair_showers(reference_showers, compared_showers, *, reference_label: str, compared_label: str):
    if len(reference_showers) != len(compared_showers):
        raise ValueError(
            f"Cannot compare {reference_label} to {compared_label}: "
            f"{len(reference_showers)} showers vs {len(compared_showers)} showers."
        )
    pairs = []
    for pre, post in zip(reference_showers, compared_showers, strict=True):
        if pre.shower_id != post.shower_id:
            raise ValueError(
                f"Cannot compare {reference_label} to {compared_label}: shower_id mismatch "
                f"{pre.shower_id} vs {post.shower_id}."
            )
        pairs.append((pre, post))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument(
        "--collections",
        nargs="+",
        default=list(DEFAULT_ROOT_COLLECTIONS),
        help="EDM4hep SimCalorimeterHit collection names for ROOT input.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["identity", "merge_within_cell", "merge_within_regular_subcell", "hdbscan"],
        default="merge_within_cell",
    )
    parser.add_argument("--min-cluster-size", type=int, default=5, help="HDBSCAN min_cluster_size.")
    parser.add_argument("--min-samples", type=int, default=3, help="HDBSCAN min_samples.")
    parser.add_argument("--epsilon", type=float, default=0.0, help="HDBSCAN cluster_selection_epsilon.")
    parser.add_argument(
        "--hdbscan-algorithm",
        choices=["auto", "brute", "kd_tree", "ball_tree"],
        default="brute",
        help="HDBSCAN tree-building algorithm."
    )
    parser.add_argument("--use-time", action="store_true", help="Include time as a clustering feature in HDBSCAN.")
    parser.add_argument(
        "--merge-scope",
        choices=["none", "layer", "system_layer", "cell_id"],
        default="system_layer",
        help="Detector boundary HDBSCAN is not allowed to cross.",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs for HDBSCAN (-1 for all cores).")
    parser.add_argument("--compact-xml", help="DD4hep compact XML required by geometry-aware algorithms.")
    parser.add_argument(
        "--collection-name",
        nargs="+",
        help="Readout collection name(s) required by geometry-aware algorithms.",
    )
    parser.add_argument(
        "--hdbscan-cell-id-encoding",
        help="Cell-ID encoding string used by HDBSCAN to extract system and layer.",
    )
    parser.add_argument("--grid-x", type=int, default=2, help="Number of regular subdivisions along local cell x.")
    parser.add_argument("--grid-y", type=int, default=2, help="Number of regular subdivisions along local cell y/z.")
    parser.add_argument(
        "--position-mode",
        choices=["weighted", "center"],
        default="weighted",
        help="Output position within each subcell: weighted barycenter or geometric center.",
    )
    parser.add_argument(
        "--axis",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Override the shower axis used for validation observables and profiles.",
    )
    parser.add_argument(
        "--origin",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Override the reference point used together with the validation axis.",
    )
    parser.add_argument("--outdir", default="outputs/plots")
    args = parser.parse_args()
    collections = parse_collections(args.collections)

    def resolve_hdbscan_cell_id_encodings() -> tuple[str, ...]:
        if args.hdbscan_cell_id_encoding:
            return (args.hdbscan_cell_id_encoding,)
        if args.compact_xml and args.collection_name:
            return tuple(get_dd4hep_cell_id_encoding(args.compact_xml, name) for name in args.collection_name)
        raise ValueError(
            "hdbscan assumes a cell_id can be decoded to define the unmergeable points: pass either "
            "--hdbscan-cell-id-encoding or --compact-xml together with --collection-name."
        )

    if len(args.input) > 1:
        reference_path = args.input[0]
        reference_showers = load_showers(reference_path, collections)
        base_outdir = Path(args.outdir)
        for index, compared_path in enumerate(args.input[1:], start=1):
            compared_showers = load_showers(compared_path, collections)
            pairs = pair_showers(
                reference_showers,
                compared_showers,
                reference_label=reference_path,
                compared_label=compared_path,
            )
            comparison_outdir = (
                base_outdir
                if len(args.input) == 2
                else base_outdir / f"{index:02d}_{Path(compared_path).stem}"
            )
            generate_benchmark_plots(
                pairs,
                comparison_outdir,
                axis_override=args.axis,
                origin_override=args.origin,
            )
        return

    input_path = args.input[0]
    reader = build_reader(input_path)
    if isinstance(reader, EDM4hepRootReader):
        reader.collections = collections
    if args.algorithm == "identity":
        algorithm = IdentityCompression()
    elif args.algorithm == "merge_within_cell":
        algorithm = MergeWithinCell()
    elif args.algorithm == "hdbscan":
        algorithm = HDBSCANClustering(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_epsilon=args.epsilon,
            use_time=args.use_time,
            merge_scope=args.merge_scope,
            cell_id_encoding=resolve_hdbscan_cell_id_encodings(),
            algorithm=args.hdbscan_algorithm,
            n_jobs=args.n_jobs,
        )
    else:
        if args.compact_xml is None or not args.collection_name:
            raise ValueError("--compact-xml and --collection-name are required for merge_within_regular_subcell.")
        if len(args.collection_name) != 1:
            raise ValueError("merge_within_regular_subcell requires exactly one --collection-name.")
        algorithm = MergeWithinRegularSubcell(
            x_bins=args.grid_x,
            y_bins=args.grid_y,
            position_mode=args.position_mode,
            compact_xml=args.compact_xml,
            collection_name=args.collection_name[0],
        )
    pairs = []
    for shower in reader.iter_showers():
        result = algorithm.compress(shower)
        pairs.append((shower, result.shower))
    generate_benchmark_plots(
        pairs,
        Path(args.outdir),
        axis_override=args.axis,
        origin_override=args.origin,
    )


if __name__ == "__main__":
    main()
