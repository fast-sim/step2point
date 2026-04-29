from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from step2point.algorithms.hdbscan_clustering import HDBSCANClustering
from step2point.algorithms.identity import IdentityCompression
from step2point.algorithms.merge_within_cell import MergeWithinCell
from step2point.algorithms.merge_within_regular_subcell import MergeWithinRegularSubcell
from step2point.geometry.dd4hep.factory_geometry import get_dd4hep_cell_id_encoding
from step2point.io import EDM4hepRootReader, Step2PointHDF5Reader, write_step2point_debug_hdf5, write_step2point_hdf5
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
    parser.add_argument(
        "--algorithm",
        choices=["identity", "merge_within_cell", "merge_within_regular_subcell", "hdbscan"],
        default="identity",
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
        "--outlier-policy",
        choices=["nearest_cluster", "standalone"],
        default="nearest_cluster",
        help="How HDBSCAN outlier points are handled.",
    )
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
        "--debug-events",
        nargs="+",
        type=int,
        help="Selected 0-based shower indices for which to write original points with per-point cluster labels.",
    )
    parser.add_argument(
        "--debug-output",
        help="Optional output path for the debug HDF5. Defaults to debug_<algorithm>.h5 in --output.",
    )
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


def _fallback_debug_labels(algorithm_name: str, shower) -> np.ndarray:
    if algorithm_name == "identity":
        return np.arange(shower.n_points, dtype=np.int64)
    if algorithm_name == "merge_within_cell":
        if shower.cell_id is None:
            raise ValueError("merge_within_cell debug output requires cell_id.")
        _, inverse = np.unique(shower.cell_id, return_inverse=True)
        return inverse.astype(np.int64, copy=False)
    raise ValueError(f"Algorithm '{algorithm_name}' did not provide debug cluster labels.")


def _resolve_hdbscan_cell_id_encodings(args) -> tuple[str, ...]:
    if args.hdbscan_cell_id_encoding:
        return (args.hdbscan_cell_id_encoding,)
    if args.compact_xml and args.collection_name:
        return tuple(get_dd4hep_cell_id_encoding(args.compact_xml, name) for name in args.collection_name)
    raise ValueError(
        "hdbscan assumes a cell_id can be decoded to define the unmergeable points: pass either "
        "--hdbscan-cell-id-encoding or --compact-xml together with --collection-name."
    )


def main():
    args = parse_args()
    reader = build_reader(args.input)
    if isinstance(reader, EDM4hepRootReader):
        reader.collections = parse_collections(args.collections)
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
            outlier_policy=args.outlier_policy,
            merge_scope=args.merge_scope,
            cell_id_encoding=_resolve_hdbscan_cell_id_encodings(args),
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
    validators = [EnergyConservationValidator(), CellCountRatioValidator(), ShowerMomentsValidator()]
    debug_event_indices = set(args.debug_events or [])

    compression_stats: list[dict] = []
    validation_results: list[dict] = []
    compressed_showers = []
    debug_showers = []
    debug_labels = []
    for shower_index, shower in enumerate(reader.iter_showers()):
        result = algorithm.compress(shower)
        compressed_showers.append(result.shower)
        compression_stats.append(result.stats)
        if shower_index in debug_event_indices:
            cluster_label = result.debug_data.get("cluster_label")
            if cluster_label is None:
                cluster_label = _fallback_debug_labels(args.algorithm, shower)
            cluster_label = np.ascontiguousarray(np.asarray(cluster_label), dtype=np.int64)
            if len(cluster_label) != shower.n_points:
                raise ValueError(
                    f"Debug cluster labels for shower index {shower_index} have length {len(cluster_label)} "
                    f"but shower has {shower.n_points} points."
                )
            debug_showers.append(shower.copy())
            debug_labels.append(cluster_label.copy())
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
    n_showers = len(compression_stats)
    mean_points_before = sum(float(stats["n_points_before"]) for stats in compression_stats) / n_showers if n_showers else 0.0
    mean_points_after = sum(float(stats["n_points_after"]) for stats in compression_stats) / n_showers if n_showers else 0.0
    mean_compression_ratio = (
        sum(float(stats["compression_ratio"]) for stats in compression_stats) / n_showers if n_showers else 0.0
    )
    total_points_before = sum(int(stats["n_points_before"]) for stats in compression_stats)
    total_points_after = sum(int(stats["n_points_after"]) for stats in compression_stats)
    total_compression_ratio = total_points_after / total_points_before if total_points_before else 0.0
    (outdir / f"compression_summary_{args.algorithm}.txt").write_text(
        f"compression_stats={len(compression_stats)}\n"
        f"validation_results={len(validation_results)}\n"
        f"mean_n_points_before={mean_points_before:.6f}\n"
        f"mean_n_points_after={mean_points_after:.6f}\n"
        f"mean_compression_ratio={mean_compression_ratio:.6f}\n"
        f"total_n_points_before={total_points_before}\n"
        f"total_n_points_after={total_points_after}\n"
        f"total_compression_ratio={total_compression_ratio:.6f}\n"
        f"output_hdf5={output_h5.name}\n"
    )
    print(f"wrote {output_h5}")
    print(f"wrote {outdir / f'compression_summary_{args.algorithm}.txt'}")
    if debug_showers:
        debug_output_path = Path(args.debug_output) if args.debug_output else outdir / f"debug_{args.algorithm}.h5"
        debug_h5 = write_step2point_debug_hdf5(
            debug_showers,
            debug_labels,
            debug_output_path,
            algorithm=args.algorithm,
            source_input=args.input,
            debug_event_indices=sorted(debug_event_indices),
        )
        print(f"wrote {debug_h5}")


if __name__ == "__main__":
    main()
