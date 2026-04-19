from __future__ import annotations

import argparse
from pathlib import Path

from step2point.core.shower import Shower
from step2point.geometry.dd4hep.factory_geometry import (
    barrel_layout_debug_report,
    build_barrel_layout_from_collection,
)
from step2point.io import EDM4hepRootReader, Step2PointHDF5Reader
from step2point.vis import plot_barrel_wireframe

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


def load_overlay_shower(input_path: str, shower_index: int, collections: list[str] | None) -> Shower:
    reader = build_reader(input_path)
    if isinstance(reader, EDM4hepRootReader):
        reader.collections = parse_collections(collections)
    showers = list(reader.iter_showers())
    if shower_index < 0 or shower_index >= len(showers):
        raise ValueError(f"--overlay-shower-index {shower_index} is out of range for {len(showers)} showers.")
    return showers[shower_index]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DD4hep detector cell wireframe from XML and factory logic.")
    parser.add_argument("--compact-xml", required=True, help="Path to the main compact XML.")
    parser.add_argument("--collection", required=True, help="Readout collection name, e.g. ECalBarrelCollection.")
    parser.add_argument("--layer", type=int, help="Optional 1-based layer index to plot. If omitted, all layers are drawn.")
    parser.add_argument("--outdir", default="outputs/detector_cells", help="Output directory.")
    parser.add_argument("--format", choices=("svg", "png"), default="png", help="Output format. PNG is the default; use SVG when you want to zoom deeply.")
    draw_group = parser.add_mutually_exclusive_group()
    draw_group.add_argument("--draw-modules", action="store_true", help="Draw only module/stave envelopes with full calorimeter thickness. This is the default view.")
    draw_group.add_argument("--draw-layers", action="store_true", help="Draw layer outlines repeated across modules.")
    draw_group.add_argument("--draw-cells", action="store_true", help="Draw internal cell grid. This is only supported together with --zoom.")
    parser.add_argument("--sensitive-only", action="store_true", help="With --draw-cells, draw only the sensitive slice thickness instead of the full layer thickness.")
    parser.add_argument("--zoom", action="store_true", help="Zoom to a single module. In zoom mode, cells are drawn only for that module.")
    parser.add_argument("--module", type=int, default=1, help="1-based module index used with --zoom.")
    parser.add_argument("--overlay-input", help="Optional HDF5 or EDM4hep ROOT shower source to overlay as xyzE points.")
    parser.add_argument("--overlay-shower-index", type=int, default=0, help="Shower index used with --overlay-input.")
    parser.add_argument(
        "--overlay-collections",
        nargs="+",
        default=list(DEFAULT_ROOT_COLLECTIONS),
        help="EDM4hep SimCalorimeterHit collection names for ROOT overlay input.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed resolved geometry/debug information.")
    return parser.parse_args()


def main():
    args = parse_args()
    layout = build_barrel_layout_from_collection(args.compact_xml, args.collection)
    if args.verbose:
        print(barrel_layout_debug_report(layout))
    if args.zoom and not (1 <= args.module <= layout.numsides):
        raise ValueError(f"--module must be in [1, {layout.numsides}] for {args.collection}")
    if args.draw_modules and (args.layer is not None or args.zoom):
        raise ValueError("--draw-modules cannot be combined with --layer or --zoom")
    if args.draw_cells and not args.zoom:
        raise ValueError("--draw-cells requires --zoom")
    if args.sensitive_only and not args.draw_cells:
        raise ValueError("--sensitive-only requires --draw-cells")

    draw_modules = args.draw_modules or (not args.draw_layers and not args.draw_cells)
    draw_cells = args.draw_cells
    overlay_shower = None
    if args.overlay_input is not None:
        overlay_shower = load_overlay_shower(args.overlay_input, args.overlay_shower_index, args.overlay_collections)
    outdir = Path(args.outdir)
    mode = "modules" if draw_modules else ("cells" if draw_cells else "layers")
    if draw_cells and args.sensitive_only:
        mode = f"{mode}_sensitive"
    scope = f"module_{args.module}" if args.zoom else "detector"
    label = "module_envelopes" if draw_modules else (f"layer_{args.layer}" if args.layer is not None else "all_layers")
    outpath = outdir / f"{args.collection.lower()}_{scope}_{label}_{mode}.{args.format}"
    output_xy, output_zy = plot_barrel_wireframe(
        layout,
        outpath,
        layer_index=args.layer,
        draw_cells=draw_cells,
        sensitive_only=args.sensitive_only,
        module_index=args.module if args.zoom else None,
        modules_only=draw_modules,
        overlay_shower=overlay_shower,
    )
    print(f"wrote {output_xy}")
    print(f"wrote {output_zy}")


if __name__ == "__main__":
    main()
