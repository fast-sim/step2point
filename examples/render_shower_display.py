from __future__ import annotations

import argparse
from pathlib import Path

from step2point.core.shower import Shower
from step2point.io import EDM4hepRootReader, Step2PointHDF5Reader
from step2point.vis import (
    render_shower_display_3d,
    render_shower_display_comparison_3d,
    render_shower_display_triptych_3d,
)

DEFAULT_ROOT_COLLECTIONS = (
    "ECalBarrelCollection",
    "ECalEndcapCollection",
    "HCalBarrelCollection",
    "HCalEndcapCollection",
)


def build_reader(input_path: str, collections: tuple[str, ...] | None):
    suffixes = Path(input_path).suffixes
    if suffixes[-1:] == [".root"]:
        return EDM4hepRootReader(input_path, collections=collections)
    if suffixes[-1:] in ([".h5"], [".hdf5"]):
        return Step2PointHDF5Reader(input_path)
    raise ValueError(f"Unsupported input file type for '{input_path}'. Expected .root, .h5, or .hdf5.")


def load_shower(input_path: str, shower_index: int, collections: tuple[str, ...] | None) -> Shower:
    reader = build_reader(input_path, collections)
    for index, shower in enumerate(reader.iter_showers()):
        if index == shower_index:
            return shower
    raise ValueError(f"--shower-index {shower_index} is out of range for {input_path}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Render a presentation-oriented 3D shower display.")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One, two, or three HDF5 or EDM4hep ROOT files.",
    )
    parser.add_argument("--label", nargs="+", help=argparse.SUPPRESS)
    parser.add_argument("--panel-title", nargs="+", help="Optional panel titles for 2- or 3-panel layouts.")
    parser.add_argument("--panel-subtitle", nargs="+", help="Optional panel subtitles for 2- or 3-panel layouts.")
    parser.add_argument("--shower-index", type=int, default=0, help="Shower index to render from each input file.")
    parser.add_argument("--out", required=True, help="Output image path, typically .png.")
    parser.add_argument(
        "--collections",
        nargs="+",
        default=list(DEFAULT_ROOT_COLLECTIONS),
        help="EDM4hep SimCalorimeterHit collection names for ROOT input.",
    )
    parser.add_argument(
        "--axis",
        type=float,
        nargs=3,
        metavar=("AX", "AY", "AZ"),
        help="Optional incident-axis override. If omitted, primary momentum is used when available.",
    )
    parser.add_argument(
        "--incident-label",
        help="Optional label shown next to the incident line. Defaults to primary MCParticle energy and type.",
    )
    parser.add_argument("--title", help=argparse.SUPPRESS)
    parser.add_argument("--view", type=float, nargs=2, default=(20.0, -58.0), metavar=("ELEV", "AZIM"))
    parser.add_argument("--figsize", type=float, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--dpi", type=int, default=240, help="Output DPI for raster formats.")
    parser.add_argument(
        "--crop-percentile",
        type=float,
        metavar="PERCENT",
        help="Auto-crop to the smallest cylinder around the shower axis containing this percent of deposited energy.",
    )
    parser.add_argument("--xlim", type=float, nargs=2, metavar=("XMIN", "XMAX"), help="Manual x crop/range.")
    parser.add_argument("--ylim", type=float, nargs=2, metavar=("YMIN", "YMAX"), help="Manual y crop/range.")
    parser.add_argument("--zlim", type=float, nargs=2, metavar=("ZMIN", "ZMAX"), help="Manual z crop/range.")
    return parser.parse_args()


def main():
    args = parse_args()
    collections = tuple(args.collections) if args.collections else None
    showers = [load_shower(path, args.shower_index, collections) for path in args.input]
    count = len(showers)
    common_kwargs = dict(
        axis_override=args.axis,
        incident_label=args.incident_label,
        view=tuple(args.view),
        dpi=args.dpi,
        crop_percentile=args.crop_percentile,
        xlim=tuple(args.xlim) if args.xlim else None,
        ylim=tuple(args.ylim) if args.ylim else None,
        zlim=tuple(args.zlim) if args.zlim else None,
    )
    if count == 1:
        figsize = tuple(args.figsize) if args.figsize else (16.0, 12.0)
        output = render_shower_display_3d(
            showers,
            args.out,
            figsize=figsize,
            **common_kwargs,
        )
    elif count == 2:
        figsize = tuple(args.figsize) if args.figsize else (14.5, 10.5)
        output = render_shower_display_comparison_3d(
            showers,
            args.out,
            panel_titles=args.panel_title,
            panel_subtitles=args.panel_subtitle,
            figsize=figsize,
            **common_kwargs,
        )
    elif count == 3:
        figsize = tuple(args.figsize) if args.figsize else (20.0, 10.5)
        output = render_shower_display_triptych_3d(
            showers,
            args.out,
            panel_titles=args.panel_title,
            panel_subtitles=args.panel_subtitle,
            figsize=figsize,
            **common_kwargs,
        )
    else:
        raise ValueError("--input must contain 1, 2, or 3 files.")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
