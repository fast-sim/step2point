from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.collections import PolyCollection

from step2point.core.shower import Shower
from step2point.geometry.dd4hep.factory_geometry import (
    BarrelLayout,
    module_cell_strip_polygons_xy,
    module_cell_strip_polygons_zy,
    module_grid_lines_xy_zy,
    module_envelope_outline_xy_zy,
    module_layer_outline_xy_zy,
)


def _segment_bounds(segments: list[np.ndarray]) -> tuple[float, float, float, float]:
    points = np.concatenate(segments, axis=0)
    return (
        float(np.min(points[:, 0])),
        float(np.max(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 1])),
    )


def _polygon_bounds(polygons: list[np.ndarray]) -> tuple[float, float, float, float]:
    points = np.concatenate(polygons, axis=0)
    return (
        float(np.min(points[:, 0])),
        float(np.max(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 1])),
    )


def _collection_bounds(
    segments: list[np.ndarray],
    polygons: list[np.ndarray],
) -> tuple[float, float, float, float]:
    if segments:
        return _segment_bounds(segments)
    if polygons:
        return _polygon_bounds(polygons)
    raise ValueError("No drawable geometry available to determine plot bounds.")


def _expand_bounds(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    frac: float = 0.05,
    min_pad: float = 2.0,
) -> tuple[float, float, float, float]:
    xpad = max((xmax - xmin) * frac, min_pad)
    ypad = max((ymax - ymin) * frac, min_pad)
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


def _scatter_area_from_data_diameter(
    ax: plt.Axes,
    fig: plt.Figure,
    data_diameter_x: float,
) -> float:
    p0 = ax.transData.transform((0.0, 0.0))
    p1 = ax.transData.transform((data_diameter_x, 0.0))
    diameter_px = max(float(abs(p1[0] - p0[0])), 1.0)
    diameter_pt = diameter_px * 72.0 / fig.dpi
    return diameter_pt * diameter_pt


def plot_barrel_wireframe(
    layout: BarrelLayout,
    output_path: str | Path,
    layer_index: int | None = None,
    draw_cells: bool = False,
    sensitive_only: bool = False,
    module_index: int | None = None,
    modules_only: bool = False,
    overlay_shower: Shower | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    xlim_points: tuple[float, float] | None = None,
    ylim_points: tuple[float, float] | None = None,
    zlim_points: tuple[float, float] | None = None,
) -> tuple[Path, Path]:
    if modules_only:
        xy_segments, zy_segments = module_envelope_outline_xy_zy(layout)
        xy_polygons: list[np.ndarray] = []
        zy_polygons: list[np.ndarray] = []
    else:
        selected_layers = [layer_index] if layer_index is not None else [layer.layer_index for layer in layout.layers]
        xy_segments = []
        zy_segments = []
        xy_polygons = []
        zy_polygons = []
        for selected in selected_layers:
            if draw_cells:
                xy_polygons.extend(
                    module_cell_strip_polygons_xy(
                        layout,
                        selected,
                        module_index=module_index,
                        sensitive_only=sensitive_only,
                    )
                )
                zy_polygons.extend(
                    module_cell_strip_polygons_zy(
                        layout,
                        selected,
                        module_index=module_index,
                        sensitive_only=sensitive_only,
                    )
                )
            else:
                outline_xy, outline_zy = module_layer_outline_xy_zy(layout, selected, module_index=module_index)
                xy_segments.extend(outline_xy)
                zy_segments.extend(outline_zy)

    if draw_cells and module_index is not None and sensitive_only:
        fig_xy, ax_xy = plt.subplots(figsize=(20, 20))
        fig_zy, ax_zy = plt.subplots(figsize=(24, 20))
    elif draw_cells and module_index is not None:
        fig_xy, ax_xy = plt.subplots(figsize=(16, 16))
        fig_zy, ax_zy = plt.subplots(figsize=(20, 16))
    else:
        fig_xy, ax_xy = plt.subplots(figsize=(8, 8))
        fig_zy, ax_zy = plt.subplots(figsize=(10, 8))
    line_width = 1.2 if modules_only else (0.55 if draw_cells and module_index is not None else (0.35 if draw_cells else 0.6))
    if xy_polygons:
        ax_xy.add_collection(
            PolyCollection(
                xy_polygons,
                facecolors=(0.121, 0.466, 0.705, 0.08),
                edgecolors="tab:blue",
                linewidths=0.45 if draw_cells and module_index is not None else 0.25,
            )
        )
    if zy_polygons:
        ax_zy.add_collection(
            PolyCollection(
                zy_polygons,
                facecolors=(0.121, 0.466, 0.705, 0.08),
                edgecolors="tab:blue",
                linewidths=0.45 if draw_cells and module_index is not None else 0.25,
            )
        )
    ax_xy.add_collection(LineCollection(xy_segments, colors="tab:blue", linewidths=line_width))
    ax_zy.add_collection(LineCollection(zy_segments, colors="tab:blue", linewidths=line_width))

    ax_xy.autoscale_view()
    ax_zy.autoscale_view()
    if xlim is not None:
        ax_xy.set_xlim(*xlim)
    if ylim is not None:
        ax_xy.set_ylim(*ylim)
        ax_zy.set_ylim(*ylim)
    if zlim is not None:
        ax_zy.set_xlim(*zlim)
    ax_xy.set_aspect("equal")
    fig_xy.canvas.draw()
    fig_zy.canvas.draw()

    if overlay_shower is not None and overlay_shower.n_points > 0:
        energy = np.asarray(overlay_shower.E, dtype=np.float64)
        colors = np.log10(np.clip(energy, 1e-12, None))
        xy_auto_bounds = _expand_bounds(*_collection_bounds(xy_segments, xy_polygons))
        zy_auto_bounds = _expand_bounds(*_collection_bounds(zy_segments, zy_polygons))
        x_bounds = xlim_points if xlim_points is not None else (xy_auto_bounds[0], xy_auto_bounds[1])
        y_bounds = ylim_points if ylim_points is not None else (
            min(xy_auto_bounds[2], zy_auto_bounds[2]),
            max(xy_auto_bounds[3], zy_auto_bounds[3]),
        )
        z_bounds = zlim_points if zlim_points is not None else (zy_auto_bounds[0], zy_auto_bounds[1])
        point_mask = (
            (overlay_shower.x >= x_bounds[0])
            & (overlay_shower.x <= x_bounds[1])
            & (overlay_shower.y >= y_bounds[0])
            & (overlay_shower.y <= y_bounds[1])
            & (overlay_shower.z >= z_bounds[0])
            & (overlay_shower.z <= z_bounds[1])
        )
        if draw_cells and module_index is not None:
            reference_layer_index = layer_index if layer_index is not None else 1
            sensor_thickness = 2.0 * layout.layers[reference_layer_index - 1].sensitive_half_thickness_mm
            fraction = 0.35 if sensitive_only else 0.5
            xy_area = _scatter_area_from_data_diameter(ax_xy, fig_xy, sensor_thickness * fraction)
            zy_area = _scatter_area_from_data_diameter(ax_zy, fig_zy, sensor_thickness * fraction)
            xy_sizes = np.full_like(energy, xy_area, dtype=np.float64)
            zy_sizes = np.full_like(energy, zy_area, dtype=np.float64)
        else:
            xy_sizes = 1.0 + 6.0 * np.sqrt(energy / max(float(np.max(energy)), 1e-12))
            zy_sizes = xy_sizes
        ax_xy.scatter(
            overlay_shower.x[point_mask],
            overlay_shower.y[point_mask],
            s=xy_sizes[point_mask],
            c=colors[point_mask],
            cmap="inferno",
            alpha=0.75 if draw_cells and module_index is not None else 0.8,
            linewidths=0.0,
            zorder=3,
        )
        ax_zy.scatter(
            overlay_shower.z[point_mask],
            overlay_shower.y[point_mask],
            s=zy_sizes[point_mask],
            c=colors[point_mask],
            cmap="inferno",
            alpha=0.75 if draw_cells and module_index is not None else 0.8,
            linewidths=0.0,
            zorder=3,
        )
    ax_xy.set_xlabel("x (mm)")
    ax_xy.set_ylabel("y (mm)")
    ax_zy.set_xlabel("z (mm)")
    ax_zy.set_ylabel("y (mm)")
    module_label = f" module {module_index}" if module_index is not None else ""
    if modules_only:
        title_kind = "module envelopes"
    elif layer_index is None:
        title_kind = "cell wireframe" if draw_cells else "layer/module outline"
        ax_xy.set_title(f"{layout.detector_name}{module_label} full global {title_kind} (XY)")
        ax_zy.set_title(f"{layout.detector_name}{module_label} full global {title_kind} (ZY)")
    else:
        title_kind = "cell wireframe" if draw_cells else "layer/module outline"
        ax_xy.set_title(f"{layout.detector_name}{module_label} layer {layer_index} global {title_kind} (XY)")
        ax_zy.set_title(f"{layout.detector_name}{module_label} layer {layer_index} global {title_kind} (ZY)")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_xy = output.with_name(f"{output.stem}_xy{output.suffix}")
    output_zy = output.with_name(f"{output.stem}_zy{output.suffix}")
    if output.suffix.lower() == ".png" and draw_cells and module_index is not None and sensitive_only:
        save_dpi = 1000
    elif output.suffix.lower() == ".png" and draw_cells and module_index is not None:
        save_dpi = 300
    else:
        save_dpi = None
    fig_xy.savefig(output_xy, bbox_inches="tight", dpi=save_dpi)
    fig_zy.savefig(output_zy, bbox_inches="tight", dpi=save_dpi)
    plt.close(fig_xy)
    plt.close(fig_zy)
    return output_xy, output_zy
