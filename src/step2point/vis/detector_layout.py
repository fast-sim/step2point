from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

from step2point.core.shower import Shower
from step2point.geometry.dd4hep.bitfield import decode_dd4hep_cell_id
from step2point.geometry.dd4hep.factory_geometry import (
    BarrelLayout,
    module_cell_strip_polygons_xy,
    module_cell_strip_polygons_xz,
    module_cell_strip_polygons_zy,
    module_envelope_outline_xy_xz_zy,
    module_layer_outline_xy_xz_zy,
)
from step2point.vis.detector_layout_utils import (
    PROJECTIONS,
    WorldBounds,
    collection_bounds,
    expand_bounds,
    filter_geometry_to_bounds,
    layer_intersects_ylim,
    overlay_color_spec,
    scatter_area_from_data_diameter,
    x_bins_intersect_limits,
    z_bins_intersect_limits,
)


def plot_barrel_wireframe(
    layout: BarrelLayout,
    output_path: str | Path,
    layer_index: int | None = None,
    draw_cells: bool = False,
    sensitive_only: bool = False,
    module_index: int | None = None,
    modules_only: bool = False,
    overlay_shower: Shower | None = None,
    annotate_cell_id: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
) -> tuple[Path, Path, Path]:
    bounds = WorldBounds(xlim=xlim, ylim=ylim, zlim=zlim)
    if modules_only:
        xy_segments, xz_segments, zy_segments = module_envelope_outline_xy_xz_zy(layout)
        xy_polygons: list[np.ndarray] = []
        xz_polygons: list[np.ndarray] = []
        zy_polygons: list[np.ndarray] = []
    else:
        selected_layers = [layer_index] if layer_index is not None else [layer.layer_index for layer in layout.layers]
        xy_segments = []
        xz_segments = []
        zy_segments = []
        xy_polygons = []
        xz_polygons = []
        zy_polygons = []
        for selected in selected_layers:
            if draw_cells:
                layer = layout.layers[selected - 1]
                if z_bins_intersect_limits(layer, zlim):
                    xy_polygons.extend(
                        module_cell_strip_polygons_xy(
                            layout,
                            selected,
                            module_index=module_index,
                            sensitive_only=sensitive_only,
                        )
                    )
                if layer_intersects_ylim(layout, layer, int(module_index), sensitive_only, ylim):
                    xz_polygons.extend(
                        module_cell_strip_polygons_xz(
                            layout,
                            selected,
                            module_index=module_index,
                            sensitive_only=sensitive_only,
                        )
                    )
                if x_bins_intersect_limits(layout, layer, int(module_index), sensitive_only, xlim):
                    zy_polygons.extend(
                        module_cell_strip_polygons_zy(
                            layout,
                            selected,
                            module_index=module_index,
                            sensitive_only=sensitive_only,
                        )
                    )
            else:
                outline_xy, outline_xz, outline_zy = module_layer_outline_xy_xz_zy(
                    layout,
                    selected,
                    module_index=module_index,
                )
                xy_segments.extend(outline_xy)
                xz_segments.extend(outline_xz)
                zy_segments.extend(outline_zy)

    if draw_cells and module_index is not None and sensitive_only:
        fig_xy, ax_xy = plt.subplots(figsize=(20, 20))
        fig_xz, ax_xz = plt.subplots(figsize=(20, 20))
        fig_zy, ax_zy = plt.subplots(figsize=(24, 20))
    elif draw_cells and module_index is not None:
        fig_xy, ax_xy = plt.subplots(figsize=(16, 16))
        fig_xz, ax_xz = plt.subplots(figsize=(16, 16))
        fig_zy, ax_zy = plt.subplots(figsize=(20, 16))
    else:
        fig_xy, ax_xy = plt.subplots(figsize=(8, 8))
        fig_xz, ax_xz = plt.subplots(figsize=(10, 8))
        fig_zy, ax_zy = plt.subplots(figsize=(10, 8))

    xy_segments, xy_polygons = filter_geometry_to_bounds(xy_segments, xy_polygons, bounds, PROJECTIONS["xy"])
    xz_segments, xz_polygons = filter_geometry_to_bounds(xz_segments, xz_polygons, bounds, PROJECTIONS["xz"])
    zy_segments, zy_polygons = filter_geometry_to_bounds(zy_segments, zy_polygons, bounds, PROJECTIONS["zy"])

    line_width = 1.2 if modules_only else (0.55 if draw_cells and module_index is not None else (0.35 if draw_cells else 0.6))
    cell_face = (0.121, 0.466, 0.705, 0.10)
    cell_edge = (0.121, 0.466, 0.705, 0.28)
    if xy_polygons:
        ax_xy.add_collection(
            PolyCollection(
                xy_polygons,
                facecolors=cell_face,
                edgecolors=cell_edge,
                linewidths=0.60 if draw_cells and module_index is not None else 0.35,
            )
        )
    if xz_polygons:
        ax_xz.add_collection(
            PolyCollection(
                xz_polygons,
                facecolors=cell_face,
                edgecolors=cell_edge,
                linewidths=0.70 if draw_cells and module_index is not None else 0.40,
            )
        )
    if zy_polygons:
        ax_zy.add_collection(
            PolyCollection(
                zy_polygons,
                facecolors=cell_face,
                edgecolors=cell_edge,
                linewidths=0.70 if draw_cells and module_index is not None else 0.40,
            )
        )
    ax_xy.add_collection(LineCollection(xy_segments, colors="tab:blue", linewidths=line_width))
    ax_xz.add_collection(LineCollection(xz_segments, colors="tab:blue", linewidths=line_width))
    ax_zy.add_collection(LineCollection(zy_segments, colors="tab:blue", linewidths=line_width))

    ax_xy.autoscale_view()
    ax_xz.autoscale_view()
    ax_zy.autoscale_view()
    if xlim is not None:
        ax_xy.set_xlim(*xlim)
        ax_xz.set_xlim(*xlim)
    if ylim is not None:
        ax_xy.set_ylim(*ylim)
        ax_zy.set_ylim(*ylim)
    if zlim is not None:
        ax_xz.set_ylim(*zlim)
        ax_zy.set_xlim(*zlim)
    ax_xy.set_aspect("equal")
    ax_xz.set_aspect("equal")
    fig_xy.canvas.draw()
    fig_xz.canvas.draw()
    fig_zy.canvas.draw()

    if overlay_shower is not None and overlay_shower.n_points > 0:
        energy = np.asarray(overlay_shower.E, dtype=np.float64)
        xy_auto_bounds = expand_bounds(*collection_bounds(xy_segments, xy_polygons))
        xz_auto_bounds = expand_bounds(*collection_bounds(xz_segments, xz_polygons))
        zy_auto_bounds = expand_bounds(*collection_bounds(zy_segments, zy_polygons))
        resolved_bounds = WorldBounds.resolved(
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            fallback_x=(
                min(xy_auto_bounds[0], xz_auto_bounds[0]),
                max(xy_auto_bounds[1], xz_auto_bounds[1]),
            ),
            fallback_y=(
                min(xy_auto_bounds[2], zy_auto_bounds[2]),
                max(xy_auto_bounds[3], zy_auto_bounds[3]),
            ),
            fallback_z=(zy_auto_bounds[0], zy_auto_bounds[1]),
        )
        point_mask = resolved_bounds.point_mask(overlay_shower)
        if annotate_cell_id and module_index is not None and overlay_shower.cell_id is not None:
            module_mask = np.array(
                [
                    decode_dd4hep_cell_id(int(cell_id), layout.cell_id_encoding).get("module") == module_index
                    for cell_id in overlay_shower.cell_id
                ],
                dtype=bool,
            )
            point_mask &= module_mask
        if draw_cells and module_index is not None:
            reference_layer_index = layer_index if layer_index is not None else 1
            sensor_thickness = 2.0 * layout.layers[reference_layer_index - 1].sensitive_half_thickness_mm
            fraction = 0.35 if sensitive_only else 0.5
            xy_area = scatter_area_from_data_diameter(ax_xy, fig_xy, sensor_thickness * fraction)
            xz_area = scatter_area_from_data_diameter(ax_xz, fig_xz, sensor_thickness * fraction)
            zy_area = scatter_area_from_data_diameter(ax_zy, fig_zy, sensor_thickness * fraction)
            xy_sizes = np.full_like(energy, xy_area, dtype=np.float64)
            xz_sizes = np.full_like(energy, xz_area, dtype=np.float64)
            zy_sizes = np.full_like(energy, zy_area, dtype=np.float64)
        else:
            xy_sizes = 1.0 + 6.0 * np.sqrt(energy / max(float(np.max(energy)), 1e-12))
            xz_sizes = xy_sizes
            zy_sizes = xy_sizes
        color_values, cmap = overlay_color_spec(overlay_shower, point_mask)
        ax_xy.scatter(
            overlay_shower.x[point_mask],
            overlay_shower.y[point_mask],
            s=xy_sizes[point_mask],
            c=color_values,
            cmap=cmap,
            alpha=0.75 if draw_cells and module_index is not None else 0.8,
            linewidths=0.0,
            zorder=3,
        )
        ax_xz.scatter(
            overlay_shower.x[point_mask],
            overlay_shower.z[point_mask],
            s=xz_sizes[point_mask],
            c=color_values,
            cmap=cmap,
            alpha=0.75 if draw_cells and module_index is not None else 0.8,
            linewidths=0.0,
            zorder=3,
        )
        ax_zy.scatter(
            overlay_shower.z[point_mask],
            overlay_shower.y[point_mask],
            s=zy_sizes[point_mask],
            c=color_values,
            cmap=cmap,
            alpha=0.75 if draw_cells and module_index is not None else 0.8,
            linewidths=0.0,
            zorder=3,
        )
        if annotate_cell_id and overlay_shower.cell_id is not None:
            selected_indices = np.flatnonzero(point_mask)
            if selected_indices.size:
                first_decoded = decode_dd4hep_cell_id(
                    int(overlay_shower.cell_id[selected_indices[0]]),
                    layout.cell_id_encoding,
                )
                label_fields = tuple(field for field in ("module", "layer", "x", "y", "z") if field in first_decoded)
            else:
                label_fields = ()
            for idx in selected_indices:
                decoded = decode_dd4hep_cell_id(int(overlay_shower.cell_id[idx]), layout.cell_id_encoding)
                label = ", ".join(f"{field}={decoded[field]}" for field in label_fields if field in decoded)
                if not label:
                    continue
                ax_xy.annotate(
                    label,
                    (float(overlay_shower.x[idx]), float(overlay_shower.y[idx])),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                    color="black",
                    alpha=0.8,
                    zorder=4,
                )
                ax_xz.annotate(
                    label,
                    (float(overlay_shower.x[idx]), float(overlay_shower.z[idx])),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                    color="black",
                    alpha=0.8,
                    zorder=4,
                )
                ax_zy.annotate(
                    label,
                    (float(overlay_shower.z[idx]), float(overlay_shower.y[idx])),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                    color="black",
                    alpha=0.8,
                    zorder=4,
                )
    ax_xy.set_xlabel(PROJECTIONS["xy"].xlabel)
    ax_xy.set_ylabel(PROJECTIONS["xy"].ylabel)
    ax_xz.set_xlabel(PROJECTIONS["xz"].xlabel)
    ax_xz.set_ylabel(PROJECTIONS["xz"].ylabel)
    ax_zy.set_xlabel(PROJECTIONS["zy"].xlabel)
    ax_zy.set_ylabel(PROJECTIONS["zy"].ylabel)
    module_label = f" module {module_index}" if module_index is not None else ""
    if modules_only:
        title_kind = "module envelopes"
        ax_xy.set_title(f"{layout.detector_name}{module_label} full global {title_kind} (XY)")
        ax_xz.set_title(f"{layout.detector_name}{module_label} full global {title_kind} (XZ)")
        ax_zy.set_title(f"{layout.detector_name}{module_label} full global {title_kind} (ZY)")
    elif layer_index is None:
        title_kind = "cell wireframe" if draw_cells else "layer/module outline"
        ax_xy.set_title(f"{layout.detector_name}{module_label} full global {title_kind} (XY)")
        ax_xz.set_title(f"{layout.detector_name}{module_label} full global {title_kind} (XZ)")
        ax_zy.set_title(f"{layout.detector_name}{module_label} full global {title_kind} (ZY)")
    else:
        title_kind = "cell wireframe" if draw_cells else "layer/module outline"
        ax_xy.set_title(f"{layout.detector_name}{module_label} layer {layer_index} global {title_kind} (XY)")
        ax_xz.set_title(f"{layout.detector_name}{module_label} layer {layer_index} global {title_kind} (XZ)")
        ax_zy.set_title(f"{layout.detector_name}{module_label} layer {layer_index} global {title_kind} (ZY)")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_xy = output.with_name(f"{output.stem}_xy{output.suffix}")
    output_xz = output.with_name(f"{output.stem}_xz{output.suffix}")
    output_zy = output.with_name(f"{output.stem}_zy{output.suffix}")
    if output.suffix.lower() == ".png" and draw_cells and module_index is not None and sensitive_only:
        save_dpi = 300
    elif output.suffix.lower() == ".png" and draw_cells and module_index is not None:
        save_dpi = 150
    else:
        save_dpi = None
    fig_xy.savefig(output_xy, bbox_inches="tight", dpi=save_dpi)
    fig_xz.savefig(output_xz, bbox_inches="tight", dpi=save_dpi)
    fig_zy.savefig(output_zy, bbox_inches="tight", dpi=save_dpi)
    plt.close(fig_xy)
    plt.close(fig_xz)
    plt.close(fig_zy)
    return output_xy, output_xz, output_zy
