from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import ListedColormap

from step2point.core.shower import Shower
from step2point.geometry.dd4hep.bitfield import decode_dd4hep_cell_id
from step2point.geometry.dd4hep.factory_geometry import (
    BarrelLayout,
    barrel_module_basis,
    module_cell_strip_polygons_xy,
    module_cell_strip_polygons_xz,
    module_cell_strip_polygons_zy,
    module_envelope_outline_xy_xz_zy,
    module_layer_outline_xy_xz_zy,
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


def _polygon_intersects_bounds(
    polygon: np.ndarray,
    x_bounds: tuple[float, float] | None,
    y_bounds: tuple[float, float] | None,
) -> bool:
    xmin = float(np.min(polygon[:, 0]))
    xmax = float(np.max(polygon[:, 0]))
    ymin = float(np.min(polygon[:, 1]))
    ymax = float(np.max(polygon[:, 1]))
    if x_bounds is not None and (xmax <= x_bounds[0] or xmin >= x_bounds[1]):
        return False
    if y_bounds is not None and (ymax <= y_bounds[0] or ymin >= y_bounds[1]):
        return False
    return True


def _segment_intersects_bounds(
    segment: np.ndarray,
    x_bounds: tuple[float, float] | None,
    y_bounds: tuple[float, float] | None,
) -> bool:
    xmin = float(np.min(segment[:, 0]))
    xmax = float(np.max(segment[:, 0]))
    ymin = float(np.min(segment[:, 1]))
    ymax = float(np.max(segment[:, 1]))
    if x_bounds is not None and (xmax <= x_bounds[0] or xmin >= x_bounds[1]):
        return False
    if y_bounds is not None and (ymax <= y_bounds[0] or ymin >= y_bounds[1]):
        return False
    return True


def _filter_geometry_to_bounds(
    segments: list[np.ndarray],
    polygons: list[np.ndarray],
    x_bounds: tuple[float, float] | None,
    y_bounds: tuple[float, float] | None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    filtered_segments = [segment for segment in segments if _segment_intersects_bounds(segment, x_bounds, y_bounds)]
    filtered_polygons = [polygon for polygon in polygons if _polygon_intersects_bounds(polygon, x_bounds, y_bounds)]
    return filtered_segments, filtered_polygons


def _z_bins_intersect_limits(layer, zlim: tuple[float, float] | None) -> bool:
    if zlim is None:
        return True
    max_z_index = int(np.ceil(layer.half_z_mm / layer.pitch_z_mm - 0.5))
    z_centers = np.arange(-max_z_index, max_z_index + 1, dtype=np.int32) * layer.pitch_z_mm
    return bool(np.any((z_centers >= zlim[0]) & (z_centers <= zlim[1])))


def _x_bins_intersect_limits(
    layout: BarrelLayout,
    layer,
    module_index: int,
    sensitive_only: bool,
    xlim: tuple[float, float] | None,
) -> bool:
    if xlim is None:
        return True
    center_xy, radial, tangent = barrel_module_basis(layout, layer.layer_index, module_index)
    radial_center_xy = (
        center_xy + (layer.sensitive_radius_mm - layout.sect_center_radius_mm) * radial
        if sensitive_only
        else center_xy + (layer.layer_center_radius_mm - layout.sect_center_radius_mm) * radial
    )
    max_x_index = int(np.ceil(layer.half_tangent_mm / layer.pitch_tangent_mm - 0.5))
    x_centers = np.arange(-max_x_index, max_x_index + 1, dtype=np.int32) * layer.pitch_tangent_mm
    projected_x = radial_center_xy[0] + x_centers.astype(np.float64) * tangent[0]
    return bool(np.any((projected_x >= xlim[0]) & (projected_x <= xlim[1])))


def _layer_intersects_ylim(
    layout: BarrelLayout,
    layer,
    module_index: int,
    sensitive_only: bool,
    ylim: tuple[float, float] | None,
) -> bool:
    if ylim is None:
        return True
    center_xy, radial, tangent = barrel_module_basis(layout, layer.layer_index, module_index)
    radial_half_extent = layer.sensitive_half_thickness_mm if sensitive_only else layer.half_thickness_mm
    radial_center_xy = (
        center_xy + (layer.sensitive_radius_mm - layout.sect_center_radius_mm) * radial
        if sensitive_only
        else center_xy + (layer.layer_center_radius_mm - layout.sect_center_radius_mm) * radial
    )
    corners_xy = np.array(
        [
            radial_center_xy - layer.half_tangent_mm * tangent - radial_half_extent * radial,
            radial_center_xy + layer.half_tangent_mm * tangent - radial_half_extent * radial,
            radial_center_xy + layer.half_tangent_mm * tangent + radial_half_extent * radial,
            radial_center_xy - layer.half_tangent_mm * tangent + radial_half_extent * radial,
        ],
        dtype=np.float64,
    )
    ymin = float(np.min(corners_xy[:, 1]))
    ymax = float(np.max(corners_xy[:, 1]))
    return not (ymax < ylim[0] or ymin > ylim[1])


def _cluster_label_cmap() -> ListedColormap:
    colors = list(plt.get_cmap("tab20").colors)
    colors.extend(plt.get_cmap("Dark2").colors)
    return ListedColormap(colors[:28], name="step2point_cluster_labels")


def _overlay_color_spec(shower: Shower, point_mask: np.ndarray) -> tuple[np.ndarray, str | None]:
    cluster_label = shower.metadata.get("cluster_label")
    if cluster_label is not None:
        labels = np.asarray(cluster_label, dtype=np.int64)
        unique_labels, inverse = np.unique(labels[point_mask], return_inverse=True)
        if unique_labels.size:
            return inverse.astype(np.float64), _cluster_label_cmap()
    energy = np.asarray(shower.E, dtype=np.float64)
    return np.log10(np.clip(energy[point_mask], 1e-12, None)), "inferno"


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
                if _z_bins_intersect_limits(layer, zlim):
                    xy_polygons.extend(
                        module_cell_strip_polygons_xy(
                            layout,
                            selected,
                            module_index=module_index,
                            sensitive_only=sensitive_only,
                        )
                    )
                if _layer_intersects_ylim(layout, layer, int(module_index), sensitive_only, ylim):
                    xz_polygons.extend(
                        module_cell_strip_polygons_xz(
                            layout,
                            selected,
                            module_index=module_index,
                            sensitive_only=sensitive_only,
                        )
                    )
                if _x_bins_intersect_limits(layout, layer, int(module_index), sensitive_only, xlim):
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

    xy_segments, xy_polygons = _filter_geometry_to_bounds(xy_segments, xy_polygons, xlim, ylim)
    xz_segments, xz_polygons = _filter_geometry_to_bounds(xz_segments, xz_polygons, xlim, zlim)
    zy_segments, zy_polygons = _filter_geometry_to_bounds(zy_segments, zy_polygons, zlim, ylim)

    line_width = 1.2 if modules_only else (0.55 if draw_cells and module_index is not None else (0.35 if draw_cells else 0.6))
    cell_face = (0.121, 0.466, 0.705, 0.10)
    if xy_polygons:
        ax_xy.add_collection(
            PolyCollection(
                xy_polygons,
                facecolors=cell_face,
                edgecolors=cell_face,
                linewidths=0.40 if draw_cells and module_index is not None else 0.25,
            )
        )
    if xz_polygons:
        ax_xz.add_collection(
            PolyCollection(
                xz_polygons,
                facecolors=cell_face,
                edgecolors=cell_face,
                linewidths=0.50 if draw_cells and module_index is not None else 0.30,
            )
        )
    if zy_polygons:
        ax_zy.add_collection(
            PolyCollection(
                zy_polygons,
                facecolors=cell_face,
                edgecolors=cell_face,
                linewidths=0.50 if draw_cells and module_index is not None else 0.30,
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
        xy_auto_bounds = _expand_bounds(*_collection_bounds(xy_segments, xy_polygons))
        xz_auto_bounds = _expand_bounds(*_collection_bounds(xz_segments, xz_polygons))
        zy_auto_bounds = _expand_bounds(*_collection_bounds(zy_segments, zy_polygons))
        x_bounds = (
            xlim
            if xlim is not None
            else (
                min(xy_auto_bounds[0], xz_auto_bounds[0]),
                max(xy_auto_bounds[1], xz_auto_bounds[1]),
            )
        )
        y_bounds = (
            ylim
            if ylim is not None
            else (
                min(xy_auto_bounds[2], zy_auto_bounds[2]),
                max(xy_auto_bounds[3], zy_auto_bounds[3]),
            )
        )
        z_bounds = zlim if zlim is not None else (zy_auto_bounds[0], zy_auto_bounds[1])
        point_mask = (
            (overlay_shower.x >= x_bounds[0])
            & (overlay_shower.x <= x_bounds[1])
            & (overlay_shower.y >= y_bounds[0])
            & (overlay_shower.y <= y_bounds[1])
            & (overlay_shower.z >= z_bounds[0])
            & (overlay_shower.z <= z_bounds[1])
        )
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
            xy_area = _scatter_area_from_data_diameter(ax_xy, fig_xy, sensor_thickness * fraction)
            xz_area = _scatter_area_from_data_diameter(ax_xz, fig_xz, sensor_thickness * fraction)
            zy_area = _scatter_area_from_data_diameter(ax_zy, fig_zy, sensor_thickness * fraction)
            xy_sizes = np.full_like(energy, xy_area, dtype=np.float64)
            xz_sizes = np.full_like(energy, xz_area, dtype=np.float64)
            zy_sizes = np.full_like(energy, zy_area, dtype=np.float64)
        else:
            xy_sizes = 1.0 + 6.0 * np.sqrt(energy / max(float(np.max(energy)), 1e-12))
            xz_sizes = xy_sizes
            zy_sizes = xy_sizes
        color_values, cmap = _overlay_color_spec(overlay_shower, point_mask)
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
    ax_xy.set_xlabel("x (mm)")
    ax_xy.set_ylabel("y (mm)")
    ax_xz.set_xlabel("x (mm)")
    ax_xz.set_ylabel("z (mm)")
    ax_zy.set_xlabel("z (mm)")
    ax_zy.set_ylabel("y (mm)")
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
        save_dpi = 500
    elif output.suffix.lower() == ".png" and draw_cells and module_index is not None:
        save_dpi = 300
    else:
        save_dpi = None
    fig_xy.savefig(output_xy, bbox_inches="tight", dpi=save_dpi)
    fig_xz.savefig(output_xz, bbox_inches="tight", dpi=save_dpi)
    fig_zy.savefig(output_zy, bbox_inches="tight", dpi=save_dpi)
    plt.close(fig_xy)
    plt.close(fig_xz)
    plt.close(fig_zy)
    return output_xy, output_xz, output_zy
