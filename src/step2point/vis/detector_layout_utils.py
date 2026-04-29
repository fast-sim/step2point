from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from step2point.core.shower import Shower
from step2point.geometry.dd4hep.factory_geometry import BarrelLayout, barrel_module_basis


@dataclass(frozen=True, slots=True)
class ProjectionSpec:
    name: str
    x_attr: str
    y_attr: str
    xlabel: str
    ylabel: str
    x_bounds_name: str
    y_bounds_name: str


PROJECTIONS: dict[str, ProjectionSpec] = {
    "xy": ProjectionSpec("XY", "x", "y", "x (mm)", "y (mm)", "x", "y"),
    "xz": ProjectionSpec("XZ", "x", "z", "x (mm)", "z (mm)", "x", "z"),
    "zy": ProjectionSpec("ZY", "z", "y", "z (mm)", "y (mm)", "z", "y"),
}


@dataclass(frozen=True, slots=True)
class WorldBounds:
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    zlim: tuple[float, float] | None = None

    def _contains_axis(self, values: np.ndarray, bounds: tuple[float, float] | None) -> np.ndarray:
        if bounds is None:
            return np.ones(values.shape, dtype=bool)
        return (values >= bounds[0]) & (values <= bounds[1])

    def point_mask(self, shower: Shower) -> np.ndarray:
        return (
            self._contains_axis(np.asarray(shower.x, dtype=np.float64), self.xlim)
            & self._contains_axis(np.asarray(shower.y, dtype=np.float64), self.ylim)
            & self._contains_axis(np.asarray(shower.z, dtype=np.float64), self.zlim)
        )

    def bounds_for_projection(self, projection: ProjectionSpec) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        return getattr(self, f"{projection.x_bounds_name}lim"), getattr(self, f"{projection.y_bounds_name}lim")

    @classmethod
    def resolved(
        cls,
        *,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        zlim: tuple[float, float] | None = None,
        fallback_x: tuple[float, float] | None = None,
        fallback_y: tuple[float, float] | None = None,
        fallback_z: tuple[float, float] | None = None,
    ) -> "WorldBounds":
        return cls(
            xlim=xlim if xlim is not None else fallback_x,
            ylim=ylim if ylim is not None else fallback_y,
            zlim=zlim if zlim is not None else fallback_z,
        )


def segment_bounds(segments: list[np.ndarray]) -> tuple[float, float, float, float]:
    points = np.concatenate(segments, axis=0)
    return (
        float(np.min(points[:, 0])),
        float(np.max(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 1])),
    )


def polygon_bounds(polygons: list[np.ndarray]) -> tuple[float, float, float, float]:
    points = np.concatenate(polygons, axis=0)
    return (
        float(np.min(points[:, 0])),
        float(np.max(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 1])),
    )


def collection_bounds(
    segments: list[np.ndarray],
    polygons: list[np.ndarray],
) -> tuple[float, float, float, float]:
    if segments:
        return segment_bounds(segments)
    if polygons:
        return polygon_bounds(polygons)
    raise ValueError("No drawable geometry available to determine plot bounds.")


def expand_bounds(
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


def scatter_area_from_data_diameter(
    ax: plt.Axes,
    fig: plt.Figure,
    data_diameter_x: float,
) -> float:
    p0 = ax.transData.transform((0.0, 0.0))
    p1 = ax.transData.transform((data_diameter_x, 0.0))
    diameter_px = max(float(abs(p1[0] - p0[0])), 1.0)
    diameter_pt = diameter_px * 72.0 / fig.dpi
    return diameter_pt * diameter_pt


def polygon_intersects_projection_bounds(
    polygon: np.ndarray,
    bounds: WorldBounds,
    projection: ProjectionSpec,
) -> bool:
    x_bounds, y_bounds = bounds.bounds_for_projection(projection)
    xmin = float(np.min(polygon[:, 0]))
    xmax = float(np.max(polygon[:, 0]))
    ymin = float(np.min(polygon[:, 1]))
    ymax = float(np.max(polygon[:, 1]))
    if x_bounds is not None and (xmax <= x_bounds[0] or xmin >= x_bounds[1]):
        return False
    if y_bounds is not None and (ymax <= y_bounds[0] or ymin >= y_bounds[1]):
        return False
    return True


def segment_intersects_projection_bounds(
    segment: np.ndarray,
    bounds: WorldBounds,
    projection: ProjectionSpec,
) -> bool:
    x_bounds, y_bounds = bounds.bounds_for_projection(projection)
    xmin = float(np.min(segment[:, 0]))
    xmax = float(np.max(segment[:, 0]))
    ymin = float(np.min(segment[:, 1]))
    ymax = float(np.max(segment[:, 1]))
    if x_bounds is not None and (xmax <= x_bounds[0] or xmin >= x_bounds[1]):
        return False
    if y_bounds is not None and (ymax <= y_bounds[0] or ymin >= y_bounds[1]):
        return False
    return True


def filter_geometry_to_bounds(
    segments: list[np.ndarray],
    polygons: list[np.ndarray],
    bounds: WorldBounds,
    projection: ProjectionSpec,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    filtered_segments = [segment for segment in segments if segment_intersects_projection_bounds(segment, bounds, projection)]
    filtered_polygons = [polygon for polygon in polygons if polygon_intersects_projection_bounds(polygon, bounds, projection)]
    return filtered_segments, filtered_polygons


def z_bins_intersect_limits(layer, zlim: tuple[float, float] | None) -> bool:
    if zlim is None:
        return True
    max_z_index = int(np.ceil(layer.half_z_mm / layer.pitch_z_mm - 0.5))
    z_centers = np.arange(-max_z_index, max_z_index + 1, dtype=np.int32) * layer.pitch_z_mm
    return bool(np.any((z_centers >= zlim[0]) & (z_centers <= zlim[1])))


def x_bins_intersect_limits(
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


def layer_intersects_ylim(
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


def cluster_label_cmap() -> ListedColormap:
    colors = list(plt.get_cmap("tab20").colors)
    colors.extend(plt.get_cmap("Dark2").colors)
    return ListedColormap(colors[:28], name="step2point_cluster_labels")


def overlay_color_spec(shower: Shower, point_mask: np.ndarray) -> tuple[np.ndarray, ListedColormap | str | None]:
    cluster_label = shower.metadata.get("cluster_label")
    if cluster_label is not None:
        labels = np.asarray(cluster_label, dtype=np.int64)
        unique_labels, inverse = np.unique(labels[point_mask], return_inverse=True)
        if unique_labels.size:
            return inverse.astype(np.float64), cluster_label_cmap()
    energy = np.asarray(shower.E, dtype=np.float64)
    return np.log10(np.clip(energy[point_mask], 1e-12, None)), "inferno"
