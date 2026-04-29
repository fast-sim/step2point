from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize

from step2point.metrics.spatial import estimate_shower_axis, longitudinal_radial_phi


def _prepare_outpath(outpath: str | Path) -> Path:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    return outpath


def _get_color_data(shower, color_by: str):
    if color_by == "energy":
        values = np.log10(np.clip(np.asarray(shower.E, dtype=np.float64), 1e-12, None))
        return (
            values,
            "inferno",
            Normalize(vmin=np.percentile(values, 1), vmax=np.percentile(values, 99)),
            "log10(E [GeV])",
        )
    if color_by == "pdg":
        if shower.pdg is None:
            raise ValueError("PDG colouring requires shower.pdg.")
        return _discrete_colors(shower.pdg, "tab20", "PDG")
    if color_by == "subdetector":
        subdetector = shower.metadata.get("subdetector")
        if subdetector is None:
            raise ValueError("Subdetector colouring requires shower.metadata['subdetector'].")
        return _discrete_colors(subdetector, "tab10", "Subdetector")
    raise ValueError(f"Unsupported colour mode: {color_by}")


def _discrete_colors(values, cmap_name: str, label: str):
    values = np.asarray(values)
    unique_vals = np.unique(values)
    value_to_index = {value: index for index, value in enumerate(unique_vals)}
    indices = np.array([value_to_index[value] for value in values], dtype=np.int32)
    cmap = ListedColormap(plt.get_cmap(cmap_name)(np.linspace(0.0, 1.0, len(unique_vals))))
    norm = BoundaryNorm(np.arange(-0.5, len(unique_vals) + 0.5), cmap.N)
    return indices, cmap, norm, label


def _auto_limits(values: np.ndarray, pad: float = 0.1) -> tuple[float, float]:
    q1, q2 = np.percentile(values, [1, 99])
    span = max(q2 - q1, 1.0)
    return float(q1 - pad * span), float(q2 + pad * span)


def _upper_percentile_limit(values: np.ndarray, percentile: float = 99.0, pad: float = 0.05) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 1.0
    upper = float(np.percentile(values, percentile))
    return max(upper * (1.0 + pad), 1.0)


def _zoomed_bins(values: np.ndarray, nbins: int = 20, percentile: float = 99.0, pad: float = 0.05) -> np.ndarray:
    upper = _upper_percentile_limit(values, percentile=percentile, pad=pad)
    return np.linspace(0.0, upper, nbins + 1)


def scatter_xz(shower, outpath):
    outpath = _prepare_outpath(outpath)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(shower.x, shower.z, s=4)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_shower_projections(shower, outpath, *, color_by: str = "energy"):
    outpath = _prepare_outpath(outpath)
    color, cmap, norm, label = _get_color_data(shower, color_by)
    coords = np.stack([shower.x, shower.y, shower.z], axis=1)
    xlim = _auto_limits(coords[:, 0])
    ylim = _auto_limits(coords[:, 1])
    zlim = _auto_limits(coords[:, 2])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    projections = (
        (coords[:, 0], coords[:, 1], xlim, ylim, "XY", "x (mm)", "y (mm)"),
        (coords[:, 0], coords[:, 2], xlim, zlim, "XZ", "x (mm)", "z (mm)"),
        (coords[:, 1], coords[:, 2], ylim, zlim, "YZ", "y (mm)", "z (mm)"),
    )
    for ax, (u, v, ulim, vlim, title, xlabel, ylabel) in zip(axes, projections, strict=True):
        scatter = ax.scatter(u, v, c=color, cmap=cmap, norm=norm, s=6, edgecolors="none")
        ax.set_xlim(*ulim)
        ax.set_ylim(*vlim)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.colorbar(scatter, ax=axes, shrink=0.8, label=label)
    fig.savefig(outpath)
    plt.close(fig)


def plot_shower_distributions(shower, outpath):
    outpath = _prepare_outpath(outpath)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fill_color = "#FF7A00"
    fill_alpha = 0.5
    centroid, axis = estimate_shower_axis(shower)
    long, radial, _ = longitudinal_radial_phi(
        shower,
        centroid=centroid,
        axis=axis,
        longitudinal_origin="first_deposit",
    )
    log_energy = np.log10(np.clip(np.asarray(shower.E, dtype=np.float64), 1e-12, None))
    axes[0, 0].hist(log_energy, bins=50, color=fill_color, alpha=fill_alpha, log=True)
    axes[0, 0].set_xlabel("log10(energy [GeV])")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Energy distribution")
    if shower.t is not None:
        axes[0, 1].hist(shower.t, bins=50, color=fill_color, alpha=fill_alpha, log=True)
        axes[0, 1].set_xlabel("time [ns]")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("Time distribution")
        axes[0, 1].set_xlim(left=0.0, right=float(np.percentile(shower.t, 95)))
    else:
        axes[0, 1].text(0.5, 0.5, "No time information", ha="center", va="center", transform=axes[0, 1].transAxes)
        axes[0, 1].set_axis_off()
    long_bins = _zoomed_bins(long, nbins=20)
    axes[1, 0].hist(long, bins=long_bins, weights=shower.E, color=fill_color, alpha=fill_alpha)
    axes[1, 0].set_xlabel("longitudinal (from first deposit) [mm]")
    axes[1, 0].set_ylabel("Deposited energy per bin [GeV]")
    axes[1, 0].set_title("Longitudinal profile")
    axes[1, 0].set_xlim(0.0, long_bins[-1])
    axes[1, 1].hist(radial, bins=40, weights=shower.E, color=fill_color, alpha=fill_alpha)
    axes[1, 1].set_xlabel("radial [mm]")
    axes[1, 1].set_ylabel("Deposited energy per bin [GeV]")
    axes[1, 1].set_title("Radial profile")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_shower_overview(shower, outpath, *, axis_override=None):
    """Plot an overview of one shower.

    The shower axis is PCA-derived by default. `axis_override` is optional and
    only intended for manual direction studies.
    """
    outpath = _prepare_outpath(outpath)
    centroid, axis = estimate_shower_axis(shower, axis_override=axis_override)
    coords = np.stack([shower.x, shower.y, shower.z], axis=1) - centroid
    long, radial, phi = longitudinal_radial_phi(
        shower,
        centroid=centroid,
        axis=axis,
        longitudinal_origin="first_deposit",
    )
    long_bins = _zoomed_bins(long, nbins=20)
    axis_line = np.vstack([-100.0 * axis, 100.0 * axis])
    xlim = _auto_limits(coords[:, 0])
    ylim = _auto_limits(coords[:, 1])
    zlim = _auto_limits(coords[:, 2])
    fig = plt.figure(figsize=(14, 15), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    ax_3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xz = fig.add_subplot(gs[1, 0])
    ax_yz = fig.add_subplot(gs[1, 1])
    ax_lr = fig.add_subplot(gs[2, 0])
    ax_polar = fig.add_subplot(gs[2, 1], projection="polar")
    colours = np.asarray(shower.E, dtype=np.float64)
    scatter3d = ax_3d.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colours, cmap="viridis", s=8)
    ax_3d.plot(axis_line[:, 0], axis_line[:, 1], axis_line[:, 2], color="red", linewidth=2)
    ax_3d.set_xlim(*xlim)
    ax_3d.set_ylim(*ylim)
    ax_3d.set_zlim(*zlim)
    for ax, first, second, lim1, lim2, title in (
        (ax_xy, 0, 1, xlim, ylim, "XY projection"),
        (ax_xz, 0, 2, xlim, zlim, "XZ projection"),
        (ax_yz, 1, 2, ylim, zlim, "YZ projection"),
    ):
        ax.scatter(coords[:, first], coords[:, second], c=colours, cmap="viridis", s=8)
        ax.plot(axis_line[:, first], axis_line[:, second], color="red", linewidth=2)
        ax.set_xlim(*lim1)
        ax.set_ylim(*lim2)
        ax.set_title(title)
    ax_xy.set_xlabel("x (mm)")
    ax_xy.set_ylabel("y (mm)")
    ax_xz.set_xlabel("x (mm)")
    ax_xz.set_ylabel("z (mm)")
    ax_yz.set_xlabel("y (mm)")
    ax_yz.set_ylabel("z (mm)")
    ax_lr.scatter(long, radial, c=colours, cmap="plasma", s=8)
    ax_lr.set_xlabel("Longitudinal (from first deposit) [mm]")
    ax_lr.set_ylabel("Radius [mm]")
    ax_lr.set_title("Cylindrical: long vs r")
    ax_lr.set_xlim(0.0, long_bins[-1])
    ax_polar.scatter(phi, radial, c=colours, cmap="plasma", s=8)
    ax_polar.set_title("Polar: r vs phi")
    fig.colorbar(scatter3d, ax=[ax_3d, ax_xy, ax_xz, ax_yz], shrink=0.7, label="Energy")
    fig.savefig(outpath)
    plt.close(fig)


def _display_axis(shower, axis_override=None) -> np.ndarray:
    if axis_override is not None:
        axis = np.asarray(axis_override, dtype=np.float64)
    else:
        momentum = shower.primary.get("momentum")
        axis = np.asarray(momentum, dtype=np.float64) if momentum is not None else None
    if axis is None or axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= 0.0:
        _, axis = estimate_shower_axis(shower)
    return axis / np.linalg.norm(axis)


def _particle_label_from_pdg(pdg: int) -> str:
    labels = {
        11: "e-",
        -11: "e+",
        13: "mu-",
        -13: "mu+",
        22: r"$\gamma$",
        111: "pi0",
        211: "pi+",
        -211: "pi-",
        321: "K+",
        -321: "K-",
        2112: "n",
        2212: "p",
    }
    return labels.get(int(pdg), f"PDG {int(pdg)}")


def _mass_gev_from_pdg(pdg: int) -> float:
    masses = {
        11: 0.00051099895,
        13: 0.1056583755,
        22: 0.0,
        111: 0.1349768,
        211: 0.13957039,
        321: 0.493677,
        2112: 0.93956542,
        2212: 0.93827209,
    }
    return masses.get(abs(int(pdg)), 0.0)


def _format_energy_gev(energy: float) -> str:
    if not np.isfinite(energy):
        return ""
    return f"{energy:.0f} GeV"


def _default_incident_label(shower) -> str:
    primary = shower.primary or {}
    pdg = primary.get("pdg")
    particle = _particle_label_from_pdg(int(pdg)) if pdg is not None else "incident particle"
    energy = None
    momentum = primary.get("momentum")
    if momentum is not None:
        mom = np.asarray(momentum, dtype=np.float64)
        if mom.shape == (3,) and np.isfinite(mom).all():
            mass = _mass_gev_from_pdg(int(pdg)) if pdg is not None else 0.0
            energy = float(np.sqrt(float(np.dot(mom, mom)) + mass * mass))
    energy_label = _format_energy_gev(float(energy)) if energy is not None else ""
    return f"{energy_label} {particle}".strip()


def _primary_energy_line(primary: dict) -> str:
    momentum = np.asarray(primary.get("momentum", (np.nan, np.nan, np.nan)), dtype=np.float64)
    return f"Energy: {_format_energy_gev(np.linalg.norm(momentum))}"


def _display_origin(shower) -> np.ndarray | None:
    primary = shower.primary or {}
    vertex = primary.get("vertex")
    if vertex is None:
        return None
    origin = np.asarray(vertex, dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        return None
    return origin


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.ndim != 1 or values.size != weights.size:
        raise ValueError("values and weights must be 1D arrays of equal length")
    if values.size == 0:
        raise ValueError("values must be non-empty")
    if not (0.0 <= quantile <= 1.0):
        raise ValueError("quantile must satisfy 0 <= quantile <= 1")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = np.clip(weights[order], 0.0, None)
    total_weight = float(np.sum(sorted_weights))
    if total_weight <= 0.0:
        return float(np.quantile(sorted_values, quantile))
    cumulative = np.cumsum(sorted_weights) / total_weight
    index = int(np.searchsorted(cumulative, quantile, side="left"))
    index = min(index, len(sorted_values) - 1)
    return float(sorted_values[index])


def _equal_3d_limits(ax, coords: np.ndarray, pad_fraction: float = 0.16) -> None:
    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)
    center = 0.5 * (mins + maxs)
    span = max(float(np.max(maxs - mins)), 1.0)
    half = 0.5 * span * (1.0 + pad_fraction)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def _crop_coordinates(
    coords: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    axis: np.ndarray | None = None,
    origin: np.ndarray | None = None,
    energy_containment: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
) -> np.ndarray:
    cropped = coords
    if energy_containment is not None:
        if weights is None or axis is None or origin is None:
            raise ValueError("weights, axis, and origin are required for energy-containment cropping.")
        if not (0.0 < energy_containment <= 100.0):
            raise ValueError("energy_containment must satisfy 0 < VALUE <= 100.")
        weights = np.asarray(weights, dtype=np.float64)
        rel = coords - origin
        longitudinal = rel @ axis
        radial_vec = rel - np.outer(longitudinal, axis)
        radial = np.linalg.norm(radial_vec, axis=1)
        order = np.argsort(radial, kind="stable")
        cumulative_energy = np.cumsum(weights[order])
        total_energy = float(cumulative_energy[-1]) if cumulative_energy.size else 0.0
        if total_energy > 0.0:
            threshold = total_energy * energy_containment / 100.0
            cutoff_index = min(int(np.searchsorted(cumulative_energy, threshold, side="left")), len(order) - 1)
            radius = float(radial[order[cutoff_index]])
            mask = radial <= radius
        else:
            mask = np.ones(len(coords), dtype=bool)
        if np.count_nonzero(mask) >= 2:
            cropped = cropped[mask]
    for axis_index, limits in enumerate((xlim, ylim, zlim)):
        if limits is None:
            continue
        lo, hi = limits
        if lo >= hi:
            raise ValueError("Manual display limits must satisfy MIN < MAX.")
        mask = (cropped[:, axis_index] >= lo) & (cropped[:, axis_index] <= hi)
        if np.count_nonzero(mask) >= 2:
            cropped = cropped[mask]
    return cropped


def _trim_near_white_png(path: Path, *, background_tolerance: float = 0.025, pad_px: int = 8) -> None:
    image = plt.imread(path)
    if image.ndim != 3 or image.shape[2] < 3:
        return
    rgb = image[:, :, :3]
    alpha = image[:, :, 3] if image.shape[2] >= 4 else np.ones(image.shape[:2], dtype=np.float64)
    border_width = min(8, image.shape[0] // 4, image.shape[1] // 4)
    if border_width <= 0:
        return
    border_pixels = np.concatenate(
        [
            rgb[:border_width, :, :].reshape(-1, 3),
            rgb[-border_width:, :, :].reshape(-1, 3),
            rgb[:, :border_width, :].reshape(-1, 3),
            rgb[:, -border_width:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(border_pixels, axis=0)
    differs_from_background = np.max(np.abs(rgb - background), axis=2) > background_tolerance
    content = differs_from_background & (alpha > 0.01)
    if not np.any(content):
        return
    rows, cols = np.where(content)
    y0 = max(int(np.min(rows)) - pad_px, 0)
    y1 = min(int(np.max(rows)) + pad_px + 1, image.shape[0])
    x0 = max(int(np.min(cols)) - pad_px, 0)
    x1 = min(int(np.max(cols)) + pad_px + 1, image.shape[1])
    cropped = image[y0:y1, x0:x1]
    plt.imsave(path, cropped)


def _energy_marker_sizes(energy: np.ndarray, *, min_size: float, max_size: float) -> np.ndarray:
    energy = np.asarray(energy, dtype=np.float64)
    if energy.size == 0:
        return energy
    scaled = np.sqrt(energy / max(float(np.max(energy)), 1e-12))
    return min_size + (max_size - min_size) * scaled


def _energy_marker_sizes_shared(
    energy: np.ndarray,
    *,
    energy_max: float,
    min_size: float,
    max_size: float,
) -> np.ndarray:
    energy = np.asarray(energy, dtype=np.float64)
    if energy.size == 0:
        return energy
    scaled = np.sqrt(energy / max(float(energy_max), 1e-12))
    return min_size + (max_size - min_size) * scaled


def _set_display_style(fig, ax) -> None:
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_axis_off()
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))


def _display_geometry_for_showers(
    showers,
    *,
    axis_override=None,
    crop_percentile: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    non_empty_coords = [np.stack([s.x, s.y, s.z], axis=1) for s in showers if s.n_points > 0]
    if not non_empty_coords:
        raise ValueError("At least one shower must contain points.")
    all_coords = np.concatenate(non_empty_coords, axis=0)
    reference = showers[0]
    axis = _display_axis(reference, axis_override=axis_override)
    all_energy = np.concatenate([np.asarray(s.E, dtype=np.float64) for s in showers if s.n_points > 0])
    origin = _display_origin(reference)
    if origin is None:
        origin = np.average(all_coords, axis=0, weights=all_energy)
    projected = (all_coords - origin) @ axis
    display_coords = _crop_coordinates(
        all_coords,
        weights=all_energy,
        axis=axis,
        origin=origin,
        energy_containment=crop_percentile,
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
    )
    return all_coords, all_energy, origin, axis, projected, display_coords


def _incident_line_points(
    origin: np.ndarray,
    axis: np.ndarray,
    projected: np.ndarray,
    energy: np.ndarray,
    *,
    line_length: float = 90.0,
    gap: float = 10.0,
    front_quantile: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    front_long = _weighted_quantile(projected, energy, front_quantile)
    shower_front = origin + front_long * axis
    line_start = shower_front - (gap + line_length) * axis
    line_end = shower_front - gap * axis
    return line_start, line_end


def _draw_incident_label(fig, ax, line_start: np.ndarray, line_end: np.ndarray, text: str) -> None:
    fig.canvas.draw()
    projected_line = ax.get_proj() @ np.array(
        [
            [line_start[0], line_start[1], line_start[2], 1.0],
            [line_end[0], line_end[1], line_end[2], 1.0],
        ],
        dtype=np.float64,
    ).T
    projected_line = projected_line[:2] / projected_line[3]
    display_line = ax.transData.transform(projected_line.T)
    label_delta = display_line[1] - display_line[0]
    label_norm = np.linalg.norm(label_delta)
    if np.isfinite(label_norm) and label_norm > 0.0:
        normal = np.array([-label_delta[1], label_delta[0]], dtype=np.float64) / label_norm
    else:
        normal = np.array([0.0, 1.0], dtype=np.float64)
    label_position = fig.transFigure.inverted().transform(np.mean(display_line, axis=0) + 15.0 * normal)
    label_position = np.clip(label_position, 0.02, 0.98)
    label_angle = float(np.degrees(np.arctan2(label_delta[1], label_delta[0])))
    fig.text(
        label_position[0],
        label_position[1],
        text,
        color="#666666",
        fontsize=7,
        alpha=0.78,
        rotation=label_angle,
        rotation_mode="anchor",
        ha="center",
        va="center",
    )


def render_shower_display_3d(
    showers,
    outpath,
    *,
    axis_override=None,
    incident_label: str | None = None,
    view: tuple[float, float] = (20.0, -58.0),
    figsize: tuple[float, float] = (16.0, 12.0),
    dpi: int = 240,
    crop_percentile: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
) -> Path:
    """Render a presentation-oriented 3D shower display.

    This is intentionally styled as an illustration rather than an analysis
    plot: axes, ticks, and grids are hidden; energy is shown with glow-like
    translucent point layers; and the incident direction is drawn explicitly.
    """
    outpath = _prepare_outpath(outpath)
    showers = list(showers)
    if not showers:
        raise ValueError("At least one shower is required.")

    all_coords, all_energy, origin, axis, projected, display_coords = _display_geometry_for_showers(
        showers,
        axis_override=axis_override,
        crop_percentile=crop_percentile,
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
    )
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    _set_display_style(fig, ax)
    ax.set_position((0.0, 0.0, 1.0, 1.0))

    palettes = [
        LinearSegmentedColormap.from_list(
            "display_single",
            ["#fff3d6", "#ffd166", "#ff8c42", "#ef476f", "#7b2cbf"],
        ),
        plt.get_cmap("turbo"),
        LinearSegmentedColormap.from_list("display_blue", ["#e3f2ff", "#6ec5ff", "#4361ee", "#3a0ca3"]),
    ]
    edge_colours = ["#fff6de", "#fff7d6", "#e6ecff"]

    for index, shower in enumerate(showers):
        if shower.n_points == 0:
            continue
        coords = np.stack([shower.x, shower.y, shower.z], axis=1)
        energy = np.asarray(shower.E, dtype=np.float64)
        colours = np.log10(np.clip(energy, 1e-12, None))
        norm = Normalize(vmin=float(np.percentile(colours, 4)), vmax=float(np.percentile(colours, 99)))
        cmap = palettes[index % len(palettes)]
        if len(showers) == 1:
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=colours,
                cmap=cmap,
                norm=norm,
                s=_energy_marker_sizes(energy, min_size=10.0, max_size=42.0),
                alpha=0.88,
                edgecolors="#fff7e8",
                linewidths=0.18,
                depthshade=False,
            )
        elif index == 0:
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=colours,
                cmap=cmap,
                norm=norm,
                s=_energy_marker_sizes(energy, min_size=4.0, max_size=24.0),
                alpha=0.42,
                linewidths=0.0,
                depthshade=False,
            )
        else:
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=colours,
                cmap=cmap,
                norm=norm,
                s=_energy_marker_sizes(energy, min_size=28.0, max_size=120.0),
                alpha=0.96,
                edgecolors=edge_colours[index % len(edge_colours)],
                linewidths=0.45,
                depthshade=False,
            )

    reference = showers[0]
    line_start, line_end = _incident_line_points(origin, axis, projected, all_energy)
    ax.plot(
        [line_start[0], line_end[0]],
        [line_start[1], line_end[1]],
        [line_start[2], line_end[2]],
        color="#7a7a7a",
        linewidth=0.82,
        linestyle=(0, (1.7, 2.0)),
        alpha=0.72,
    )

    _equal_3d_limits(ax, display_coords, pad_fraction=0.05)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if zlim is not None:
        ax.set_zlim(*zlim)
    ax.view_init(elev=view[0], azim=view[1])
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:  # pragma: no cover - older matplotlib fallback
        pass
    _draw_incident_label(
        fig,
        ax,
        line_start,
        line_end,
        incident_label if incident_label is not None else _default_incident_label(reference),
    )
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.savefig(outpath, facecolor=fig.get_facecolor(), pad_inches=0.0)
    plt.close(fig)
    if outpath.suffix.lower() == ".png":
        _trim_near_white_png(outpath)
    return outpath


def render_shower_display_triptych_3d(
    showers,
    outpath,
    *,
    panel_titles: list[str] | tuple[str, str, str] | None = None,
    panel_subtitles: list[str] | tuple[str, str, str] | None = None,
    axis_override=None,
    incident_label: str | None = None,
    view: tuple[float, float] = (20.0, -58.0),
    figsize: tuple[float, float] = (20.0, 10.5),
    dpi: int = 240,
    crop_percentile: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
) -> Path:
    outpath = _prepare_outpath(outpath)
    showers = list(showers)
    if len(showers) != 3:
        raise ValueError("Triptych display expects exactly three showers.")

    all_coords, all_energy, origin, axis, projected, display_coords = _display_geometry_for_showers(
        showers,
        axis_override=axis_override,
        crop_percentile=crop_percentile,
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
    )
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, height_ratios=[20.0, 4.5], hspace=0.0, wspace=0.02)
    axes = [fig.add_subplot(gs[0, i], projection="3d") for i in range(3)]

    colour_values = np.log10(np.clip(all_energy, 1e-12, None))
    global_norm = Normalize(
        vmin=float(np.percentile(colour_values, 2)),
        vmax=float(np.percentile(colour_values, 99.5)),
    )
    palette = LinearSegmentedColormap.from_list(
        "display_triptych",
        ["#fff4db", "#ffd166", "#ff8c42", "#ff4d1a", "#d81b60", "#7b2cbf"],
    )
    energy_max = float(np.max(all_energy))
    line_start, line_end = _incident_line_points(origin, axis, projected, all_energy)
    counts = [shower.n_points for shower in showers]
    if panel_titles is None:
        panel_titles = ["Detailed Shower", "Intermediate Representation", "Compressed Shower"]
    if panel_subtitles is None:
        panel_subtitles = []
        base = max(counts[0], 1)
        for count in counts:
            if count == counts[0]:
                panel_subtitles.append(f"{count:,} points")
            else:
                panel_subtitles.append(f"{count:,} points   ({base / max(count, 1):.1f}x reduction)")

    for idx, (ax, shower) in enumerate(zip(axes, showers, strict=True)):
        _set_display_style(fig, ax)
        ax.set_position(ax.get_position())
        coords = np.stack([shower.x, shower.y, shower.z], axis=1)
        energy = np.asarray(shower.E, dtype=np.float64)
        colours = np.log10(np.clip(energy, 1e-12, None))
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=colours,
            cmap=palette,
            norm=global_norm,
            s=_energy_marker_sizes_shared(energy, energy_max=energy_max, min_size=3.0, max_size=340.0),
            alpha=0.9,
            edgecolors="#fff8ef",
            linewidths=0.15 if idx == 0 else 0.2,
            depthshade=False,
        )
        ax.plot(
            [line_start[0], line_end[0]],
            [line_start[1], line_end[1]],
            [line_start[2], line_end[2]],
            color="#7a7a7a",
            linewidth=0.9,
            linestyle=(0, (1.7, 2.0)),
            alpha=0.72,
        )
        _equal_3d_limits(ax, display_coords, pad_fraction=0.02)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if zlim is not None:
            ax.set_zlim(*zlim)
        ax.view_init(elev=view[0], azim=view[1])
        try:
            ax.set_box_aspect((1.0, 1.0, 1.0))
        except AttributeError:  # pragma: no cover
            pass

    for ax, title, subtitle in zip(axes, panel_titles, panel_subtitles, strict=True):
        bbox = ax.get_position()
        fig.text(
            0.5 * (bbox.x0 + bbox.x1),
            bbox.y1 + 0.02,
            title.upper(),
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
            color="#222222",
        )
        fig.text(
            0.5 * (bbox.x0 + bbox.x1),
            bbox.y1 - 0.006,
            subtitle,
            ha="center",
            va="bottom",
            fontsize=11,
            color="#555555",
        )
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")
    legend_ax.set_facecolor("white")

    from matplotlib.patches import FancyBboxPatch

    left_box = FancyBboxPatch(
        (0.02, 0.16),
        0.56,
        0.68,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=1.2,
        edgecolor="#d0d0d0",
        facecolor="white",
        transform=legend_ax.transAxes,
    )
    right_box = FancyBboxPatch(
        (0.79, 0.20),
        0.18,
        0.60,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=1.2,
        edgecolor="#d0d0d0",
        facecolor="white",
        transform=legend_ax.transAxes,
    )
    legend_ax.add_patch(left_box)
    legend_ax.add_patch(right_box)

    legend_ax.text(
        0.06,
        0.68,
        "Color = deposited energy",
        transform=legend_ax.transAxes,
        fontsize=10,
        color="#333333",
        weight="bold",
    )
    gradient_ax = legend_ax.inset_axes([0.06, 0.38, 0.22, 0.16])
    gradient = np.linspace(0.0, 1.0, 256).reshape(1, -1)
    gradient_ax.imshow(gradient, aspect="auto", cmap=palette, origin="lower")
    gradient_ax.set_axis_off()
    legend_ax.text(0.06, 0.28, "low", transform=legend_ax.transAxes, fontsize=9, color="#666666")
    legend_ax.text(0.26, 0.28, "high", transform=legend_ax.transAxes, fontsize=9, color="#666666", ha="right")

    legend_ax.text(
        0.34,
        0.68,
        "Size = deposited energy",
        transform=legend_ax.transAxes,
        fontsize=10,
        color="#333333",
        weight="bold",
    )
    size_ax = legend_ax.inset_axes([0.34, 0.30, 0.18, 0.28])
    size_ax.axis("off")
    size_energies = np.linspace(0.05, 1.0, 5) * energy_max
    size_values = _energy_marker_sizes_shared(size_energies, energy_max=energy_max, min_size=3.0, max_size=340.0)
    xs = np.linspace(0.12, 0.88, len(size_values))
    size_ax.scatter(xs, np.full_like(xs, 0.5), s=size_values, c="#ff9f1c", edgecolors="#fff5eb", linewidths=0.2)
    size_ax.set_xlim(0.0, 1.0)
    size_ax.set_ylim(0.0, 1.0)
    legend_ax.text(0.34, 0.28, "low", transform=legend_ax.transAxes, fontsize=9, color="#666666")
    legend_ax.text(0.52, 0.28, "high", transform=legend_ax.transAxes, fontsize=9, color="#666666", ha="right")

    primary = showers[0].primary or {}
    info_lines = [
        "Shower information",
        f"Incident particle: {_particle_label_from_pdg(int(primary.get('pdg', 0)))}",
        _primary_energy_line(primary),
        "Detector: Open Data Detector",
    ]
    y = 0.67
    for i, line in enumerate(info_lines):
        legend_ax.text(
            0.80,
            y,
            line,
            transform=legend_ax.transAxes,
            fontsize=10 if i else 11,
            color="#333333",
            weight="bold" if i == 0 else None,
        )
        y -= 0.16 if i == 0 else 0.15

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.15)
    fig.savefig(outpath, facecolor=fig.get_facecolor(), pad_inches=0.02)
    plt.close(fig)
    if outpath.suffix.lower() == ".png":
        _trim_near_white_png(outpath)
    return outpath


def render_shower_display_comparison_3d(
    showers,
    outpath,
    *,
    panel_titles: list[str] | tuple[str, str] | None = None,
    panel_subtitles: list[str] | tuple[str, str] | None = None,
    axis_override=None,
    incident_label: str | None = None,
    view: tuple[float, float] = (20.0, -58.0),
    figsize: tuple[float, float] = (14.5, 10.5),
    dpi: int = 240,
    crop_percentile: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
) -> Path:
    outpath = _prepare_outpath(outpath)
    showers = list(showers)
    if len(showers) != 2:
        raise ValueError("Comparison display expects exactly two showers.")

    all_coords, all_energy, origin, axis, projected, display_coords = _display_geometry_for_showers(
        showers,
        axis_override=axis_override,
        crop_percentile=crop_percentile,
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
    )
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, height_ratios=[20.0, 4.8], hspace=0.0, wspace=0.03)
    axes = [fig.add_subplot(gs[0, i], projection="3d") for i in range(2)]

    colour_values = np.log10(np.clip(all_energy, 1e-12, None))
    global_norm = Normalize(
        vmin=float(np.percentile(colour_values, 2)),
        vmax=float(np.percentile(colour_values, 99.5)),
    )
    palette = LinearSegmentedColormap.from_list(
        "display_comparison",
        ["#fff4db", "#ffd166", "#ff8c42", "#ff4d1a", "#d81b60", "#7b2cbf"],
    )
    energy_max = float(np.max(all_energy))
    line_start, line_end = _incident_line_points(origin, axis, projected, all_energy)

    counts = [shower.n_points for shower in showers]
    if panel_titles is None:
        panel_titles = ["Detailed Shower", "Compressed Shower"]
    if panel_subtitles is None:
        panel_subtitles = []
        base = max(counts[0], 1)
        for count in counts:
            if count == counts[0]:
                panel_subtitles.append(f"{count:,} points")
            else:
                panel_subtitles.append(f"{count:,} points   ({base / max(count, 1):.1f}x reduction)")

    for ax, shower in zip(axes, showers, strict=True):
        _set_display_style(fig, ax)
        coords = np.stack([shower.x, shower.y, shower.z], axis=1)
        energy = np.asarray(shower.E, dtype=np.float64)
        colours = np.log10(np.clip(energy, 1e-12, None))
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=colours,
            cmap=palette,
            norm=global_norm,
            s=_energy_marker_sizes_shared(energy, energy_max=energy_max, min_size=3.0, max_size=340.0),
            alpha=0.9,
            edgecolors="#fff8ef",
            linewidths=0.18,
            depthshade=False,
        )
        ax.plot(
            [line_start[0], line_end[0]],
            [line_start[1], line_end[1]],
            [line_start[2], line_end[2]],
            color="#7a7a7a",
            linewidth=0.9,
            linestyle=(0, (1.7, 2.0)),
            alpha=0.72,
        )
        _equal_3d_limits(ax, display_coords, pad_fraction=0.02)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if zlim is not None:
            ax.set_zlim(*zlim)
        ax.view_init(elev=view[0], azim=view[1])
        try:
            ax.set_box_aspect((1.0, 1.0, 1.0))
        except AttributeError:  # pragma: no cover
            pass

    for ax, title, subtitle in zip(axes, panel_titles, panel_subtitles, strict=True):
        bbox = ax.get_position()
        fig.text(
            0.5 * (bbox.x0 + bbox.x1),
            bbox.y1 + 0.02,
            title.upper(),
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
            color="#222222",
        )
        fig.text(
            0.5 * (bbox.x0 + bbox.x1),
            bbox.y1 - 0.006,
            subtitle,
            ha="center",
            va="bottom",
            fontsize=11,
            color="#555555",
        )

    footer_ax = fig.add_subplot(gs[1, :])
    footer_ax.axis("off")
    footer_ax.set_facecolor("white")

    from matplotlib.patches import FancyBboxPatch

    footer_box = FancyBboxPatch(
        (0.03, 0.14),
        0.62,
        0.70,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=1.2,
        edgecolor="#d0d0d0",
        facecolor="white",
        transform=footer_ax.transAxes,
    )
    info_box = FancyBboxPatch(
        (0.79, 0.20),
        0.18,
        0.60,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=1.2,
        edgecolor="#d0d0d0",
        facecolor="white",
        transform=footer_ax.transAxes,
    )
    footer_ax.add_patch(footer_box)
    footer_ax.add_patch(info_box)

    footer_ax.text(
        0.06,
        0.64,
        "Color = deposited energy",
        transform=footer_ax.transAxes,
        fontsize=10,
        color="#333333",
        weight="bold",
    )
    gradient_ax = footer_ax.inset_axes([0.06, 0.36, 0.24, 0.16])
    gradient = np.linspace(0.0, 1.0, 256).reshape(1, -1)
    gradient_ax.imshow(gradient, aspect="auto", cmap=palette, origin="lower")
    gradient_ax.set_axis_off()
    footer_ax.text(0.06, 0.26, "low", transform=footer_ax.transAxes, fontsize=9, color="#666666")
    footer_ax.text(0.30, 0.26, "high", transform=footer_ax.transAxes, fontsize=9, color="#666666", ha="right")

    footer_ax.text(
        0.34,
        0.68,
        "Size = deposited energy",
        transform=footer_ax.transAxes,
        fontsize=10,
        color="#333333",
        weight="bold",
    )
    size_ax = footer_ax.inset_axes([0.34, 0.30, 0.18, 0.28])
    size_ax.axis("off")
    size_energies = np.linspace(0.05, 1.0, 5) * energy_max
    size_values = _energy_marker_sizes_shared(size_energies, energy_max=energy_max, min_size=3.0, max_size=340.0)
    xs = np.linspace(0.12, 0.88, len(size_values))
    size_ax.scatter(xs, np.full_like(xs, 0.5), s=size_values, c="#ff9f1c", edgecolors="#fff5eb", linewidths=0.2)
    size_ax.set_xlim(0.0, 1.0)
    size_ax.set_ylim(0.0, 1.0)
    footer_ax.text(0.34, 0.28, "low", transform=footer_ax.transAxes, fontsize=9, color="#666666")
    footer_ax.text(0.52, 0.28, "high", transform=footer_ax.transAxes, fontsize=9, color="#666666", ha="right")

    primary = showers[0].primary or {}
    info_lines = [
        "Shower information",
        f"Incident particle: {_particle_label_from_pdg(int(primary.get('pdg', 0)))}",
        _primary_energy_line(primary),
        "Detector: Open Data Detector",
    ]
    y = 0.67
    for i, line in enumerate(info_lines):
        footer_ax.text(
            0.80,
            y,
            line,
            transform=footer_ax.transAxes,
            fontsize=10 if i else 11,
            color="#333333",
            weight="bold" if i == 0 else None,
        )
        y -= 0.16 if i == 0 else 0.15

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.15)
    fig.savefig(outpath, facecolor=fig.get_facecolor(), pad_inches=0.02)
    plt.close(fig)
    if outpath.suffix.lower() == ".png":
        _trim_near_white_png(outpath)
    return outpath
