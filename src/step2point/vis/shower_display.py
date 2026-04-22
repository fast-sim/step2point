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

    non_empty_coords = [np.stack([s.x, s.y, s.z], axis=1) for s in showers if s.n_points > 0]
    if not non_empty_coords:
        raise ValueError("At least one shower must contain points.")
    all_coords = np.concatenate(non_empty_coords, axis=0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    _set_display_style(fig, ax)
    ax.set_position((0.0, 0.0, 1.0, 1.0))

    palettes = [
        LinearSegmentedColormap.from_list("display_orange", ["#ffeb99", "#ff8f1f", "#d62828"]),
        LinearSegmentedColormap.from_list("display_teal", ["#b8fff4", "#23c9b7", "#15616d"]),
        LinearSegmentedColormap.from_list("display_blue", ["#d9f0ff", "#4ea8de", "#4361ee"]),
    ]
    edge_colours = ["#fff0b8", "#d8fff9", "#dde9ff"]

    for index, shower in enumerate(showers):
        if shower.n_points == 0:
            continue
        coords = np.stack([shower.x, shower.y, shower.z], axis=1)
        energy = np.asarray(shower.E, dtype=np.float64)
        colours = np.log10(np.clip(energy, 1e-12, None))
        norm = Normalize(vmin=float(np.percentile(colours, 4)), vmax=float(np.percentile(colours, 99)))
        cmap = palettes[index % len(palettes)]
        if index == 0:
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=colours,
                cmap=cmap,
                norm=norm,
                s=_energy_marker_sizes(energy, min_size=2.0, max_size=18.0),
                alpha=0.20,
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
                s=_energy_marker_sizes(energy, min_size=18.0, max_size=70.0),
                alpha=0.88,
                edgecolors=edge_colours[index % len(edge_colours)],
                linewidths=0.35,
                depthshade=False,
            )

    reference = showers[0]
    axis = _display_axis(reference, axis_override=axis_override)
    all_energy = np.concatenate([np.asarray(s.E, dtype=np.float64) for s in showers if s.n_points > 0])
    origin = np.average(all_coords, axis=0, weights=all_energy)
    projected = (all_coords - origin) @ axis
    span_long = float(np.ptp(projected))
    line_length = max(0.24 * span_long, 12.0)
    gap = max(0.025 * span_long, 1.5)
    front_cut = np.percentile(projected, 8.0)
    front_mask = projected <= front_cut
    front_coords = all_coords[front_mask]
    front_energy = all_energy[front_mask]
    if front_coords.size and float(np.sum(front_energy)) > 0.0:
        front_anchor = np.average(front_coords, axis=0, weights=front_energy)
    else:
        front_anchor = all_coords[int(np.argmin(projected))]
    front_long = float((front_anchor - origin) @ axis)
    shower_front = front_anchor + (float(np.min(projected)) - front_long) * axis
    line_start = shower_front - (gap + line_length) * axis
    line_end = shower_front - gap * axis
    ax.plot(
        [line_start[0], line_end[0]],
        [line_start[1], line_end[1]],
        [line_start[2], line_end[2]],
        color="#7a7a7a",
        linewidth=0.82,
        linestyle=(0, (1.7, 2.0)),
        alpha=0.72,
    )

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
        incident_label if incident_label is not None else _default_incident_label(reference),
        color="#666666",
        fontsize=7,
        alpha=0.78,
        rotation=label_angle,
        rotation_mode="anchor",
        ha="center",
        va="center",
    )
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.savefig(outpath, facecolor=fig.get_facecolor(), pad_inches=0.0)
    plt.close(fig)
    if outpath.suffix.lower() == ".png":
        _trim_near_white_png(outpath)
    return outpath
