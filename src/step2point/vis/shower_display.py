from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

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
