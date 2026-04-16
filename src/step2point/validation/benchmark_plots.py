from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from step2point.metrics.energy import aggregate_cell_energy, energy_ratio
from step2point.metrics.spatial import estimate_shower_axis, longitudinal_radial_phi
from step2point.validation.observables import aggregate_observables, compute_shower_observables
from step2point.validation.plotting import plot_hist, plot_overlay_hist, plot_overlay_line


@dataclass(slots=True)
class PlotArtifacts:
    outdir: Path


def _safe_log10(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.log10(np.clip(values, eps, None))


def _profile(values: np.ndarray, weights: np.ndarray, bins: np.ndarray) -> np.ndarray:
    sums, _ = np.histogram(values, bins=bins, weights=weights)
    counts, _ = np.histogram(values, bins=bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        prof = np.divide(sums, np.maximum(counts, 1), dtype=np.float64)
    return prof


def _moment(coord: np.ndarray, weights: np.ndarray, order: int) -> float:
    if weights.size == 0 or np.sum(weights) <= 0:
        return float("nan")
    return float(np.average(coord**order, weights=weights))


def generate_benchmark_plots(pairs, outdir: str | Path) -> PlotArtifacts:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    energy_ratios = []
    cell_ratios = []
    point_ratios = []
    pre_cell_logs = []
    post_cell_logs = []
    pre_point_logs = []
    post_point_logs = []
    long_m1_pre, long_m1_post, long_m2_pre, long_m2_post = [], [], [], []
    rad_m1_pre, rad_m1_post, rad_m2_pre, rad_m2_post = [], [], [], []
    long_profiles_pre = []
    long_profiles_post = []
    radial_profiles_pre = []
    radial_profiles_post = []
    phi_profiles_pre = []
    phi_profiles_post = []

    long_bins = np.linspace(-200.0, 200.0, 41)
    radial_bins = np.linspace(0.0, 200.0, 41)
    phi_bins = np.linspace(-np.pi, np.pi, 41)

    for pre, post in pairs:
        energy_ratios.append(energy_ratio(pre, post))
        point_ratios.append(post.n_points / pre.n_points if pre.n_points else np.nan)

        if pre.cell_id is not None and post.cell_id is not None:
            _, pre_cell_e = aggregate_cell_energy(pre)
            _, post_cell_e = aggregate_cell_energy(post)
            cell_ratios.append(len(np.unique(post.cell_id)) / max(len(np.unique(pre.cell_id)), 1))
            pre_cell_logs.extend(_safe_log10(pre_cell_e))
            post_cell_logs.extend(_safe_log10(post_cell_e))

        pre_point_logs.extend(_safe_log10(pre.E))
        post_point_logs.extend(_safe_log10(post.E))

        centroid, axis = estimate_shower_axis(pre)
        long_pre, radial_pre, phi_pre = longitudinal_radial_phi(
            pre,
            centroid=centroid,
            axis=axis,
            longitudinal_origin="first_deposit",
        )
        long_post, radial_post, phi_post = longitudinal_radial_phi(
            post,
            centroid=centroid,
            axis=axis,
            longitudinal_origin="first_deposit",
        )

        long_m1_pre.append(_moment(long_pre, pre.E, 1))
        long_m1_post.append(_moment(long_post, post.E, 1))
        long_m2_pre.append(_moment(long_pre, pre.E, 2))
        long_m2_post.append(_moment(long_post, post.E, 2))
        rad_m1_pre.append(_moment(radial_pre, pre.E, 1))
        rad_m1_post.append(_moment(radial_post, post.E, 1))
        rad_m2_pre.append(_moment(radial_pre, pre.E, 2))
        rad_m2_post.append(_moment(radial_post, post.E, 2))

        long_profiles_pre.append(_profile(long_pre, pre.E, long_bins))
        long_profiles_post.append(_profile(long_post, post.E, long_bins))
        radial_profiles_pre.append(_profile(radial_pre, pre.E, radial_bins))
        radial_profiles_post.append(_profile(radial_post, post.E, radial_bins))
        phi_profiles_pre.append(_profile(phi_pre, pre.E, phi_bins))
        phi_profiles_post.append(_profile(phi_post, post.E, phi_bins))

    plot_hist(energy_ratios, outdir / "energy_ratio.png", "Energy ratio", "E_post / E_pre")
    plot_hist(cell_ratios, outdir / "cell_count_ratio.png", "Cell count ratio", "N_cells_post / N_cells_pre")
    plot_hist(point_ratios, outdir / "point_count_ratio.png", "Point count ratio", "N_points_post / N_points_pre")
    plot_overlay_hist(
        pre_cell_logs, post_cell_logs, outdir / "log_cell_energy.png", "Cell energy spectrum", "log10(cell energy)", logy=True
    )
    plot_overlay_hist(
        pre_point_logs,
        post_point_logs,
        outdir / "log_point_energy.png",
        "Point energy spectrum",
        "log10(point energy)",
        logy=True,
    )

    long_centers = 0.5 * (long_bins[:-1] + long_bins[1:])
    radial_centers = 0.5 * (radial_bins[:-1] + radial_bins[1:])
    phi_centers = 0.5 * (phi_bins[:-1] + phi_bins[1:])
    plot_overlay_line(
        long_centers,
        np.mean(long_profiles_pre, axis=0),
        np.mean(long_profiles_post, axis=0),
        outdir / "longitudinal_profile_overlay.png",
        "Longitudinal profile",
        "longitudinal coordinate",
    )
    plot_overlay_line(
        radial_centers,
        np.mean(radial_profiles_pre, axis=0),
        np.mean(radial_profiles_post, axis=0),
        outdir / "radial_profile_overlay.png",
        "Radial profile",
        "radial coordinate",
    )
    plot_overlay_line(
        phi_centers,
        np.mean(phi_profiles_pre, axis=0),
        np.mean(phi_profiles_post, axis=0),
        outdir / "phi_profile_overlay.png",
        "Phi profile",
        "phi",
    )

    plot_overlay_hist(long_m1_pre, long_m1_post, outdir / "longitudinal_moment_1.png", "Longitudinal first moment", "m1")
    plot_overlay_hist(long_m2_pre, long_m2_post, outdir / "longitudinal_moment_2.png", "Longitudinal second moment", "m2")
    plot_overlay_hist(rad_m1_pre, rad_m1_post, outdir / "radial_moment_1.png", "Radial first moment", "m1")
    plot_overlay_hist(rad_m2_pre, rad_m2_post, outdir / "radial_moment_2.png", "Radial second moment", "m2")

    return PlotArtifacts(outdir=outdir)


def generate_observables_matrix(showers, outpath: str | Path, *, selected_index: int | None = None, axis_override=None):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    all_data = [compute_shower_observables(shower, axis_override=axis_override) for shower in showers]
    average_data = aggregate_observables(all_data)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    event_count = len(all_data)
    avg_color = "#0057D9"
    selected_color = "#FF7A00"

    def plot_avg(ax, data_list, key, xlabel, logy: bool = False):
        bins = data_list[0][key][1]
        values = np.array([row[key][0] for row in data_list], dtype=np.float64)
        centers = 0.5 * (bins[:-1] + bins[1:])
        mean_values = np.mean(values, axis=0)
        ax.plot(centers, mean_values, color=avg_color, linewidth=2.2, alpha=0.22, linestyle="--", zorder=3)
        ax.scatter(
            centers,
            mean_values,
            color=avg_color,
            s=32,
            marker="o",
            linewidths=0.0,
            label=f"average over {event_count} showers",
            zorder=4,
        )
        ax.set_xlabel(xlabel)
        if logy:
            ax.set_yscale("log")
        if key == "long_profile":
            ax.set_xlim(left=0.0)

    plot_avg(axes[0, 0], all_data, "long_profile", "longitudinal (from first deposit)")
    plot_avg(axes[0, 1], all_data, "r_profile", "radial", logy=True)
    plot_avg(axes[0, 2], all_data, "log_energy", "log10(energy)", logy=True)

    for i, (key, label) in enumerate(
        zip(
            ["mean_long", "mean_r", "total_energy"],
            ["first longitudinal moment", "first radial moment", "total energy"],
            strict=True,
        )
    ):
        axes[1, i].hist(average_data[key], color=avg_color, bins=20, alpha=0.45, label=f"average over {event_count} showers")
        axes[1, i].set_xlabel(label)

    for i, (key, label) in enumerate(
        zip(
            ["var_long", "var_r", "num_steps"],
            ["second longitudinal moment", "second radial moment", "number of steps"],
            strict=True,
        )
    ):
        axes[2, i].hist(average_data[key], color=avg_color, bins=20, alpha=0.45, label=f"average over {event_count} showers")
        axes[2, i].set_xlabel(label)

    if selected_index is not None:
        selected = all_data[selected_index]
        for i, key in enumerate(["long_profile", "r_profile", "log_energy"]):
            centers = 0.5 * (selected[key][1][:-1] + selected[key][1][1:])
            axes[0, i].plot(centers, selected[key][0], color=selected_color, linewidth=1.3, alpha=0.22, linestyle="--", zorder=1)
            axes[0, i].scatter(
                centers,
                selected[key][0],
                color=selected_color,
                s=26,
                marker="s",
                linewidths=0.0,
                label=f"shower {selected_index}",
                zorder=2,
            )
            axes[0, i].legend()
        for i, key in enumerate(["mean_long", "mean_r", "total_energy"]):
            axes[1, i].axvline(
                float(selected[key]), color=selected_color, linewidth=2.0, linestyle="--", label=f"shower {selected_index}"
            )
            axes[1, i].legend()
        for i, key in enumerate(["var_long", "var_r", "num_steps"]):
            axes[2, i].axvline(
                float(selected[key]), color=selected_color, linewidth=2.0, linestyle="--", label=f"shower {selected_index}"
            )
            axes[2, i].legend()
    else:
        for ax in axes[0]:
            ax.legend()
        for ax in axes[1]:
            ax.legend()
        for ax in axes[2]:
            ax.legend()

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
