from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from step2point.metrics.energy import aggregate_cell_energy, energy_ratio
from step2point.metrics.spatial import estimate_shower_axis, longitudinal_radial_phi
from step2point.validation.observables import aggregate_observables, compute_shower_observables
from step2point.validation.plotting import (
    plot_hist,
    plot_hist_series,
    plot_overlay_hist,
    plot_overlay_hist_multi,
    plot_overlay_line,
    plot_overlay_line_multi,
)


@dataclass(slots=True)
class PlotArtifacts:
    outdir: Path


def _upper_percentile_limit(values: np.ndarray, percentile: float = 99.0, pad: float = 0.05) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 1.0
    upper = float(np.percentile(values, percentile))
    return max(upper * (1.0 + pad), 1.0)


def _safe_log10(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.log10(np.clip(values, eps, None))


def _profile(values: np.ndarray, weights: np.ndarray, bins: np.ndarray) -> np.ndarray:
    sums, _ = np.histogram(values, bins=bins, weights=weights)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.zeros(len(bins) - 1, dtype=np.float64)
    return sums.astype(np.float64) / total


def _moment(coord: np.ndarray, weights: np.ndarray, order: int) -> float:
    if weights.size == 0 or np.sum(weights) <= 0:
        return float("nan")
    return float(np.average(coord**order, weights=weights))


def _longitudinal_radial_phi_with_reference(
    shower,
    *,
    centroid: np.ndarray,
    axis: np.ndarray,
    longitudinal_origin_projection: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    long_raw, radial, phi = longitudinal_radial_phi(
        shower,
        centroid=centroid,
        axis=axis,
        longitudinal_origin="centroid",
    )
    return long_raw - longitudinal_origin_projection, radial, phi


def _compute_benchmark_data(pairs, *, axis_override=None, origin_override=None) -> dict[str, object]:
    data: dict[str, object] = {
        "energy_ratios": [],
        "cell_ratios": [],
        "point_ratios": [],
        "pre_cell_logs": [],
        "post_cell_logs": [],
        "pre_point_logs": [],
        "post_point_logs": [],
        "long_m1_pre": [],
        "long_m1_post": [],
        "long_m2_pre": [],
        "long_m2_post": [],
        "rad_m1_pre": [],
        "rad_m1_post": [],
        "rad_m2_pre": [],
        "rad_m2_post": [],
        "long_profiles_pre": [],
        "long_profiles_post": [],
        "radial_profiles_pre": [],
        "radial_profiles_post": [],
        "phi_profiles_pre": [],
        "phi_profiles_post": [],
    }

    long_bins = np.linspace(-200.0, 200.0, 41)
    radial_bins = np.linspace(0.0, 200.0, 41)
    phi_bins = np.linspace(-np.pi, np.pi, 41)

    for pre, post in pairs:
        data["energy_ratios"].append(energy_ratio(pre, post))
        data["point_ratios"].append(post.n_points / pre.n_points if pre.n_points else np.nan)

        if pre.cell_id is not None and post.cell_id is not None:
            _, pre_cell_e = aggregate_cell_energy(pre)
            _, post_cell_e = aggregate_cell_energy(post)
            data["cell_ratios"].append(len(np.unique(post.cell_id)) / max(len(np.unique(pre.cell_id)), 1))
            data["pre_cell_logs"].extend(_safe_log10(pre_cell_e))
            data["post_cell_logs"].extend(_safe_log10(post_cell_e))

        data["pre_point_logs"].extend(_safe_log10(pre.E))
        data["post_point_logs"].extend(_safe_log10(post.E))

        centroid, axis = estimate_shower_axis(pre, axis_override=axis_override)
        if origin_override is not None:
            centroid = np.asarray(origin_override, dtype=np.float64)
            if centroid.shape != (3,):
                raise ValueError(f"origin_override must be a length-3 vector, got shape {centroid.shape}.")
            long_origin = 0.0
        else:
            long_pre_raw, _, _ = longitudinal_radial_phi(
                pre,
                centroid=centroid,
                axis=axis,
                longitudinal_origin="centroid",
            )
            long_origin = float(np.min(long_pre_raw)) if long_pre_raw.size else 0.0
        long_pre, radial_pre, phi_pre = _longitudinal_radial_phi_with_reference(
            pre,
            centroid=centroid,
            axis=axis,
            longitudinal_origin_projection=long_origin,
        )
        long_post, radial_post, phi_post = _longitudinal_radial_phi_with_reference(
            post,
            centroid=centroid,
            axis=axis,
            longitudinal_origin_projection=long_origin,
        )

        data["long_m1_pre"].append(_moment(long_pre, pre.E, 1))
        data["long_m1_post"].append(_moment(long_post, post.E, 1))
        data["long_m2_pre"].append(_moment(long_pre, pre.E, 2))
        data["long_m2_post"].append(_moment(long_post, post.E, 2))
        data["rad_m1_pre"].append(_moment(radial_pre, pre.E, 1))
        data["rad_m1_post"].append(_moment(radial_post, post.E, 1))
        data["rad_m2_pre"].append(_moment(radial_pre, pre.E, 2))
        data["rad_m2_post"].append(_moment(radial_post, post.E, 2))

        data["long_profiles_pre"].append(_profile(long_pre, pre.E, long_bins))
        data["long_profiles_post"].append(_profile(long_post, post.E, long_bins))
        data["radial_profiles_pre"].append(_profile(radial_pre, pre.E, radial_bins))
        data["radial_profiles_post"].append(_profile(radial_post, post.E, radial_bins))
        data["phi_profiles_pre"].append(_profile(phi_pre, pre.E, phi_bins))
        data["phi_profiles_post"].append(_profile(phi_post, post.E, phi_bins))

    data["long_bins"] = long_bins
    data["radial_bins"] = radial_bins
    data["phi_bins"] = phi_bins
    return data


def generate_benchmark_plots(
    pairs,
    outdir: str | Path,
    *,
    axis_override=None,
    origin_override=None,
    pre_label: str = "pre",
    post_label: str = "post",
) -> PlotArtifacts:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = _compute_benchmark_data(pairs, axis_override=axis_override, origin_override=origin_override)

    plot_hist(data["energy_ratios"], outdir / "energy_ratio.png", "Energy ratio", "E_post / E_pre")
    plot_hist(data["cell_ratios"], outdir / "cell_count_ratio.png", "Cell count ratio", "N_cells_post / N_cells_pre")
    plot_hist(data["point_ratios"], outdir / "point_count_ratio.png", "Point count ratio", "N_points_post / N_points_pre")
    plot_overlay_hist(
        data["pre_cell_logs"],
        data["post_cell_logs"],
        outdir / "log_cell_energy.png",
        "Cell energy spectrum",
        "log10(cell energy [GeV])",
        logy=True,
        pre_label=pre_label,
        post_label=post_label,
    )
    plot_overlay_hist(
        data["pre_point_logs"],
        data["post_point_logs"],
        outdir / "log_point_energy.png",
        "Point energy spectrum",
        "log10(point energy [GeV])",
        logy=True,
        pre_label=pre_label,
        post_label=post_label,
    )

    long_centers = 0.5 * (data["long_bins"][:-1] + data["long_bins"][1:])
    radial_centers = 0.5 * (data["radial_bins"][:-1] + data["radial_bins"][1:])
    phi_centers = 0.5 * (data["phi_bins"][:-1] + data["phi_bins"][1:])
    plot_overlay_line(
        long_centers,
        np.mean(data["long_profiles_pre"], axis=0),
        np.mean(data["long_profiles_post"], axis=0),
        outdir / "longitudinal_profile_overlay.png",
        "Longitudinal profile",
        "longitudinal coordinate [mm]",
        ylabel="Energy fraction",
        pre_label=pre_label,
        post_label=post_label,
    )
    plot_overlay_line(
        radial_centers,
        np.mean(data["radial_profiles_pre"], axis=0),
        np.mean(data["radial_profiles_post"], axis=0),
        outdir / "radial_profile_overlay.png",
        "Radial profile",
        "radial coordinate [mm]",
        ylabel="Energy fraction",
        pre_label=pre_label,
        post_label=post_label,
    )
    plot_overlay_line(
        phi_centers,
        np.mean(data["phi_profiles_pre"], axis=0),
        np.mean(data["phi_profiles_post"], axis=0),
        outdir / "phi_profile_overlay.png",
        "Phi profile",
        "phi",
        ylabel="Energy fraction",
        pre_label=pre_label,
        post_label=post_label,
    )

    plot_overlay_hist(
        data["long_m1_pre"],
        data["long_m1_post"],
        outdir / "longitudinal_moment_1.png",
        "Longitudinal first moment",
        "m1",
        pre_label=pre_label,
        post_label=post_label,
    )
    plot_overlay_hist(
        data["long_m2_pre"],
        data["long_m2_post"],
        outdir / "longitudinal_moment_2.png",
        "Longitudinal second moment",
        "m2",
        pre_label=pre_label,
        post_label=post_label,
    )
    plot_overlay_hist(
        data["rad_m1_pre"],
        data["rad_m1_post"],
        outdir / "radial_moment_1.png",
        "Radial first moment",
        "m1",
        pre_label=pre_label,
        post_label=post_label,
    )
    plot_overlay_hist(
        data["rad_m2_pre"],
        data["rad_m2_post"],
        outdir / "radial_moment_2.png",
        "Radial second moment",
        "m2",
        pre_label=pre_label,
        post_label=post_label,
    )

    return PlotArtifacts(outdir=outdir)


def generate_benchmark_comparison_plots(
    comparisons,
    outdir: str | Path,
    *,
    axis_override=None,
    origin_override=None,
    pre_label: str = "pre",
) -> PlotArtifacts:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    series = [
        (
            compared_label,
            _compute_benchmark_data(
                pairs,
                axis_override=axis_override,
                origin_override=origin_override,
            ),
        )
        for compared_label, pairs in comparisons
    ]
    reference_data = series[0][1]

    plot_hist_series(
        [(label, data["energy_ratios"]) for label, data in series],
        outdir / "energy_ratio.png",
        "Energy ratio",
        "E_post / E_pre",
    )
    plot_hist_series(
        [(label, data["cell_ratios"]) for label, data in series],
        outdir / "cell_count_ratio.png",
        "Cell count ratio",
        "N_cells_post / N_cells_pre",
    )
    plot_hist_series(
        [(label, data["point_ratios"]) for label, data in series],
        outdir / "point_count_ratio.png",
        "Point count ratio",
        "N_points_post / N_points_pre",
    )
    plot_overlay_hist_multi(
        reference_data["pre_cell_logs"],
        [(label, data["post_cell_logs"]) for label, data in series],
        outdir / "log_cell_energy.png",
        "Cell energy spectrum",
        "log10(cell energy [GeV])",
        logy=True,
        pre_label=pre_label,
    )
    plot_overlay_hist_multi(
        reference_data["pre_point_logs"],
        [(label, data["post_point_logs"]) for label, data in series],
        outdir / "log_point_energy.png",
        "Point energy spectrum",
        "log10(point energy [GeV])",
        logy=True,
        pre_label=pre_label,
    )

    long_centers = 0.5 * (reference_data["long_bins"][:-1] + reference_data["long_bins"][1:])
    radial_centers = 0.5 * (reference_data["radial_bins"][:-1] + reference_data["radial_bins"][1:])
    phi_centers = 0.5 * (reference_data["phi_bins"][:-1] + reference_data["phi_bins"][1:])
    plot_overlay_line_multi(
        long_centers,
        np.mean(reference_data["long_profiles_pre"], axis=0),
        [(label, np.mean(data["long_profiles_post"], axis=0)) for label, data in series],
        outdir / "longitudinal_profile_overlay.png",
        "Longitudinal profile",
        "longitudinal coordinate [mm]",
        ylabel="Energy fraction",
        pre_label=pre_label,
    )
    plot_overlay_line_multi(
        radial_centers,
        np.mean(reference_data["radial_profiles_pre"], axis=0),
        [(label, np.mean(data["radial_profiles_post"], axis=0)) for label, data in series],
        outdir / "radial_profile_overlay.png",
        "Radial profile",
        "radial coordinate [mm]",
        ylabel="Energy fraction",
        pre_label=pre_label,
    )
    plot_overlay_line_multi(
        phi_centers,
        np.mean(reference_data["phi_profiles_pre"], axis=0),
        [(label, np.mean(data["phi_profiles_post"], axis=0)) for label, data in series],
        outdir / "phi_profile_overlay.png",
        "Phi profile",
        "phi",
        ylabel="Energy fraction",
        pre_label=pre_label,
    )
    plot_overlay_hist_multi(
        reference_data["long_m1_pre"],
        [(label, data["long_m1_post"]) for label, data in series],
        outdir / "longitudinal_moment_1.png",
        "Longitudinal first moment",
        "m1",
        pre_label=pre_label,
    )
    plot_overlay_hist_multi(
        reference_data["long_m2_pre"],
        [(label, data["long_m2_post"]) for label, data in series],
        outdir / "longitudinal_moment_2.png",
        "Longitudinal second moment",
        "m2",
        pre_label=pre_label,
    )
    plot_overlay_hist_multi(
        reference_data["rad_m1_pre"],
        [(label, data["rad_m1_post"]) for label, data in series],
        outdir / "radial_moment_1.png",
        "Radial first moment",
        "m1",
        pre_label=pre_label,
    )
    plot_overlay_hist_multi(
        reference_data["rad_m2_pre"],
        [(label, data["rad_m2_post"]) for label, data in series],
        outdir / "radial_moment_2.png",
        "Radial second moment",
        "m2",
        pre_label=pre_label,
    )

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

    all_long_values = np.concatenate(
        [np.asarray(row["long_values"], dtype=np.float64) for row in all_data if np.size(row["long_values"]) > 0]
    )
    long_bins = np.linspace(0.0, _upper_percentile_limit(all_long_values), 16)
    radial_bins = np.linspace(
        0.0,
        max(float(np.max(row["radial_values"])) for row in all_data if np.size(row["radial_values"]) > 0),
        51,
    )
    log_energy_min = min(float(np.min(row["log_energy_values"])) for row in all_data if np.size(row["log_energy_values"]) > 0)
    log_energy_max = max(float(np.max(row["log_energy_values"])) for row in all_data if np.size(row["log_energy_values"]) > 0)
    log_energy_bins = np.linspace(log_energy_min, log_energy_max, 51)

    def histogram_values(row, key):
        if key == "long_profile":
            return np.histogram(row["long_values"], bins=long_bins, weights=row["weights"])[0]
        if key == "r_profile":
            return np.histogram(row["radial_values"], bins=radial_bins, weights=row["weights"])[0]
        if key == "log_energy":
            return np.histogram(row["log_energy_values"], bins=log_energy_bins)[0]
        raise ValueError(f"Unsupported profile key: {key}")

    def plot_avg(ax, data_list, key, xlabel, logy: bool = False):
        if key == "long_profile":
            bins = long_bins
        elif key == "r_profile":
            bins = radial_bins
        elif key == "log_energy":
            bins = log_energy_bins
        else:
            raise ValueError(f"Unsupported profile key: {key}")
        values = np.array([histogram_values(row, key) for row in data_list], dtype=np.float64)
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
            ax.set_xlim(0.0, long_bins[-1])

    plot_avg(axes[0, 0], all_data, "long_profile", "longitudinal (from first deposit) [mm]")
    plot_avg(axes[0, 1], all_data, "r_profile", "radial [mm]", logy=True)
    plot_avg(axes[0, 2], all_data, "log_energy", "log10(energy [GeV])", logy=True)

    for i, (key, label) in enumerate(
        zip(
            ["mean_long", "mean_r", "total_energy"],
            ["first longitudinal moment [mm]", "first radial moment [mm]", "total deposited energy [GeV]"],
            strict=True,
        )
    ):
        axes[1, i].hist(average_data[key], color=avg_color, bins=20, alpha=0.45, label=f"average over {event_count} showers")
        axes[1, i].set_xlabel(label)

    for i, (key, label) in enumerate(
        zip(
            ["var_long", "var_r", "num_steps"],
            ["second longitudinal moment [mm²]", "second radial moment [mm²]", "number of steps"],
            strict=True,
        )
    ):
        axes[2, i].hist(average_data[key], color=avg_color, bins=20, alpha=0.45, label=f"average over {event_count} showers")
        axes[2, i].set_xlabel(label)

    if selected_index is not None:
        selected = all_data[selected_index]
        for i, key in enumerate(["long_profile", "r_profile", "log_energy"]):
            if key == "long_profile":
                bins = long_bins
            elif key == "r_profile":
                bins = radial_bins
            else:
                bins = log_energy_bins
            centers = 0.5 * (bins[:-1] + bins[1:])
            values = histogram_values(selected, key)
            axes[0, i].plot(centers, values, color=selected_color, linewidth=1.3, alpha=0.22, linestyle="--", zorder=1)
            axes[0, i].scatter(
                centers,
                values,
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
