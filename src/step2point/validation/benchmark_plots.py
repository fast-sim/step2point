from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from step2point.metrics.energy import aggregate_cell_energy, energy_ratio
from step2point.metrics.spatial import estimate_shower_axis
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


def shower_coordinates(shower):
    centroid, axis = estimate_shower_axis(shower)
    coords = np.stack([shower.x, shower.y, shower.z], axis=1)
    rel = coords - centroid
    long = rel @ axis
    radial_vec = rel - np.outer(long, axis)
    radial = np.linalg.norm(radial_vec, axis=1)

    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, axis)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = ref - np.dot(ref, axis) * axis
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    phi = np.arctan2(radial_vec @ e2, radial_vec @ e1)
    return long, radial, phi


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

        long_pre, radial_pre, phi_pre = shower_coordinates(pre)
        long_post, radial_post, phi_post = shower_coordinates(post)

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
    plot_overlay_hist(pre_cell_logs, post_cell_logs, outdir / "log_cell_energy.png", "Cell energy spectrum", "log10(cell energy)", logy=True)
    plot_overlay_hist(pre_point_logs, post_point_logs, outdir / "log_point_energy.png", "Point energy spectrum", "log10(point energy)", logy=True)

    long_centers = 0.5 * (long_bins[:-1] + long_bins[1:])
    radial_centers = 0.5 * (radial_bins[:-1] + radial_bins[1:])
    phi_centers = 0.5 * (phi_bins[:-1] + phi_bins[1:])
    plot_overlay_line(long_centers, np.mean(long_profiles_pre, axis=0), np.mean(long_profiles_post, axis=0), outdir / "longitudinal_profile_overlay.png", "Longitudinal profile", "longitudinal coordinate")
    plot_overlay_line(radial_centers, np.mean(radial_profiles_pre, axis=0), np.mean(radial_profiles_post, axis=0), outdir / "radial_profile_overlay.png", "Radial profile", "radial coordinate")
    plot_overlay_line(phi_centers, np.mean(phi_profiles_pre, axis=0), np.mean(phi_profiles_post, axis=0), outdir / "phi_profile_overlay.png", "Phi profile", "phi")

    plot_overlay_hist(long_m1_pre, long_m1_post, outdir / "longitudinal_moment_1.png", "Longitudinal first moment", "m1")
    plot_overlay_hist(long_m2_pre, long_m2_post, outdir / "longitudinal_moment_2.png", "Longitudinal second moment", "m2")
    plot_overlay_hist(rad_m1_pre, rad_m1_post, outdir / "radial_moment_1.png", "Radial first moment", "m1")
    plot_overlay_hist(rad_m2_pre, rad_m2_post, outdir / "radial_moment_2.png", "Radial second moment", "m2")

    return PlotArtifacts(outdir=outdir)
