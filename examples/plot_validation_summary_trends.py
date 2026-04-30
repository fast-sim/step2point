from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def series_name(comparison: dict[str, object]) -> str:
    label = str(comparison.get("label") or "post")
    algorithm = comparison.get("algorithm")
    if label != "post":
        return label
    if algorithm:
        return str(algorithm)
    return label


def collect_metric_points(summary_paths: list[str], metric: str):
    grouped: dict[str, list[tuple[float, float, float]]] = {}
    for summary_path in summary_paths:
        summary = load_summary(summary_path)
        for comparison in summary.get("comparisons", []):
            if not isinstance(comparison, dict):
                continue
            energy = comparison.get("input_energy_GeV")
            distributions = comparison.get("distributions", {})
            metric_summary = distributions.get(metric, {}) if isinstance(distributions, dict) else {}
            mean = metric_summary.get("mean") if isinstance(metric_summary, dict) else None
            std = metric_summary.get("std") if isinstance(metric_summary, dict) else None
            if energy is None or mean is None or std is None:
                continue
            grouped.setdefault(series_name(comparison), []).append((float(energy), float(mean), float(std)))
    return grouped


def plot_metric_vs_energy(summary_paths: list[str], metric: str, outpath: str | Path) -> None:
    grouped = collect_metric_points(summary_paths, metric)
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#0057D9", "#C2185B", "#1B9E77", "#D95F02", "#6A3D9A", "#B8860B"]

    for idx, (name, points) in enumerate(sorted(grouped.items())):
        points = sorted(points, key=lambda item: item[0])
        x = np.asarray([item[0] for item in points], dtype=np.float64)
        y = np.asarray([item[1] for item in points], dtype=np.float64)
        y_std = np.asarray([item[2] for item in points], dtype=np.float64)
        color = colors[idx % len(colors)]
        ax.plot(x, y, color=color, linewidth=2.0, marker="o", markersize=5, label=name)
        ax.fill_between(x, y - y_std, y + y_std, color=color, alpha=0.15)

    title_map = {
        "energy_ratio": "Energy ratio vs incident energy",
        "point_count_ratio": "Point count ratio vs incident energy",
        "cell_count_ratio": "Cell count ratio vs incident energy",
        "n_points_post": "Post-compression point count vs incident energy",
        "n_points_pre": "Pre-compression point count vs incident energy",
        "n_cells_post": "Post-compression cell count vs incident energy",
        "n_cells_pre": "Pre-compression cell count vs incident energy",
    }
    ylabel_map = {
        "energy_ratio": "Mean E_post / E_pre",
        "point_count_ratio": "Mean N_points_post / N_points_pre",
        "cell_count_ratio": "Mean N_cells_post / N_cells_pre",
        "n_points_post": "Mean N_points_post",
        "n_points_pre": "Mean N_points_pre",
        "n_cells_post": "Mean N_cells_post",
        "n_cells_pre": "Mean N_cells_pre",
    }
    ax.set_title(title_map.get(metric, metric))
    ax.set_xlabel("Incident energy [GeV]")
    ax.set_ylabel(ylabel_map.get(metric, metric))
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", nargs="+", required=True, help="Validation summary JSON files.")
    parser.add_argument(
        "--metric",
        nargs="+",
        default=["n_points_post", "point_count_ratio", "cell_count_ratio", "energy_ratio"],
        help="Metrics to plot versus input energy.",
    )
    parser.add_argument("--outdir", default="outputs/validation_summary_trends")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    for metric in args.metric:
        plot_metric_vs_energy(args.summary, metric, outdir / f"{metric}_vs_energy.png")


if __name__ == "__main__":
    main()
