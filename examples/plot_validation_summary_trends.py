from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator


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


def collect_reference_metric_points(summary_paths: list[str], metric: str, reference_label: str):
    baseline_metric = {
        "n_points_post": "n_points_pre",
        "n_cells_post": "n_cells_pre",
    }.get(metric)
    if baseline_metric is None:
        return {}

    grouped: dict[str, list[tuple[float, float, float]]] = {reference_label: []}
    for summary_path in summary_paths:
        summary = load_summary(summary_path)
        comparisons = summary.get("comparisons", [])
        if not comparisons or not isinstance(comparisons[0], dict):
            continue
        comparison = comparisons[0]
        energy = comparison.get("input_energy_GeV")
        distributions = comparison.get("distributions", {})
        metric_summary = distributions.get(baseline_metric, {}) if isinstance(distributions, dict) else {}
        mean = metric_summary.get("mean") if isinstance(metric_summary, dict) else None
        std = metric_summary.get("std") if isinstance(metric_summary, dict) else None
        if energy is None or mean is None or std is None:
            continue
        grouped[reference_label].append((float(energy), float(mean), float(std)))
    return grouped


def plot_metric_vs_energy(
    summary_paths: list[str],
    metric: str,
    outpath: str | Path,
    *,
    reference_label: str = "Geant4 steps",
) -> None:
    grouped = collect_metric_points(summary_paths, metric)
    grouped.update(collect_reference_metric_points(summary_paths, metric, reference_label))
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
        "point_count_ratio": "Ratio of numner of points vs incident energy",
        "cell_count_ratio": "Ratio of number of cells vs incident energy",
        "n_points_post": "Number of points vs incident energy",
        "n_points_pre": "Pre-compression point count vs incident energy",
        "n_cells_post": "Number of cells vs incident energy",
        "n_cells_pre": "Pre-compression cell count vs incident energy",
    }
    ylabel_map = {
        "energy_ratio": "Ratio of <E>",
        "point_count_ratio": "Ratio of <# points>",
        "cell_count_ratio": "Ratio of <# cells>",
        "n_points_post": "<# points>",
        "n_points_pre": "<# points>",
        "n_cells_post": "<# cells>",
        "n_cells_pre": "<# cells>",
    }
    ax.set_title(title_map.get(metric, metric))
    ax.set_xlabel("Incident energy [GeV]")
    ax.set_ylabel(ylabel_map.get(metric, metric))
    if metric in {"n_points_post", "n_points_pre"}:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.grid(True, which="major", axis="y", color="#B8B8B8", alpha=0.55, linewidth=0.8)
        ax.grid(True, which="minor", axis="y", color="#D8D8D8", alpha=0.35, linewidth=0.6)
    elif metric in {"n_cells_post", "n_cells_pre"}:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.grid(True, which="major", axis="y", color="#B8B8B8", alpha=0.45, linewidth=0.8)
    else:
        ax.grid(True, which="major", axis="y", color="#B8B8B8", alpha=0.35, linewidth=0.8)
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
    parser.add_argument(
        "--reference-label",
        default="Geant4 steps",
        help="Legend label used for the original-shower baseline on absolute count plots.",
    )
    parser.add_argument("--outdir", default="outputs/validation_summary_trends")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    for metric in args.metric:
        plot_metric_vs_energy(
            args.summary,
            metric,
            outdir / f"{metric}_vs_energy.png",
            reference_label=args.reference_label,
        )


if __name__ == "__main__":
    main()
