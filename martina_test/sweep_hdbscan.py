"""
HDBSCAN parameter sweep for photon showers in EDM4HEP calorimeter data.
Runs the pipeline over a grid of (min_cluster_size, min_samples) and
produces summary plots.

Usage:
    PYTHONPATH=src python sweep_hdbscan.py

Outputs:
    outputs/sweep/hdbscan_mcs{N}_ms{M}/   — one folder per run
    outputs/sweep/summary.csv             — collected metrics
    outputs/sweep/plots/                  — all figures
"""

import itertools
import json
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────
INPUT = "/eos/project/f/fast/edm4hep_frombenchmark/test_small.edm4hep.root"
COLLECTION = "EcalBarrelCollection"
MERGE_SCOPE = "system_layer"
BASE_OUTPUT = Path("step2point/outputs/hdbscan_sweep")

MIN_CLUSTER_SIZES = [3, 5, 8, 12, 20]
MIN_SAMPLES_LIST = [2, 3, 4]
# ─────────────────────────────────────────────


def run_pipeline(mcs: int, ms: int) -> Path:
    out_dir = BASE_OUTPUT / f"hdbscan_mcs{mcs}_ms{ms}"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = " ".join(
        [
            f"--input {INPUT}",
            "--algorithm hdbscan",
            "--hdbscan-cell-id-encoding system:5,module:3,stave:4,tower:4,layer:6,wafer:6,slice:4,cellX:32:-16,cellY:-16",
            f"--collections {COLLECTION}",
            f"--min-cluster-size {mcs}",
            f"--min-samples {ms}",
            f"--merge-scope {MERGE_SCOPE}",
            "--use-time",
            f"--output {out_dir}",
        ]
    )

    cmd = f"""
        cd /eos/user/m/mamozzan/step2point/ && \
        source .venv-key4hep/bin/activate && \
        python examples/run_step2point_pipeline.py {args}
    """

    log_path = out_dir / "run.log"
    print(f"  → mcs={mcs:2d}, ms={ms}  ...", end=" ", flush=True)
    with open(log_path, "w") as log:
        print(f"Running command:\n{cmd}\nLogging to {log_path}")
        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",  # needed for `source`
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(status)
    return out_dir


def parse_metrics(out_dir: Path, mcs: int, ms: int) -> dict:
    """
    Extract metrics from the pipeline output directory.

    Priority:
      1. metrics.json  (if your pipeline writes one)
      2. run.log       (regex fallback — adjust patterns to your log format)
    """
    row = {"mcs": mcs, "ms": ms}

    # ── 1. Try metrics.json ──────────────────────────────────────────────
    json_path = out_dir / "metrics.json"
    if json_path.exists():
        with open(json_path) as f:
            row.update(json.load(f))
        return row

    # ── 2. Fallback: scrape run.log ──────────────────────────────────────
    log_path = out_dir / "run.log"
    if not log_path.exists():
        return row

    text = log_path.read_text()

    patterns = {
        # adjust these regexes to match your actual log lines
        "n_clusters": r"(?:n_clusters|clusters)\s*[=:]\s*(\d+)",
        "n_noise": r"(?:noise|noise points)\s*[=:]\s*(\d+)",
        "n_points": r"(?:total points|n_points|nhits)\s*[=:]\s*(\d+)",
        "noise_fraction": r"noise fraction\s*[=:]\s*([0-9.]+)",
        "mean_cluster_size": r"mean cluster size\s*[=:]\s*([0-9.]+)",
    }

    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            row[key] = float(m.group(1))

    # Derived metrics
    if "n_noise" in row and "n_points" in row and row["n_points"] > 0:
        row.setdefault("noise_fraction", row["n_noise"] / row["n_points"])

    return row


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

PLOT_DIR = BASE_OUTPUT / "plots"
CMAP = "viridis"


def _heatmap(df: pd.DataFrame, metric: str, title: str, fmt: str = ".1f"):
    """Generic pivot heatmap over (mcs, ms) grid."""
    if metric not in df.columns:
        print(f"  [skip] '{metric}' not in data")
        return

    pivot = df.pivot_table(index="mcs", columns="ms", values=metric)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap=CMAP)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("min_samples")
    ax.set_ylabel("min_cluster_size")
    ax.set_title(title)

    # annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center", color="white", fontsize=8, fontweight="bold")

    fig.tight_layout()
    path = PLOT_DIR / f"heatmap_{metric}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_line_per_ms(df: pd.DataFrame, metric: str, ylabel: str):
    """One line per min_samples, x-axis = min_cluster_size."""
    if metric not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for ms, grp in df.groupby("ms"):
        grp_sorted = grp.sort_values("mcs")
        ax.plot(grp_sorted["mcs"], grp_sorted[metric], marker="o", label=f"min_samples={ms}")
    ax.set_xlabel("min_cluster_size")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs min_cluster_size")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    path = PLOT_DIR / f"line_{metric}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_scatter_tradeoff(df: pd.DataFrame):
    """Scatter: noise_fraction vs n_clusters, coloured by mcs."""
    if not {"noise_fraction", "n_clusters"}.issubset(df.columns):
        print("  [skip] tradeoff scatter — need noise_fraction and n_clusters")
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(df["n_clusters"], df["noise_fraction"], c=df["mcs"], cmap=CMAP, s=80, zorder=3)
    plt.colorbar(sc, ax=ax, label="min_cluster_size")
    for _, row in df.iterrows():
        ax.annotate(
            f"ms={int(row['ms'])}",
            (row["n_clusters"], row["noise_fraction"]),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=7,
        )
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("noise fraction")
    ax.set_title("Clustering trade-off: noise vs. n_clusters")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = PLOT_DIR / "scatter_tradeoff.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def make_all_plots(df: pd.DataFrame):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n── Generating plots ──────────────────────────────")

    # heatmaps
    _heatmap(df, "n_clusters", "Number of clusters", fmt=".0f")
    _heatmap(df, "n_noise", "Number of noise points", fmt=".0f")
    _heatmap(df, "noise_fraction", "Noise fraction", fmt=".3f")
    _heatmap(df, "mean_cluster_size", "Mean cluster size", fmt=".1f")
    _heatmap(df, "n_points", "Total points", fmt=".0f")

    # line plots
    plot_line_per_ms(df, "n_clusters", "n_clusters")
    plot_line_per_ms(df, "noise_fraction", "noise fraction")

    # trade-off scatter
    plot_scatter_tradeoff(df)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────


def main():
    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

    combos = [
        (mcs, ms)
        for mcs, ms in itertools.product(MIN_CLUSTER_SIZES, MIN_SAMPLES_LIST)
        if ms <= mcs  # skip invalid HDBSCAN combos
    ]

    print(f"── Running {len(combos)} configurations ──────────────────────")
    rows = []
    for mcs, ms in combos:
        out_dir = run_pipeline(mcs, ms)
        row = parse_metrics(out_dir, mcs, ms)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = BASE_OUTPUT / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n── Summary saved to {csv_path}")
    print(df.to_string(index=False))

    make_all_plots(df)
    print("\nDone. Plots in", PLOT_DIR)


if __name__ == "__main__":
    main()
