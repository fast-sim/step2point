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

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────
INPUT = "/eos/project/f/fast/edm4hep_frombenchmark/test_small.edm4hep.root"
COLLECTION = "EcalBarrelCollection"
MERGE_SCOPE = "system_layer"
BASE_OUTPUT = Path("outputs/hdbscan_sweep")

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
    if log_path.exists():
        print(f"  [skip] output exists at {out_dir}")
        return out_dir

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


def plot_h5_overlay_histograms(
    root_dir,
    h5_name="compressed_hdbscan.h5",
):
    """
    Creates two overlay histograms:

        - hits_per_event_overlay.png
        - hit_energy_overlay.png

    One colored line per (mcs, ms) configuration.
    """
    root_dir = Path(root_dir)

    plot_dir = root_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    hits_data = []
    energy_data = []

    # configurations to skip
    ms_toskip = [2, 4, 2, 4, 2, 2, 4, 2, 4]
    mcs_toskip = [20, 20, 12, 12, 3, 5, 5, 8, 8]
    # --------------------------------------------------
    # Read some configurations

    for folder in sorted(root_dir.glob("hdbscan_mcs*_ms*")):
        match = re.search(r"mcs(\d+)_ms(\d+)", folder.name)

        if match is None:
            continue

        mcs = int(match.group(1))
        ms = int(match.group(2))
        if ms in ms_toskip and mcs in mcs_toskip:
            continue

        h5_file = folder / h5_name

        if not h5_file.exists():
            continue

        with h5py.File(h5_file, "r") as f:
            steps = f["steps"]

            energy = steps["energy"][:]
            event_id = steps["event_id"][:]

        _, inverse = np.unique(
            event_id,
            return_inverse=True,
        )

        hits_per_event = np.bincount(inverse)

        label = f"mcs={mcs}, ms={ms}"

        hits_data.append((hits_per_event, label))
        energy_data.append((energy[energy > 0], label))

    # max hits per event across all configs (for common bins)
    max_h = max((values.max() for values, _ in hits_data), default=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    b_ = np.linspace(0, max_h, 20)  # common bins for all configs
    for values, label in hits_data:
        ax.hist(
            values,
            bins=b_,  # larger bins
            density=True,
            histtype="step",
            linewidth=2,
            label=label,
        )

    ax.set_xlabel("Hits per event")
    ax.set_ylabel("Density")
    ax.set_title("Hits per event")

    ax.legend(
        fontsize=8,
        ncol=2,
    )

    fig.tight_layout()

    out_file = plot_dir / "hits_per_event_overlay.png"

    fig.savefig(
        out_file,
        dpi=150,
    )

    plt.close(fig)

    print(f"saved {out_file}")

    # --------------------------------------------------
    # Hit energy (log scale)
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    all_energy = np.concatenate([vals for vals, _ in energy_data])

    bins = np.logspace(
        np.log10(all_energy.min()),
        np.log10(all_energy.max()),
        80,
    )

    for values, label in energy_data:
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Hit energy")
    ax.set_ylabel("Density")
    ax.set_title("Hit energy distribution")

    ax.legend(
        fontsize=8,
        ncol=2,
    )

    fig.tight_layout()

    out_file = plot_dir / "hit_energy_overlay.png"

    fig.savefig(
        out_file,
        dpi=150,
    )

    plt.close(fig)

    print(f"saved {out_file}")


def plot_compression_ratios(
    root_dir,
    txt_name="compression_summary_hdbscan.txt",
):
    root_dir = Path(root_dir)

    rows = []

    for folder in sorted(root_dir.glob("hdbscan_mcs*_ms*")):
        match = re.search(r"mcs(\d+)_ms(\d+)", folder.name)
        if match is None:
            continue

        mcs = int(match.group(1))
        ms = int(match.group(2))

        txt_file = folder / txt_name

        if not txt_file.exists():
            continue

        row = {"mcs": mcs, "ms": ms}

        with open(txt_file) as f:
            for line in f:
                if "=" not in line:
                    continue

                key, value = line.strip().split("=", 1)
                try:
                    row[key] = float(value)
                except ValueError:
                    pass

        rows.append(row)

    if not rows:
        print("No summary files found.")
        return

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharex=True,
        sharey=True,
    )
    # plot merge within cell point
    txt_within_cell = "outputs/pipeline2_merge_within_cell/test_small/compression_summary_merge_within_cell.txt"
    if Path(txt_within_cell).exists():
        row = {}
        with open(txt_within_cell) as f:
            for line in f:
                if "=" not in line:
                    continue

                key, value = line.strip().split("=", 1)
                try:
                    row[key] = float(value)
                except ValueError:
                    pass

        if "mean_compression_ratio" in row and "total_compression_ratio" in row:
            axes[0].axhline(
                row["mean_compression_ratio"],
                color="k",
                linestyle="--",
                label="merge within cell (mean)",
            )
            axes[1].axhline(
                row["total_compression_ratio"],
                color="k",
                linestyle="--",
                label="merge within cell (total)",
            )

    metrics = [
        "mean_compression_ratio",
        "total_compression_ratio",
    ]

    titles = [
        "Mean compression ratio",
        "Total compression ratio",
    ]

    for ax, metric, title in zip(axes, metrics, titles):
        for ms, grp in sorted(df.groupby("ms")):
            grp = grp.sort_values("mcs")

            ax.plot(
                grp["mcs"],
                grp[metric],
                marker="o",
                linewidth=2,
                label=f"ms={ms}",
            )

        ax.set_xlabel("min_cluster_size")
        ax.set_ylabel("compression ratio")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[0].legend(
        title="min_samples",
        frameon=False,
    )

    fig.tight_layout()

    out_file = root_dir / "plots/compression_ratios.png"

    fig.savefig(
        out_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"saved {out_file}")


def make_all_plots():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n── Generating plots ──────────────────────────────")

    # line plots
    plot_compression_ratios(BASE_OUTPUT, txt_name="compression_summary_hdbscan.txt")
    plot_h5_overlay_histograms(BASE_OUTPUT, h5_name="compressed_hdbscan.h5")

    print(f"\nDone. Plots saved in: {PLOT_DIR.resolve()}")


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

    print(h5py.File(out_dir / "compressed_hdbscan.h5", "r").keys())
    df = pd.DataFrame(rows)
    csv_path = BASE_OUTPUT / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n── Summary saved to {csv_path}")

    make_all_plots()
    print("\nDone. Plots in", PLOT_DIR)


if __name__ == "__main__":
    main()
