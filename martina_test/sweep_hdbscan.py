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
MERGE_SCOPE = "cell_id"
BASE_OUTPUT = Path("outputs/hdbscan_sweep")

MIN_CLUSTER_SIZES = [5, 10, 15, 25, 40, 60, 80]
MIN_SAMPLES_LIST = [3, 5, 8, 12, 20, 40, 60]
EPSILON = [0.0]  # , 0.5, 1.0, 2.0, 5.0]
# ─────────────────────────────────────────────


def run_pipeline(mcs: int, ms: int, epsilon: float) -> Path:
    out_dir = BASE_OUTPUT / f"hdbscan_mcs{mcs}_ms{ms}"
    if epsilon != 0:
        out_dir = Path(str(out_dir) + f"_eps{epsilon}")
    out_dir.mkdir(parents=True, exist_ok=True)

    args = " ".join(
        [
            f"--input {INPUT}",
            "--algorithm hdbscan",
            "--hdbscan-cell-id-encoding system:5,module:3,stave:4,tower:4,layer:6,wafer:6,slice:4,cellX:32:-16,cellY:-16",
            f"--collections {COLLECTION}",
            f"--min-cluster-size {mcs}",
            f"--min-samples {ms}",
            f"--epsilon {epsilon}",
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
    cmd_cc3 = f"""
        cd /eos/user/m/mamozzan/step2point/ && \
        source .venv-key4hep/bin/activate && \
        python martina_test/convert_to_cc3_format.py {out_dir / "compressed_hdbscan.h5"} --pc_save_folder {out_dir}
    """
    log_path = out_dir / "run.log"
    print(f"  → mcs={mcs:2d}, ms={ms} epsilon={epsilon} ...", end=" ", flush=True)

    file_dir = out_dir / "compressed_hdbscan.h5"
    if file_dir.exists():
        print(f"  [skip] output exists at {out_dir}")
    else:
        with open(log_path, "w") as log:
            print(f"Running command:\n{cmd}\nLogging to {log_path}")
            result = subprocess.run(
                cmd,
                shell=True,
                executable="/bin/bash",  # needed for `source`
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            status = "OK" if (result is not None and result.returncode == 0) else f"FAILED (rc={result.returncode})"
            print(status)

    log_path = out_dir / "run_cc3.log"
    pc_file = out_dir / "input_cc3.h5"
    if pc_file.exists():
        print("  [check] cc3_input.h5 exists")
    else:
        with open(log_path, "w") as log:
            result = subprocess.run(
                cmd_cc3,
                shell=True,
                executable="/bin/bash",  # needed for `source`
                stdout=log,
                stderr=subprocess.STDOUT,
            )

        status = "OK" if (result is not None and result.returncode == 0) else f"FAILED (rc={result.returncode})"
        print(status)
    return out_dir


def parse_metrics(out_dir: Path, mcs: int, ms: int, epsilon: float) -> dict:
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


# Shared color map: same ms value → same color in every plot
def _ms_color(ms: int, all_ms: list) -> tuple:
    cmap = plt.cm.get_cmap("tab10" if len(all_ms) <= 10 else "tab20")
    idx = sorted(all_ms).index(ms)
    return cmap(idx / max(len(all_ms) - 1, 1))


def plot_per_layer_histograms(
    root_dir,
    h5_name="input_cc3.h5",
    n_layers=30,
    n_layers_to_plot=10,
):
    root_dir = Path(root_dir)
    plot_dir = root_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    sampled_layers = np.linspace(0, n_layers - 1, n_layers_to_plot, dtype=int)

    extra_datasets = [
        (
            "/eos/user/m/mamozzan/step2point/outputs/pipeline2_identity/test_small/input_cc3.h5",
            "identity",
        ),
        (
            "/eos/user/m/mamozzan/step2point/outputs/pipeline2_merge_within_cell/test_small/input_cc3.h5",
            "merge within cell",
        ),
    ]

    # --------------------------------------------------
    # Loader
    # returns:
    #   hits_per_layer   : (n_events, n_layers)  int  — hit count per event per layer
    #   energy_per_layer : list[n_layers] of 1-D arrays — all hit energies in that layer
    #   mean_r_per_layer : (n_layers,)  energy-weighted mean r
    #   total_e_per_layer: (n_layers,)  total energy deposited
    #   sigma_r_per_layer: (n_layers,)  energy-weighted sigma r
    # --------------------------------------------------
    radial_bins = np.linspace(0, 200, 20)  # adjust as needed

    def load_layer_data(path):
        with h5py.File(path, "r") as f:
            data = f["events"][:]  # (n_events, n_hits, 4)

        x = data[:, :, 0]  # (n_events, n_hits)
        y = data[:, :, 1]
        z = data[:, :, 2]
        energy = data[:, :, 3]
        xc = np.average(x, weights=energy, axis=1)  # (n_events,)
        yc = np.average(y, weights=energy, axis=1)
        r = np.zeros_like(x)
        for i in range(x.shape[0]):
            r[i] = np.sqrt((x[i] - xc[i]) ** 2 + (y[i] - yc[i]) ** 2)

        n_events = data.shape[0]

        # map z to integer layer index
        unique_z = np.unique(z[energy > 0])
        if unique_z.size == n_layers:
            # z has exactly one value per layer — use lookup
            z_to_lyr = {v: i for i, v in enumerate(np.sort(unique_z))}
            layer = np.vectorize(z_to_lyr.get)(z, -1).astype(int)
        else:
            # bin continuously
            edges = np.linspace(z[energy > 0].min(), z[energy > 0].max(), n_layers + 1)
            layer = (np.digitize(z, edges) - 1).clip(0, n_layers - 1)
            layer[energy == 0] = -1  # mark padding as invalid

        # (n_events, n_layers) — count hits per event per layer
        hits_per_layer = np.zeros((n_events, n_layers), dtype=np.int64)
        for lyr in range(n_layers):
            hits_per_layer[:, lyr] = ((layer == lyr) & (energy > 0)).sum(axis=1)

        # per-layer flat arrays for energy / profiles
        energy_per_layer = []
        mean_r_per_layer = np.zeros(n_layers)
        total_e_per_r = np.zeros(len(radial_bins))  # optional: mean energy per unit radius
        counts_per_layer = np.zeros(n_layers)
        sigma_r_per_layer = np.zeros(n_layers)

        for lyr in range(n_layers):
            mask = (layer == lyr) & (energy > 0)
            e_lyr = energy[mask]
            r_lyr = r[mask]
            energy_per_layer.append(e_lyr)
            counts_per_layer[lyr] = mask.sum()
            if e_lyr.size > 0:
                mu_r = np.average(r_lyr, weights=e_lyr)
                mean_r_per_layer[lyr] = mu_r
                sigma_r_per_layer[lyr] = np.sqrt(np.average((r_lyr - mu_r) ** 2, weights=e_lyr))

        # computing total e per radial bin
        for j in range(len(radial_bins) - 1):
            mask = (r >= radial_bins[j]) & (r < radial_bins[j + 1] + 1) & (energy > 0)
            if np.any(mask):
                total_e_per_r[j] = energy[mask].mean()

        return (
            hits_per_layer,  # [1]  (n_events, n_layers)
            energy_per_layer,  # [2]  list of 1-D arrays
            mean_r_per_layer,  # [3]  (n_layers,)
            counts_per_layer,  # [4]  (n_layers,)
            sigma_r_per_layer,  # [5]  (n_layers,)
            total_e_per_r,  # [6]  (100,) optional radial profile
        )

    # --------------------------------------------------
    # Color helpers
    # --------------------------------------------------
    def random_colors(n):
        cmap = plt.cm.get_cmap("tab20" if n <= 20 else "turbo")
        return [cmap(i / max(n - 1, 1)) for i in range(n)]

    extra_plot_kwargs = {"histtype": "stepfilled", "alpha": 0.3, "linewidth": 1}

    # --------------------------------------------------
    # Load reference datasets
    # --------------------------------------------------
    extra_records = []
    for path, label in extra_datasets:
        try:
            data = load_layer_data(path)
        except Exception as exc:
            print(f"[warn] Could not load {path}: {exc}")
            continue
        extra_records.append((label, *data))

    # --------------------------------------------------
    # Scan hdbscan configurations
    # --------------------------------------------------
    ms_toskip = {2, 3, 4, 5, 8, 12, 20, 40, 60}
    mcs_toskip = {3, 5, 8, 10, 12, 15, 20, 25, 40, 60}

    hdb_records = []
    hdb_ms_list = []
    for folder in sorted(root_dir.glob("hdbscan_mcs*_ms*")):
        match = re.search(r"mcs(\d+)_ms(\d+)(?:_eps([\d.]+))?", folder.name)
        if match is None:
            continue
        mcs = int(match.group(1))
        ms = int(match.group(2))
        epsilon = float(match.group(3)) if match.group(3) is not None else 0.0
        if epsilon != 0:
            continue
        if mcs in mcs_toskip and ms in ms_toskip:
            continue
        h5_file = folder / h5_name
        if not h5_file.exists():
            continue
        try:
            data = load_layer_data(h5_file)
        except Exception as exc:
            print(f"[warn] Could not load {h5_file}: {exc}")
            continue
        label = f"ms={ms}"
        hdb_records.append((label, *data))
        hdb_ms_list.append(ms)

    if not hdb_records and not extra_records:
        print("No data found; skipping per-layer plots.")
        return

    paired = sorted(zip(hdb_ms_list, hdb_records), key=lambda x: x[0])
    hdb_ms_list, hdb_records = ([x[0] for x in paired], [x[1] for x in paired]) if paired else ([], [])
    all_ms = sorted(set(hdb_ms_list))
    hdb_colors = [_ms_color(ms, all_ms) for ms in hdb_ms_list]
    extra_colors = ["dimgrey", "silver"]
    layers = np.arange(n_layers)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.8, label=rec[0]) for rec, c in zip(hdb_records, hdb_colors)
    ] + [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5, hatch="//", label=rec[0]) for rec, c in zip(extra_records, extra_colors)]

    # ================================================================
    # Figure 1 — distribution of hits per event, per layer
    #            each sub-plot: x = #hits in that layer, y = density
    # ================================================================
    ncols = min(5, n_layers_to_plot)
    nrows = int(np.ceil(n_layers_to_plot / ncols))

    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5), squeeze=False)
    fig1.suptitle(f"Hits per layer distribution ({n_layers_to_plot} sampled layers), mcs={mcs}", fontsize=12)

    for plot_idx, lyr in enumerate(sampled_layers):
        ax = axes1[plot_idx // ncols][plot_idx % ncols]

        # collect per-event hit counts for this layer from all datasets
        all_counts = np.concatenate([rec[1][:, lyr] for rec in hdb_records + extra_records])
        if all_counts.max() == 0:
            ax.set_title(f"Layer {lyr} (empty)", fontsize=8, pad=2)
            ax.tick_params(labelsize=6)
            continue

        # bins = np.arange(0, all_counts.max() + 2) - 0.5  # integer bins
        # bins = np.linspace(0, all_counts.max(), 30) - 0.5  # alternative: uniform bins
        bins = np.logspace(-0.5, np.log10(all_counts.max() + 0.5), 30) - 0.5  # alternative: log-spaced bins
        for rec, color in zip(extra_records, extra_colors):
            counts = rec[1][:, lyr]
            ax.hist(counts, bins=bins, density=True, color=color, label=rec[0], **extra_plot_kwargs)

        for rec, color in zip(hdb_records, hdb_colors):
            counts = rec[1][:, lyr]
            ax.hist(counts, bins=bins, density=True, histtype="step", linewidth=1.5, color=color, label=rec[0])

        ax.set_title(f"Layer {lyr}", fontsize=8, pad=2)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.tick_params(labelsize=6)
        ax.set_xlabel("Hits / event", fontsize=6)
        ax.set_ylabel("Density", fontsize=6)

    for plot_idx in range(n_layers_to_plot, nrows * ncols):
        axes1[plot_idx // ncols][plot_idx % ncols].set_visible(False)

    fig1.legend(
        handles=legend_handles, loc="lower center", ncol=min(6, len(legend_handles)), fontsize=7, bbox_to_anchor=(0.5, 0.0)
    )
    fig1.tight_layout(rect=[0, 0.06, 1, 0.97])
    out1 = plot_dir / "hits_per_layer_overlay.png"
    fig1.savefig(out1, dpi=150)
    plt.close(fig1)
    print(f"saved {out1}")

    # ================================================================
    # Figure 2 — energy distribution per layer (log scale)
    # ================================================================
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5), squeeze=False)
    fig2.suptitle(f"Hit energy per layer ({n_layers_to_plot} sampled layers)", fontsize=12)

    for plot_idx, lyr in enumerate(sampled_layers):
        ax = axes2[plot_idx // ncols][plot_idx % ncols]

        arrays = [rec[2][lyr] for rec in hdb_records + extra_records if rec[2][lyr].size > 0]
        if not arrays:
            ax.set_title(f"Layer {lyr}", fontsize=8, pad=2)
            ax.tick_params(labelsize=6)
            continue
        all_e = np.concatenate(arrays)

        lo = max(all_e.min(), 1e-20)
        hi = all_e.max()
        if lo >= hi:
            lo = hi * 0.1
        bins = np.logspace(np.log10(lo), np.log10(hi), 25)

        for rec, color in zip(extra_records, extra_colors):
            vals = rec[2][lyr]
            if vals.size:
                ax.hist(vals, bins=bins, density=True, color=color, label=rec[0], **extra_plot_kwargs)

        for rec, color in zip(hdb_records, hdb_colors):
            vals = rec[2][lyr]
            if vals.size:
                ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.5, color=color, label=rec[0])

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Layer {lyr}", fontsize=8, pad=2)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("Energy", fontsize=6)
        ax.set_ylabel("Density", fontsize=6)

    for plot_idx in range(n_layers_to_plot, nrows * ncols):
        axes2[plot_idx // ncols][plot_idx % ncols].set_visible(False)

    fig2.legend(
        handles=legend_handles, loc="lower center", ncol=min(6, len(legend_handles)), fontsize=7, bbox_to_anchor=(0.5, 0.0)
    )
    fig2.tight_layout(rect=[0, 0.06, 1, 0.97])
    out2 = plot_dir / "energy_per_layer_overlay.png"
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    print(f"saved {out2}")

    # ================================================================
    # Figure 3 — radial profile + total energy vs layer (all 78 layers)
    # ================================================================
    fig3, (ax_r, ax_e) = plt.subplots(1, 2, figsize=(13, 5))
    fig3.suptitle("Radial profile & total energy vs layer", fontsize=12)

    for rec, color in zip(extra_records, extra_colors):
        kw = dict(color=color, linewidth=2, alpha=0.6, linestyle="--", label=rec[0])
        ax_r.plot(radial_bins, rec[6], **kw)
        ax_e.plot(layers, rec[4], **kw)

    for rec, color in zip(hdb_records, hdb_colors):
        kw = dict(color=color, linewidth=1.5, label=rec[0])
        ax_r.plot(radial_bins, rec[6], **kw)
        ax_e.plot(layers, rec[4], **kw)

    ax_r.set_xlabel("Radius [mm]")
    ax_r.set_ylabel("Total e ")
    ax_r.set_title("Radial profile  ⟨r⟩")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(fontsize=7, ncol=2)

    ax_e.set_xlabel("Layer index")
    ax_e.set_ylabel("Counts per layer")
    ax_e.set_title("Longitudinal counts profile")
    ax_e.grid(True, alpha=0.3)
    ax_e.legend(fontsize=7, ncol=2)

    fig3.tight_layout()
    out3 = plot_dir / "radial_longitudinal_profiles.png"
    fig3.savefig(out3, dpi=150)
    plt.close(fig3)
    print(f"saved {out3}")

    # ================================================================
    # Figure 4 — σ_r vs layer (all 78 layers)
    # ================================================================
    fig4, ax_sr = plt.subplots(figsize=(8, 5))
    fig4.suptitle("Radial second moment σ_r vs layer", fontsize=12)

    for rec, color in zip(extra_records, extra_colors):
        ax_sr.plot(layers, rec[5], color=color, linewidth=2, alpha=0.6, linestyle="--", label=rec[0])

    for rec, color in zip(hdb_records, hdb_colors):
        ax_sr.plot(layers, rec[5], color=color, linewidth=1.5, label=rec[0])

    ax_sr.set_xlabel("Layer index")
    ax_sr.set_ylabel("σ_r  [mm]")
    ax_sr.set_yscale("log")
    ax_sr.set_title("Radial second moment  σ_r")
    ax_sr.grid(True, alpha=0.3)
    ax_sr.legend(fontsize=7, ncol=2)

    fig4.tight_layout()
    out4 = plot_dir / "second_momenta.png"
    fig4.savefig(out4, dpi=150)
    plt.close(fig4)
    print(f"saved {out4}")


def plot_h5_overlay_histograms(
    root_dir,
    h5_name="compressed_hdbscan.h5",
):
    root_dir = Path(root_dir)
    plot_dir = root_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    hits_data = []
    energy_data = []

    # --------------------------------------------------
    # Load the two reference datasets
    # --------------------------------------------------
    extra_datasets = [
        (
            "/eos/user/m/mamozzan/step2point/outputs/pipeline2_merge_within_cell/test_small/compressed_merge_within_cell.h5",
            "merge within cell",
        ),
        (
            "/eos/user/m/mamozzan/step2point/outputs/pipeline2_identity/test_small/compressed_identity.h5",
            "identity",
        ),
    ]

    extra_plot_kwargs = {"histtype": "stepfilled", "alpha": 0.3, "linewidth": 1}

    extra_hits = []
    extra_energy = []

    for path, label in extra_datasets:
        with h5py.File(path, "r") as f:
            steps = f["steps"]
            energy = steps["energy"][:]
            event_id = steps["event_id"][:]

        _, inverse = np.unique(event_id, return_inverse=True)
        hits_per_event = np.bincount(inverse)

        extra_hits.append((hits_per_event, label))
        extra_energy.append((energy[energy > 0], label))

    # --------------------------------------------------
    # Read hdbscan configurations
    # --------------------------------------------------
    ms_toskip = [2, 3, 4, 5, 8, 12, 20, 40, 60]
    mcs_toskip = [3, 5, 8, 10, 12, 15, 20, 25, 40, 60]

    for folder in sorted(root_dir.glob("hdbscan_mcs*_ms*")):
        match = re.search(r"mcs(\d+)_ms(\d+)(?:_eps([\d.]+))?", folder.name)
        if match is None:
            continue
        mcs = int(match.group(1))
        ms = int(match.group(2))
        epsilon = float(match.group(3)) if match.group(3) is not None else 0.0
        if epsilon != 0:
            continue
        if mcs in mcs_toskip:
            if ms in ms_toskip:
                continue
        h5_file = folder / h5_name
        if not h5_file.exists():
            continue
        with h5py.File(h5_file, "r") as f:
            steps = f["steps"]
            energy = steps["energy"][:]
            event_id = steps["event_id"][:]

        _, inverse = np.unique(event_id, return_inverse=True)
        hits_per_event = np.bincount(inverse)
        label = f"ms={ms}"
        hits_data.append((hits_per_event, label, ms))
        energy_data.append((energy[energy > 0], label, ms))

    hits_data.sort(key=lambda x: x[2])
    energy_data.sort(key=lambda x: x[2])
    all_ms = sorted({ms for _, _, ms in hits_data})
    hits_data = [(vals, label, _ms_color(ms, all_ms)) for vals, label, ms in hits_data]
    energy_data = [(vals, label, _ms_color(ms, all_ms)) for vals, label, ms in energy_data]

    colors_extra = ["dimgrey", "silver"]
    extra_hits = [(vals, label, color) for (vals, label), color in zip(extra_hits, colors_extra)]
    extra_energy = [(vals, label, color) for (vals, label), color in zip(extra_energy, colors_extra)]

    # --------------------------------------------------
    # Hits per event
    # --------------------------------------------------
    all_hits_max = max(
        max((v.max() for v, _, _ in hits_data), default=0),
        max((v.max() for v, _, _ in extra_hits), default=0),
    )
    b_ = np.linspace(0, all_hits_max, 40)

    fig, ax = plt.subplots(figsize=(8, 6))

    # filled reference datasets first (so they sit behind the step lines)
    for values, label, color in extra_hits:
        ax.hist(values, bins=b_, density=True, label=label, color=color, **extra_plot_kwargs)

    for values, label, color in hits_data:
        ax.hist(values, bins=b_, density=True, histtype="step", linewidth=2, label=label, color=color)

    ax.set_xlabel("Points per event")
    ax.set_ylabel("Density")
    ax.set_title("Points per event")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_file = plot_dir / "points_per_event_overlay.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"saved {out_file}")

    # --------------------------------------------------
    # Hit energy (log scale)
    # --------------------------------------------------
    all_energy = np.concatenate([vals for vals, _, _ in energy_data] + [vals for vals, _, _ in extra_energy])
    bins = np.logspace(np.log10(5e-14), np.log10(all_energy.max()), 40)

    fig, ax = plt.subplots(figsize=(8, 6))

    for values, label, color in extra_energy:
        ax.hist(values, bins=bins, density=True, label=label, color=color, **extra_plot_kwargs)

    for values, label, color in energy_data:
        ax.hist(values, bins=bins, density=True, histtype="step", linewidth=2, label=label, color=color)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Point energy")
    ax.set_ylabel("Density")
    ax.set_ylim([1e-2, 3e8])
    ax.set_title("Point energy distribution")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_file = plot_dir / "hit_energy_overlay.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"saved {out_file}")


def _draw_heatmap(ax, df, metric, title):
    mcs_vals = sorted(df["mcs"].unique())
    ms_vals = sorted(df["ms"].unique())
    pivot = df.pivot(index="ms", columns="mcs", values=metric)
    pivot = pivot.reindex(index=ms_vals, columns=mcs_vals)
    Z = pivot.values.astype(float)

    im = ax.pcolormesh(
        np.arange(len(mcs_vals) + 1),
        np.arange(len(ms_vals) + 1),
        Z,
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(mcs_vals)) + 0.5)
    ax.set_xticklabels(mcs_vals, fontsize=8)
    ax.set_yticks(np.arange(len(ms_vals)) + 0.5)
    ax.set_yticklabels(ms_vals, fontsize=8)
    for i in range(len(ms_vals)):
        for j in range(len(mcs_vals)):
            val = Z[i, j]
            if not np.isnan(val):
                ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white")
    ax.set_xlabel("min_cluster_size")
    ax.set_ylabel("min_samples")
    ax.set_title(title)


def plot_compression_ratios(
    root_dir,
    txt_name="compression_summary_hdbscan.txt",
):
    root_dir = Path(root_dir)

    rows = []

    for folder in sorted(root_dir.glob("hdbscan_mcs*_ms*")):
        # The (?:_eps([\d.]+))? part makes the epsilon suffix optional
        match = re.search(r"mcs(\d+)_ms(\d+)(?:_eps([\d.]+))?", folder.name)
        if match is None:
            continue

        mcs = int(match.group(1))
        ms = int(match.group(2))
        epsilon = float(match.group(3)) if match.group(3) is not None else 0.0
        if epsilon != 0:
            continue
        if ms == 4:
            continue
        txt_file = folder / txt_name
        if not txt_file.exists():
            continue

        row = {"mcs": mcs, "ms": ms, "epsilon": epsilon}

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
    all_ms = sorted(df["ms"].unique().tolist())

    metrics = ["mean_compression_ratio", "total_compression_ratio"]
    titles = ["Mean compression ratio", "Total compression ratio"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ── top row: line plots ──────────────────────────────────────────────
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
        for ax, key in zip(axes[0], metrics):
            if key in row:
                ax.axhline(row[key], color="k", linestyle="--", label="merge within cell")

    for ax, metric, title in zip(axes[0], metrics, titles):
        for ms, grp in sorted(df.groupby("ms")):
            grp = grp.sort_values("mcs")
            ax.plot(grp["mcs"], grp[metric], marker="o", linewidth=2, color=_ms_color(ms, all_ms), label=f"ms={ms}")
        ax.set_xlabel("min_cluster_size")
        ax.set_ylabel("compression ratio")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[0, 0].legend(title="min_samples", frameon=False)

    # ── bottom row: heatmaps ─────────────────────────────────────────────
    for ax, metric, title in zip(axes[1], metrics, titles):
        _draw_heatmap(ax, df, metric, f"{title} (heatmap)")

    fig.tight_layout()
    out_file = root_dir / "plots/compression_ratios.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_file}")


def plot_clusters_per_cell_histogram(
    root_dir,
    h5_name="input_cc3.h5",
    n_layers=30,
):
    root_dir = Path(root_dir)
    plot_dir = root_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Spatial cell bins — 5 mm cells
    cell_bins = np.linspace(-450, 450, int(900 / 5) + 1)
    n_bins = len(cell_bins) - 1

    # --------------------------------------------------
    # Loader: for each event×layer, bin cluster centroids into (x, y) cells
    # and return the flat distribution of clusters-per-occupied-voxel.
    def load_clusters_per_cell(path):
        with h5py.File(path, "r") as f:
            data = f["events"][:]  # (n_events, n_pts, 4)
        counts = []
        for shower in data:
            x, y, z, e = shower[:, 0], shower[:, 1], shower[:, 2], shower[:, 3]
            valid = e > 0
            if not valid.any():
                continue
            layer_idx = np.floor(z[valid]).astype(int).clip(0, n_layers - 1)
            for lyr in range(n_layers):
                in_lyr = layer_idx == lyr
                if not in_lyr.any():
                    continue
                cx = np.clip(np.digitize(x[valid][in_lyr], cell_bins) - 1, 0, n_bins - 1)
                cy = np.clip(np.digitize(y[valid][in_lyr], cell_bins) - 1, 0, n_bins - 1)
                voxel_ids = cx * n_bins + cy
                _, cnts = np.unique(voxel_ids, return_counts=True)
                counts.extend(cnts.tolist())
        return np.array(counts, dtype=int)

    # --------------------------------------------------
    # Load all hdbscan configs; store keyed by (mcs, ms, eps)
    # eps=None  → no epsilon argument (base config)
    # eps=float → explicit cluster-selection epsilon
    hdb_data = {}
    for folder in sorted(root_dir.glob("hdbscan_mcs*_ms*")):
        match = re.search(r"mcs(\d+)_ms(\d+)(?:_eps([\d.]+))?$", folder.name)
        if match is None:
            continue
        mcs = int(match.group(1))
        ms = int(match.group(2))
        eps = float(match.group(3)) if match.group(3) is not None else None
        h5_file = folder / h5_name
        if not h5_file.exists():
            continue
        try:
            counts = load_clusters_per_cell(h5_file)
        except Exception as exc:
            print(f"[warn] Could not load {h5_file}: {exc}")
            continue
        hdb_data[(mcs, ms, eps)] = counts

    if not hdb_data:
        print("No data found for cluster-per-cell histograms; skipping.")
        return

    # --------------------------------------------------
    # Shared helpers

    def sort_by_clusters(records):
        """Sort descending by total cluster count (most clusters first)."""
        return sorted(records, key=lambda x: int(x[1].sum()), reverse=True)

    def make_scan_figure(records, title, out_file):
        """
        records : [(label, counts), …] already sorted most→fewest clusters.
        Colors shift sequentially from 'plasma' — first entry (most clusters)
        gets the brightest colour; last entry gets the darkest.
        """
        if not records:
            return
        all_c = np.concatenate([c for _, c in records])
        if len(all_c) == 0:
            return
        max_count = int(all_c.max())
        bins = np.linspace(0, max_count, 30)

        n = len(records)
        cmap = plt.cm.get_cmap("plasma")
        colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.suptitle(title, fontsize=12)
        for (label, counts), color in zip(records, colors):
            ax.hist(counts, bins=bins, density=True, histtype="step", linewidth=1.5, color=color, label=label)

        ax.set_xlabel("Clusters per (layer, x-bin, y-bin) voxel")
        ax.set_ylabel("Density")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
        print(f"saved {out_file}")

    # --------------------------------------------------
    # Figure 1 — vary ms (fix mcs to the value with the most ms variants)
    mcs_to_ms = {}
    for mcs, ms, eps in hdb_data:
        if eps is None:
            mcs_to_ms.setdefault(mcs, set()).add(ms)
    if mcs_to_ms:
        fix_mcs = max(mcs_to_ms, key=lambda k: (len(mcs_to_ms[k]), k))
        vary_ms = sort_by_clusters(
            [(f"ms={ms}", hdb_data[(fix_mcs, ms, None)]) for ms in sorted(mcs_to_ms[fix_mcs]) if (fix_mcs, ms, None) in hdb_data]
        )
        make_scan_figure(
            vary_ms,
            f"Clusters per voxel — varying ms  (mcs={fix_mcs} fixed)",
            plot_dir / "clusters_per_cell_vary_ms.png",
        )

    # --------------------------------------------------
    # Figure 2 — vary mcs (fix ms to the value with the most mcs variants)
    ms_to_mcs = {}
    for mcs, ms, eps in hdb_data:
        if eps is None:
            ms_to_mcs.setdefault(ms, set()).add(mcs)
    if ms_to_mcs:
        fix_ms = max(ms_to_mcs, key=lambda k: (len(ms_to_mcs[k]), -k))
        vary_mcs = sort_by_clusters(
            [
                (f"mcs={mcs}", hdb_data[(mcs, fix_ms, None)])
                for mcs in sorted(ms_to_mcs[fix_ms])
                if (mcs, fix_ms, None) in hdb_data
            ]
        )
        make_scan_figure(
            vary_mcs,
            f"Clusters per voxel — varying mcs  (ms={fix_ms} fixed)",
            plot_dir / "clusters_per_cell_vary_mcs.png",
        )

    # --------------------------------------------------
    # Figure 3 — vary epsilon (ms=8 only; include base eps=None as reference)
    eps_records = []
    for mcs, ms, eps in sorted(hdb_data, key=lambda k: (k[0], k[1], k[2] if k[2] is not None else -1.0)):
        if ms != 8:
            continue
        if eps is None:
            label = f"mcs={mcs}, eps=default"
        else:
            label = f"mcs={mcs}, eps={eps}"
        eps_records.append((label, hdb_data[(mcs, ms, eps)]))
    if eps_records:
        # separate eps=default (dashed) from explicit eps (solid) entries,
        # each group sorted by cluster count
        base_recs = sort_by_clusters([(lbl, c) for lbl, c in eps_records if "default" in lbl])
        scan_recs = sort_by_clusters([(lbl, c) for lbl, c in eps_records if "default" not in lbl])

        all_eps_counts = np.concatenate([c for _, c in eps_records])
        max_count = int(all_eps_counts.max())
        bins = np.linspace(0, max_count, 30)

        n_scan = len(scan_recs)
        cmap = plt.cm.get_cmap("plasma")
        scan_colors = [cmap(i / max(n_scan - 1, 1)) for i in range(n_scan)]
        base_colors = ["dimgrey", "silver"]

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.suptitle("Clusters per voxel — varying epsilon  (ms=8)", fontsize=12)
        for (label, counts), color in zip(base_recs, base_colors):
            ax.hist(counts, bins=bins, density=True, histtype="step", linewidth=2.0, linestyle="--", color=color, label=label)
        for (label, counts), color in zip(scan_recs, scan_colors):
            ax.hist(counts, bins=bins, density=True, histtype="step", linewidth=1.5, color=color, label=label)
        ax.set_xlabel("Clusters per (layer, x-bin, y-bin) voxel")
        ax.set_ylabel("Density")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        fig.tight_layout()
        out_eps = plot_dir / "clusters_per_cell_vary_eps.png"
        fig.savefig(out_eps, dpi=150)
        plt.close(fig)
        print(f"saved {out_eps}")


def plot_clusters_per_cell(
    root_dir,
    h5_name="input_cc3.h5",
    n_layers=30,
):
    root_dir = Path(root_dir)
    plot_dir = root_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    cell_bins = np.linspace(-450, 450, int(900 / 5) + 1)
    n_bins = len(cell_bins) - 1

    def compute_metrics(path):
        with h5py.File(path, "r") as f:
            data = f["events"][:]  # (n_events, n_pts, 4)
        counts = []
        for shower in data:
            x, y, z, e = shower[:, 0], shower[:, 1], shower[:, 2], shower[:, 3]
            valid = e > 0
            if not valid.any():
                continue
            layer_idx = np.floor(z[valid]).astype(int).clip(0, n_layers - 1)
            for lyr in range(n_layers):
                in_lyr = layer_idx == lyr
                if not in_lyr.any():
                    continue
                cx = np.clip(np.digitize(x[valid][in_lyr], cell_bins) - 1, 0, n_bins - 1)
                cy = np.clip(np.digitize(y[valid][in_lyr], cell_bins) - 1, 0, n_bins - 1)
                voxel_ids = cx * n_bins + cy
                _, cnts = np.unique(voxel_ids, return_counts=True)
                counts.extend(cnts.tolist())
        counts = np.array(counts, dtype=int)
        if counts.size == 0:
            return np.nan, np.nan
        return float(counts.max()), float((counts > 5).mean())

    ref_datasets = [
        (
            "/eos/user/m/mamozzan/step2point/outputs/pipeline2_merge_within_cell/test_small/input_cc3.h5",
            "merge within cell",
        ),
    ]

    rows = []
    for folder in sorted(root_dir.glob("hdbscan_mcs*_ms*")):
        match = re.search(r"mcs(\d+)_ms(\d+)(?:_eps([\d.]+))?", folder.name)
        if match is None:
            continue
        mcs = int(match.group(1))
        ms = int(match.group(2))
        epsilon = float(match.group(3)) if match.group(3) is not None else 0.0
        if epsilon != 0:
            continue
        h5_file = folder / h5_name
        if not h5_file.exists():
            continue
        try:
            mean_cpc, frac_overlap = compute_metrics(h5_file)
        except Exception as exc:
            print(f"[warn] Could not load {h5_file}: {exc}")
            continue
        rows.append({"mcs": mcs, "ms": ms, "epsilon": epsilon, "max_clusters_per_cell": mean_cpc, "frac_overlap": frac_overlap})

    if not rows:
        print("No data found for clusters-per-cell plot.")
        return

    df = pd.DataFrame(rows)
    all_ms = sorted(df["ms"].unique().tolist())

    metrics = ["max_clusters_per_cell", "frac_overlap"]
    titles = ["Max clusters per occupied cell", "Fraction of cells with >5 clusters"]
    ylabels = ["Max clusters per cell", "Fraction overlapping cells"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ── top row: line plots ──────────────────────────────────────────────
    ref_styles = ["--", ":"]
    ref_colors = ["k", "grey"]
    for (path, label), color, ls in zip(ref_datasets, ref_colors, ref_styles):
        try:
            max_cpc, frac_overlap = compute_metrics(path)
            for ax, val in zip(axes[0], [max_cpc, frac_overlap]):
                ax.axhline(val, color=color, linestyle=ls, linewidth=1.5, label=label)
        except Exception as exc:
            print(f"[warn] Could not load reference {path}: {exc}")

    for ax, metric, title, ylabel in zip(axes[0], metrics, titles, ylabels):
        for ms, grp in sorted(df.groupby("ms")):
            grp = grp.sort_values("mcs")
            ax.plot(grp["mcs"], grp[metric], marker="o", linewidth=2, color=_ms_color(ms, all_ms), label=f"ms={ms}")
        ax.set_xlabel("min_cluster_size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[0, 0].legend(title="min_samples", frameon=False)

    # ── bottom row: heatmaps ─────────────────────────────────────────────
    for ax, metric, title in zip(axes[1], metrics, titles):
        _draw_heatmap(ax, df, metric, f"{title} (heatmap)")

    fig.tight_layout()
    out_file = plot_dir / "clusters_per_cell.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_file}")


def make_all_plots():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n── Generating plots ──────────────────────────────")

    # line plots
    plot_compression_ratios(BASE_OUTPUT, txt_name="compression_summary_hdbscan.txt")
    plot_clusters_per_cell(BASE_OUTPUT, h5_name="input_cc3.h5", n_layers=30)
    plot_h5_overlay_histograms(BASE_OUTPUT, h5_name="compressed_hdbscan.h5")
    plot_per_layer_histograms(BASE_OUTPUT, h5_name="input_cc3.h5", n_layers=30, n_layers_to_plot=10)
    plot_clusters_per_cell_histogram(BASE_OUTPUT, h5_name="input_cc3.h5")
    print(f"\nDone. Plots saved in: {PLOT_DIR.resolve()}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────


def main():
    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

    combos = [
        (mcs, ms, epsilon)
        for mcs, ms, epsilon in itertools.product(MIN_CLUSTER_SIZES, MIN_SAMPLES_LIST, EPSILON)
        if ms <= mcs  # skip invalid HDBSCAN combos
    ]

    print(f"── Running {len(combos)} configurations ──────────────────────")
    rows = []
    for mcs, ms, epsilon in combos:
        out_dir = run_pipeline(mcs, ms, epsilon)
        row = parse_metrics(out_dir, mcs, ms, epsilon)
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
