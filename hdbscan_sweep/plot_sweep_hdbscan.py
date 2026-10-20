"""
Plots of an HDBSCAN sweep made by sweep_hdbscan.py, read from its output folder:
compression ratio vs (min_cluster_size, min_samples), the points-per-shower
distribution, and the plots of the step2point/CaloClouds-3
paper (mean compression ratio vs m_cs; point energy and, given the cell-ID
layer positions, the longitudinal profile at one m_cs), with the
reference runs if there are any.
sweep_hdbscan.py calls this at the end; run it alone to remake the plots:

    python hdbscan_sweep/plot_sweep_hdbscan.py [sweep_dir] [--preset ild]
"""

import argparse
import os
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]  # step2point repo root

# Set by make_all_plots()
REFERENCE_DIR = None

# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────


# Shared color map: same ms value → same color in every plot
def _ms_color(ms: int, all_ms: list) -> tuple:
    cmap = plt.cm.get_cmap("tab10" if len(all_ms) <= 10 else "tab20")
    idx = sorted(all_ms).index(ms)
    return cmap(idx / max(len(all_ms) - 1, 1))


def plot_points_per_shower(sweep_dir, out_file, mcs=20):
    """Points-per-shower distribution of the runs at one m_cs, with identity as grey band and the references."""
    runs = _sweep_runs(sweep_dir, mcs)
    if not runs:
        return

    def points_per_shower(path):
        with h5py.File(path, "r") as f:
            return np.unique(f["steps"]["event_id"][:], return_counts=True)[1]

    sweep = [(ms, points_per_shower(f / "compressed_hdbscan.h5")) for _, ms, f in runs]
    refs = {}
    if REFERENCE_DIR is not None:
        for algo in ["identity", *REF_COLORS]:
            h5 = Path(REFERENCE_DIR, algo, f"compressed_{algo}.h5")
            if h5.exists():
                refs[algo] = points_per_shower(h5)
    all_ms = [ms for ms, _ in sweep]
    counts = [*refs.values(), *(n for _, n in sweep)]
    bins = np.logspace(np.log10(min(n.min() for n in counts)), np.log10(max(n.max() for n in counts)), 40)
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        if "identity" in refs:
            ax.hist(
                refs["identity"],
                bins=bins,
                histtype="stepfilled",
                color="#eeede9",
                edgecolor="#c9c8c2",
                label="identity (no merging)",
            )
        if "merge_within_cell" in refs:
            ax.hist(
                refs["merge_within_cell"],
                bins=bins,
                histtype="stepfilled",
                color="#56b4e9",
                alpha=0.45,
                edgecolor=REF_COLORS["merge_within_cell"],
                lw=1.0,
                label="merge within cell",
            )
        if "merge_within_regular_subcell" in refs:
            ax.hist(
                refs["merge_within_regular_subcell"],
                bins=bins,
                histtype="step",
                ls="--",
                lw=1.6,
                color=REF_COLORS["merge_within_regular_subcell"],
                label="merge within regular subcell",
            )
        for ms, n in sweep:
            ax.hist(n, bins=bins, histtype="step", lw=1.6, color=_paper_color(ms, all_ms), label=f"$m_s = {ms}$")
        ax.set_xscale("log")
        ax.set_xlabel("points per shower")
        ax.set_ylabel("showers / bin")
        ax.set_title(f"$m_{{cs}} = {mcs}$")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
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
    ref_summaries = (
        []
        if REFERENCE_DIR is None
        else [
            (
                f"{REFERENCE_DIR}/merge_within_cell/compression_summary_merge_within_cell.txt",
                "merge within cell",
                "k",
            ),
            (
                f"{REFERENCE_DIR}/merge_within_regular_subcell/compression_summary_merge_within_regular_subcell.txt",
                "merge within regular subcell",
                "saddlebrown",
            ),
        ]
    )
    for txt_path, ref_label, ref_color in ref_summaries:
        if not Path(txt_path).exists():
            continue
        row = {}
        with open(txt_path) as f:
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
                ax.axhline(row[key], color=ref_color, linestyle="--", label=ref_label)

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


# ─────────────────────────────────────────────
# PAPER FIGURES (step2point/CaloClouds-3 paper, figures 2 and 3)
# ─────────────────────────────────────────────

REF_COLORS = {"merge_within_cell": "#0072b2", "merge_within_regular_subcell": "#e69f00"}
# m_s colours of the paper; other m_s values fall back to _ms_color
MS_COLORS = {2: "#9a9a92", 3: "#009e73", 5: "#a1215a", 8: "#cc79a7", 12: "#7a21dd", 20: "#455a64", 40: "#56b4e9", 60: "#0b0b0b"}
PAPER_STYLE = {"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 9, "axes.edgecolor": "#0b0b0b", "grid.color": "#e1e0d9"}


def _paper_color(ms, all_ms):
    return MS_COLORS.get(ms, _ms_color(ms, all_ms))


def _sweep_runs(sweep_dir, mcs=None):
    """(mcs, ms, folder) of every epsilon = 0 run in sweep_dir, optionally at one mcs."""
    runs = []
    for folder in sorted(Path(sweep_dir).glob("hdbscan_mcs*_ms*")):
        m = re.fullmatch(r"hdbscan_mcs(\d+)_ms(\d+)", folder.name)
        if m and (mcs is None or int(m.group(1)) == mcs):
            runs.append((int(m.group(1)), int(m.group(2)), folder))
    runs.sort()  # numeric (mcs, ms) order, not folder-name order
    if mcs is not None and not runs:
        print(f"  [skip] no runs at m_cs = {mcs} in {sweep_dir}; pick another with --profile-mcs")
    return runs


def _read_summary(path):
    row = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            try:
                row[k] = float(v)
            except ValueError:
                pass
    return row


def plot_mean_compression(sweep_dir, out_file):
    """Mean compression ratio vs m_cs, one line per m_s, references dashed; the axis above 0.4 is compressed."""
    rows = [
        {"mcs": mcs, "ms": ms, **_read_summary(f / "compression_summary_hdbscan.txt")}
        for mcs, ms, f in _sweep_runs(sweep_dir)
        if (f / "compression_summary_hdbscan.txt").exists()
    ]
    if not rows:
        return
    df = pd.DataFrame(rows)
    all_ms = sorted(df.ms.unique())
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        for ms in all_ms:
            g = df[df.ms == ms].sort_values("mcs")
            ax.plot(
                g.mcs, g.mean_compression_ratio, marker="o", ms=4, lw=1.6, color=_paper_color(ms, all_ms), label=f"$m_s = {ms}$"
            )
        for algo, color in REF_COLORS.items():
            txt = Path(REFERENCE_DIR or "", algo, f"compression_summary_{algo}.txt")
            if REFERENCE_DIR is not None and txt.exists():
                ratio = _read_summary(txt)["mean_compression_ratio"]
                ax.axhline(ratio, color=color, ls="--", lw=1.2, label=algo.replace("_", " "))
        # piecewise-linear y axis: the curves sit below ~0.3, so the range above Y_KNEE is squeezed
        y_knee, y_squeeze = 0.4, 0.25
        ax.set_yscale(
            "function",
            functions=(
                lambda y: np.where(y <= y_knee, y, y_knee + (y - y_knee) * y_squeeze),
                lambda u: np.where(u <= y_knee, u, y_knee + (u - y_knee) / y_squeeze),
            ),
        )
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, df.mcs.max() + 3)
        ax.set_xlabel(r"minimum cluster size $m_{cs}$")
        ax.set_ylabel(r"mean $N_\mathrm{after}/N_\mathrm{before}$")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"saved {out_file}")


def _point_energies(path):
    with h5py.File(path, "r") as f:
        e = f["steps"]["energy"][:]
    return e[e > 0]


def plot_point_energy(sweep_dir, out_file, mcs=20):
    """Point-energy distribution of the runs at one m_cs, with identity as grey band and the references."""
    runs = _sweep_runs(sweep_dir, mcs)
    if not runs:
        return
    sweep = [(ms, _point_energies(f / "compressed_hdbscan.h5")) for _, ms, f in runs]
    refs = {}
    if REFERENCE_DIR is not None:
        for algo in ["identity", *REF_COLORS]:
            h5 = Path(REFERENCE_DIR, algo, f"compressed_{algo}.h5")
            if h5.exists():
                refs[algo] = _point_energies(h5)
    all_ms = [ms for ms, _ in sweep]
    e_max = max(e.max() for e in [*refs.values(), *(e for _, e in sweep)])
    bins = np.logspace(-6, np.log10(e_max), 80)
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        if "identity" in refs:
            ax.hist(
                refs["identity"],
                bins=bins,
                histtype="stepfilled",
                color="#eeede9",
                edgecolor="#c9c8c2",
                label="identity (no merging)",
            )
        if "merge_within_cell" in refs:
            ax.hist(
                refs["merge_within_cell"],
                bins=bins,
                histtype="stepfilled",
                color="#56b4e9",
                alpha=0.45,
                edgecolor=REF_COLORS["merge_within_cell"],
                lw=1.0,
                label="merge within cell",
            )
        if "merge_within_regular_subcell" in refs:
            ax.hist(
                refs["merge_within_regular_subcell"],
                bins=bins,
                histtype="step",
                ls="--",
                lw=1.6,
                color=REF_COLORS["merge_within_regular_subcell"],
                label="merge within regular subcell",
            )
        for ms, e in sweep:
            ax.hist(e, bins=bins, histtype="step", lw=1.6, color=_paper_color(ms, all_ms), label=f"$m_s = {ms}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("point energy [GeV]")
        ax.set_ylabel("points / bin")
        ax.set_title(f"$m_{{cs}} = {mcs}$")
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"saved {out_file}")


def plot_longitudinal_profile(sweep_dir, out_file, layer_positions, mcs=20):
    """Mean points per layer of the runs at one m_cs. A point is in the last layer whose
    bottom (layer_positions, global y in mm, ascending) lies below its y; points below
    the first layer (other barrel segments) are left out."""
    layer_positions = np.asarray(layer_positions)
    n_layers = len(layer_positions)
    runs = _sweep_runs(sweep_dir, mcs)
    if not runs:
        return

    def profile(path):
        with h5py.File(path, "r") as f:
            st = f["steps"]
            layer = np.searchsorted(layer_positions, st["position"][:, 1], side="right") - 1
            n_showers = len(np.unique(st["event_id"][:]))
        layer = layer[(layer >= 0) & (layer < n_layers)]
        return np.bincount(layer, minlength=n_layers) / n_showers

    all_ms = [ms for _, ms, _ in runs]
    layers = np.arange(n_layers)
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        for algo, color in REF_COLORS.items():
            h5 = Path(REFERENCE_DIR or "", algo, f"compressed_{algo}.h5")
            if REFERENCE_DIR is not None and h5.exists():
                ax.plot(layers, profile(h5), color=color, ls="--", lw=1.6, label=algo.replace("_", " "))
        for _, ms, f in runs:
            ax.plot(layers, profile(f / "compressed_hdbscan.h5"), color=_paper_color(ms, all_ms), lw=1.6, label=f"$m_s = {ms}$")
        ax.set_xlabel("layer")
        ax.set_ylabel("mean points per layer")
        ax.set_xlim(layers[0], layers[-1])
        ax.set_ylim(bottom=0)
        ax.set_title(f"$m_{{cs}} = {mcs}$")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"saved {out_file}")


def make_all_plots(sweep_dir, reference_dir=None, mcs=20, layer_positions=None):
    """Make all plots of the sweep in sweep_dir into sweep_dir/plots/; the longitudinal
    profile only with layer_positions (bottom of each layer, global y in mm)."""
    global REFERENCE_DIR
    REFERENCE_DIR = reference_dir
    sweep_dir = Path(sweep_dir)
    (sweep_dir / "plots").mkdir(parents=True, exist_ok=True)
    print("\n── Generating plots ──────────────────────────────")
    plot_compression_ratios(sweep_dir, txt_name="compression_summary_hdbscan.txt")
    plot_points_per_shower(sweep_dir, sweep_dir / "plots/points_per_event_overlay.png", mcs)
    plot_mean_compression(sweep_dir, sweep_dir / "plots/mean_compression_ratio.png")
    plot_point_energy(sweep_dir, sweep_dir / "plots/point_energy.png", mcs)
    if layer_positions:
        plot_longitudinal_profile(sweep_dir, sweep_dir / "plots/longitudinal_profile.png", layer_positions, mcs)
    print(f"\nDone. Plots saved in: {(sweep_dir / 'plots').resolve()}")


def main():
    p = argparse.ArgumentParser(description="Plots of an HDBSCAN sweep made by sweep_hdbscan.py.")
    p.add_argument(
        "sweep_dir",
        nargs="?",
        type=Path,
        default=Path(os.environ.get("SWEEP_DIR", REPO_ROOT / "outputs/hdbscan_sweep")),
        help="sweep output folder (default: $SWEEP_DIR, else <repo>/outputs/hdbscan_sweep)",
    )
    p.add_argument(
        "--reference-dir",
        type=Path,
        help="reference runs, <dir>/<algorithm>/ (default: <sweep_dir>/references if it exists)",
    )
    p.add_argument("--preset", choices=["ild"], help="preset the sweep was run with")
    p.add_argument("--profile-mcs", type=int, default=20, help="m_cs of the per-run distribution plots (default 20)")
    p.add_argument(
        "--layer-positions",
        type=float,
        nargs="+",
        help="bottom of each layer along global y [mm], for the longitudinal profile (set by --preset ild)",
    )
    args = p.parse_args()
    if args.preset and not args.layer_positions:
        from sweep_hdbscan import PRESETS

        args.layer_positions = PRESETS[args.preset].get("layer_positions")
    reference_dir = args.reference_dir
    if reference_dir is None and (args.sweep_dir / "references").is_dir():
        reference_dir = args.sweep_dir / "references"
    make_all_plots(args.sweep_dir, reference_dir, args.profile_mcs, args.layer_positions)


if __name__ == "__main__":
    main()
