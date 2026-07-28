"""
Plot x, y, z, energy distributions comparing:
  - File 1 (Original HDF5): padded (n_events, max_hits, 7) format
  - File 2 (step2point):     flat format with event_id grouping

Usage:
    python plot_xyze.py
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
FILE1 = "../photons_h5/p22_th45-135_ph79-109_en5-130_seed0_ip.h5"
FILE2 = "outputs/pipeline_identity_0/compressed_identity.h5"

N_EVENTS = None  # set to an int (e.g. 100) to limit events for speed

# ── load File 1 ──────────────────────────────────────────────────────────────
print("Loading File 1 (Original)...")
with h5py.File(FILE1, "r") as f:
    events = f["events"][:N_EVENTS]  # (N, max_hits, 7)

mask1 = events[..., 3] != 0
x1 = events[..., 0][mask1].ravel()
y1 = events[..., 1][mask1].ravel()
z1 = events[..., 2][mask1].ravel()
e1 = events[..., 3][mask1].ravel()
print(f"  File 1 hits: {len(x1):,}")

# ── load File 2 ──────────────────────────────────────────────────────────────
print("Loading File 2 (step2point)...")
with h5py.File(FILE2, "r") as f:
    ev_ids = f["steps/event_id"][:]
    pos2 = f["steps/position"][:]
    e2 = f["steps/energy"][:]

if N_EVENTS is not None:
    sel = ev_ids < N_EVENTS
    pos2 = pos2[sel]
    e2 = e2[sel]

x2 = pos2[:, 0]
y2 = pos2[:, 1]
z2 = pos2[:, 2]
print(f"  File 2 hits: {len(x2):,}")

# ── plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.7,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
    }
)

BINS = 100
COLOR1 = "#2196F3"  # filled — Original
COLOR2 = "#E53935"  # line   — step2point

variables = [
    ("x  [mm]", x1, x2),
    ("y  [mm]", y1, y2),
    ("z  [mm]", z1, z2),
    ("Energy  [GeV]", e1, e2),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
fig.suptitle("Calorimeter Hit Distributions — Original vs step2point", fontsize=13, fontweight="bold")

for ax, (lbl, d1, d2) in zip(axes.flat, variables):
    lo = min(d1.min(), d2.min())
    hi = max(d1.max(), d2.max())
    edges = np.linspace(lo, hi, BINS + 1)

    # Original: filled histogram
    ax.hist(d1, bins=edges, density=True, color=COLOR1, alpha=0.4, label=f"Original  (n={len(d1):,})")
    ax.hist(d1, bins=edges, density=True, color=COLOR1, histtype="step", linewidth=1.2)

    # step2point: line only
    counts2, _ = np.histogram(d2, bins=edges, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ax.step(centres, counts2, where="mid", color=COLOR2, linewidth=1.6, label=f"step2point  (n={len(d2):,})")

    ax.set_xlabel(lbl)
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

out1 = "martina_test/hit_distributions_xyze.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")

# ── per-event total energy ────────────────────────────────────────────────────
print("Computing per-event total energies...")
with h5py.File(FILE1, "r") as f:
    ev_e1 = f["events"][:N_EVENTS, :, 3]
total_e1 = ev_e1.sum(axis=1)

with h5py.File(FILE2, "r") as f:
    ev_ids_all = f["steps/event_id"][:]
    e2_all = f["steps/energy"][:]

n_ev = len(total_e1)
total_e2 = np.array([e2_all[ev_ids_all == i].sum() for i in range(n_ev)])

fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
fig2.suptitle("Per-Event Total Deposited Energy", fontsize=13, fontweight="bold")

edges_e = np.linspace(min(total_e1.min(), total_e2.min()), max(total_e1.max(), total_e2.max()), 80)

# histogram
axes2[0].hist(total_e1, bins=edges_e, density=True, color=COLOR1, alpha=0.4, label="Original")
axes2[0].hist(total_e1, bins=edges_e, density=True, color=COLOR1, histtype="step", linewidth=1.2)
counts_e2, _ = np.histogram(total_e2, bins=edges_e, density=True)
centres_e = 0.5 * (edges_e[:-1] + edges_e[1:])
axes2[0].step(centres_e, counts_e2, where="mid", color=COLOR2, linewidth=1.6, label="step2point")
axes2[0].set_xlabel("Total Event Energy [GeV]")
axes2[0].set_ylabel("Density")
axes2[0].set_title("Distribution")
axes2[0].legend()

# scatter
axes2[1].scatter(total_e1, total_e2, s=8, alpha=0.5, color="#555555", linewidths=0)
lims = [min(total_e1.min(), total_e2.min()), max(total_e1.max(), total_e2.max())]
axes2[1].plot(lims, lims, color=COLOR2, linewidth=1.2, linestyle="--", label="y = x")
axes2[1].set_xlabel("Original total E [GeV]")
axes2[1].set_ylabel("step2point total E [GeV]")
axes2[1].set_title("Correlation")
axes2[1].legend()

out2 = "martina_test/total_energy_comparison.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")

plt.show()
