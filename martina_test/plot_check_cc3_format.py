"""
Diagnostic plots for converted CC3 shower data (.input_cc3.h5).
Usage:
    python plot_cc3_check.py path/to/file.input_cc3.h5
"""

import os
import sys

import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# ── Load ──────────────────────────────────────────────────────────────────────
assert len(sys.argv) == 2, "Usage: python plot_cc3_check.py <file.input_cc3.h5>"
path = sys.argv[1]
assert os.path.exists(path), f"File not found: {path}"

print(f"Loading {path} ...")
with h5py.File(path, "r") as f:
    energy = f["energy"][:]  # (N, 1) or (N,)
    events = f["events"][:]  # (N, max_hits, 4)
    n_points = f["n_points"][:]  # (N,)
    theta_global = f["theta_global"][:]  # (N,)
    phi_global = f["phi_global"][:]  # (N,)

energy = energy.squeeze()
N, max_hits, _ = events.shape

print(f"  N events      : {N}")
print(f"  max_hits      : {max_hits}")
print(f"  energy range  : [{energy.min():.3f}, {energy.max():.3f}]")
print(f"  n_points range: [{n_points.min()}, {n_points.max()}]")
print(f"  theta range   : [{theta_global.min():.2f}, {theta_global.max():.2f}] deg")
print(f"  phi range     : [{phi_global.min():.2f}, {phi_global.max():.2f}] deg")

# ── Basic sanity checks ───────────────────────────────────────────────────────
print("\n── Sanity checks ──")
print(f"  NaN in events : {np.isnan(events).any()}")
print(f"  NaN in energy : {np.isnan(energy).any()}")
print(f"  Inf in events : {np.isinf(events).any()}")
print(f"  events with 0 hits: {(n_points == 0).sum()}")
print(f"  events with energy <= 0: {(energy <= 0).sum()}")

# padding mask: hits with all-zero entries beyond n_points
hit_energies = events[:, :, 3]  # (N, max_hits)
# check that padding region is truly zero
sample_idx = np.where(n_points < max_hits)[0]
if len(sample_idx):
    i = sample_idx[0]
    pad_e = hit_energies[i, n_points[i] :]
    print(f"  Padding check (event {i}): non-zero padding hits = {(pad_e != 0).sum()}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
fig.suptitle(f"CC3 Diagnostic — {os.path.basename(path)}\n{N} events", fontsize=13)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1. Incident energy distribution
ax = fig.add_subplot(gs[0, 0])
ax.hist(energy, bins=30, color="steelblue", edgecolor="none")
ax.set_xlabel("Incident energy")
ax.set_ylabel("Counts")
ax.set_title("Incident energy")

# 2. Number of hits per shower
ax = fig.add_subplot(gs[0, 1])
ax.hist(n_points, bins=30, color="darkorange", edgecolor="none")
ax.set_xlabel("N hits")
ax.set_title("Hits per shower")

# 3. Total deposited energy per shower
total_dep = hit_energies.sum(axis=1)
ax = fig.add_subplot(gs[0, 2])
ax.hist(total_dep, bins=30, color="seagreen", edgecolor="none")
ax.set_xlabel("Total deposited energy")
ax.set_title("Total deposited E per shower")

# 4. Deposited vs incident energy (response)
ax = fig.add_subplot(gs[1, 0])
ax.scatter(energy, total_dep, s=1, alpha=0.3, color="purple")
ax.set_xlabel("Incident energy")
ax.set_ylabel("Total deposited E")
ax.set_title("Response: dep. vs inc. energy")

# 5. theta distribution
ax = fig.add_subplot(gs[1, 1])
ax.hist(theta_global, bins=30, color="firebrick", edgecolor="none")
ax.set_xlabel("θ_global [deg]")
ax.set_title("Theta global")

# 6. phi distribution
ax = fig.add_subplot(gs[1, 2])
ax.hist(phi_global, bins=30, color="goldenrod", edgecolor="none")
ax.set_xlabel("φ_global [deg]")
ax.set_title("Phi global")

# 7. Mean hit x vs y (lateral spread, first 5k events)
cap = min(N, 5000)
x_all = events[:cap, :, 0]
y_all = events[:cap, :, 1]
# mask padding
mask = hit_energies[:cap] > 0
ax = fig.add_subplot(gs[2, 0])
ax.scatter(x_all[mask], y_all[mask], s=0.3, alpha=0.1, color="teal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"Hit x vs y (first {cap} events)")

# 8. z distribution of hits
z_all = events[:cap, :, 2]
ax = fig.add_subplot(gs[2, 1])
bins_ = int(z_all[mask].ravel().max() - z_all[mask].ravel().min() + 1)
ax.hist(z_all[mask].ravel(), bins=bins_, color="slateblue", edgecolor="none")
ax.set_xlabel("z")
ax.set_title(f"Hit z distribution (first {cap} events)")

# 9. Hit energy distribution (log scale)
e_hits = hit_energies[:cap][mask].ravel()
ax = fig.add_subplot(gs[2, 2])
ax.hist(e_hits[e_hits > 0], bins=100, color="coral", edgecolor="none", log=True)
ax.set_xlabel("Hit energy")
ax.set_ylabel("Counts (log)")
ax.set_title("Hit energy distribution")

out_path = path.replace(".h5", ".diagnostic.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {out_path}")
plt.show()
