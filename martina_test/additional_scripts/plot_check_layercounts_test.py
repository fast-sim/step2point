"""
Sanity-check + diagnostic plots for the output of
convert_to_cc3_format_layercounts_test.py.

Checks:
  1. events[:, :n_layers, 0] (the embedded counts) match the standalone
     'layer_counts' dataset exactly.
  2. Both match an independent recount done from the actual hits in
     events[:, n_layers:, :], using the layer id stored in column 2.
  3. sum(layer_counts, axis=1) == n_points for every shower.

Usage:
    python martina_test/plot_check_layercounts_test.py <file.input_cc3_layercounts_test.h5>
"""

import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

assert len(sys.argv) == 2, "Usage: python plot_check_layercounts_test.py <file.h5>"
path = sys.argv[1]
assert os.path.exists(path), f"File not found: {path}"

print(f"Loading {path} ...")
with h5py.File(path, "r") as f:
    energy = f["energy"][:].squeeze()
    events = f["events"][:]  # (N, n_layers + max_hits, 4)
    n_points = f["n_points"][:]
    layer_counts = f["layer_counts"][:]  # (N, n_layers)

N, total_len, _ = events.shape
n_layers = layer_counts.shape[1]
max_hits = total_len - n_layers

embedded_counts = events[:, :n_layers, 0]
hits = events[:, n_layers:, :]

print(f"  N events  : {N}")
print(f"  n_layers  : {n_layers}")
print(f"  max_hits  : {max_hits}")

# --- check 1: embedded counts vs standalone dataset ---
match_embedded = np.allclose(embedded_counts, layer_counts)
print(f"\n[check] events[:, :n_layers, 0] == layer_counts : {match_embedded}")

# --- check 2: independent recount from the actual hits ---
layer_ids = np.floor(hits[..., 2]).astype(int)
valid = hits[..., 3] > 0
recount = np.zeros_like(layer_counts)
for i in range(N):
    ids = layer_ids[i][valid[i]]
    ids = ids[(ids >= 0) & (ids < n_layers)]
    recount[i] = np.bincount(ids, minlength=n_layers)[:n_layers]

match_recount = np.allclose(recount, layer_counts)
print(f"[check] recount from hits == layer_counts        : {match_recount}")
if not match_recount:
    diff = np.abs(recount - layer_counts)
    bad = np.argwhere(diff > 0)
    print(f"  mismatches: {len(bad)} (showing up to 5)")
    print(bad[:5])

# --- check 3: sums match n_points ---
sums_match = np.array_equal(layer_counts.sum(axis=1).astype(np.int64), n_points)
print(f"[check] sum(layer_counts, axis=1) == n_points     : {sums_match}")
print(f"  layer_counts sums: {layer_counts.sum(axis=1)}")
print(f"  n_points:          {n_points}")
print(f"layer in first event: {events[1, :100, 2]}")

all_ok = match_embedded and match_recount and sums_match
print(f"\nAll checks passed: {all_ok}")

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 9))
fig.suptitle(f"Layer-count sanity check — {os.path.basename(path)}\n{N} events, all_ok={all_ok}")

# 1. per-layer counts for every shower, embedded vs recount overlaid
ax = axes[0]
layer_idx = np.arange(n_layers)
for i in range(N):
    ax.plot(layer_idx, embedded_counts[i], color="steelblue", alpha=0.6, lw=1)
    ax.plot(layer_idx, recount[i], color="darkorange", alpha=0.6, lw=1, ls="--")
ax.plot([], [], color="steelblue", label="embedded (events[:, :n_layers, 0])")
ax.plot([], [], color="darkorange", ls="--", label="recount (from hits)")
ax.set_xlabel("Layer index")
ax.set_ylabel("N points in layer")
ax.set_title("Per-layer hit counts: embedded vs. recounted")
ax.legend()

# 2. total hits per shower: sum(layer_counts) vs n_points
ax = axes[1]
x = np.arange(N)
width = 0.35
ax.bar(x - width / 2, layer_counts.sum(axis=1), width, label="sum(layer_counts)", color="steelblue")
ax.bar(x + width / 2, n_points, width, label="n_points", color="darkorange")
ax.set_xlabel("Shower index")
ax.set_ylabel("Total hits")
ax.set_title("sum(layer_counts) vs n_points")
ax.legend()

plt.tight_layout()
out_path = path.replace(".h5", ".layercounts_check.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {out_path}")
plt.show()
