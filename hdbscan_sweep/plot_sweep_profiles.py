"""Longitudinal profile and point-energy distribution of the HDBSCAN sweep,
side by side in one figure. Both panels read the compressed step files
(compressed_*.h5), the same files the compression ratios are computed from.
The June input_cc3.h5 conversions of the sweep are NOT used: they predate the
16 Jun converter fix and their fourth column is a coordinate, not the energy.

Layer is assigned from the global depth y against
Metadata().layer_bottom_pos_global, as in the CaloClouds-3 projection plots;
points outside the top barrel segment (y below the first layer) are dropped
from the profile.

Usage: python plot_sweep_profiles.py <out_dir>"""

import os
import re
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Metadata (ILD layer positions) comes from CaloClouds-3 preprocessing/, found via $CC3_DIR
sys.path.insert(0, os.environ.get("CC3_DIR", "/eos/user/m/mamozzan/CaloClouds-3"))
from preprocessing.metadata import Metadata  # noqa: E402
from sweep_style import EXCLUDE_MS, MS_COLORS, REF_COLORS, apply_style  # noqa: E402

SWEEP = Path(__file__).resolve().parents[1] / "outputs/hdbscan_sweep"  # step2point/outputs
S2P = "/eos/project/f/fast/step2point_files/pipeline2_{0}/test_small/compressed_{0}.h5"
MCS = 20
LAYER_Y = np.asarray(Metadata().layer_bottom_pos_global)
N_LAYERS = len(LAYER_Y)
out_dir = Path(sys.argv[1])


def load(path):
    with h5py.File(path, "r") as f:
        st = f["steps"]
        e = st["energy"][:]
        y = st["position"][:, 1]
        n_showers = len(np.unique(st["event_id"][:]))
    layer = np.searchsorted(LAYER_Y, y, side="right") - 1
    in_seg = layer >= 0
    per_layer = np.bincount(layer[in_seg], minlength=N_LAYERS)[:N_LAYERS] / n_showers
    print(f"  {Path(path).parent.name:28s} {len(e) / n_showers:7.0f} pts/shower, {1 - in_seg.mean():.2%} outside the segment")
    return per_layer, e[e > 0]


runs = []
for folder in SWEEP.glob(f"hdbscan_mcs{MCS}_ms*"):
    m = re.fullmatch(rf"hdbscan_mcs{MCS}_ms(\d+)", folder.name)  # epsilon = 0 only
    if m and int(m.group(1)) not in {4} | EXCLUDE_MS:
        runs.append((int(m.group(1)), folder / "compressed_hdbscan.h5"))
runs.sort()

apply_style()
ident = load(S2P.format("identity"))
refs = [
    (load(S2P.format(k)), label)
    for k, label in [("merge_within_cell", "merge within cell"), ("merge_within_regular_subcell", "merge within regular subcell")]
]
sweep = [(ms, load(p)) for ms, p in runs]

fig, (ax_l, ax_e) = plt.subplots(1, 2, figsize=(9.5, 3.5))
layers = np.arange(N_LAYERS)

# ── longitudinal profile ─────────────────────────────────────────────────
for (prof, _), label in refs:
    ax_l.plot(layers, prof, color=REF_COLORS[label], ls="--", lw=1.6)
for ms, (prof, _) in sweep:
    ax_l.plot(layers, prof, color=MS_COLORS[ms], lw=1.6)
ax_l.set_xlabel("layer")
ax_l.set_ylabel("mean points per layer")
ax_l.set_xlim(0, N_LAYERS - 1)
ax_l.set_ylim(bottom=0)
ax_l.grid(True, alpha=0.3)

# ── point energy ─────────────────────────────────────────────────────────
e_max = max(v.max() for v in [ident[1]] + [r[0][1] for r in refs] + [s[1][1] for s in sweep])
bins = np.logspace(-6, np.log10(e_max), 80)
# identity (unmerged steps) is the reference, drawn as the grey band like
# Geant4 in the projection plots
ax_e.hist(ident[1], bins=bins, histtype="stepfilled", color="#eeede9", edgecolor="#c9c8c2", label="identity (no merging)")
for (_, e), label in refs:
    if label == "merge within cell":  # filled, light blue
        ax_e.hist(
            e, bins=bins, histtype="stepfilled", color="#56b4e9", alpha=0.45, edgecolor=REF_COLORS[label], lw=1.0, label=label
        )
    else:
        ax_e.hist(e, bins=bins, histtype="step", ls="--", lw=1.6, color=REF_COLORS[label], label=label)
for ms, (_, e) in sweep:
    ax_e.hist(e, bins=bins, histtype="step", lw=1.6, color=MS_COLORS[ms], label=f"$m_s = {ms}$")
ax_e.set_xscale("log")
ax_e.set_yscale("log")
ax_e.set_xlabel("point energy [GeV]")
ax_e.set_ylabel("points / bin")

ax_e.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout()
fig.savefig(out_dir / "sweep_profiles.pdf", bbox_inches="tight")
fig.savefig(out_dir / "sweep_profiles.png", dpi=110, bbox_inches="tight")
print("saved", out_dir)
