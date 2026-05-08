"""
Plot x, y, z, energy distributions comparing:
  - File 1 (ROOT):       raw EDM4hep .root file read with uproot
  - File 2 (step2point): flat HDF5 format with event_id grouping

Usage:
    python plot_xyze.py
"""

import awkward as ak
import h5py
import matplotlib.pyplot as plt
import numpy as np
import uproot

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT_FILE = "../photons_root/p22_th45-135_ph79-109_en5-130_seed0_ip.edm4hep.root"
HDF5_FILE = "outputs/pipeline_identity_0/compressed_identity.h5"

N_EVENTS = None  # set to an int (e.g. 100) to speed things up

# ── collections to read from ROOT ─────────────────────────────────────────────
# Each entry: (contribution_collection, section_label)
# Only collections with non-zero hits in this file are included.
COLLECTIONS = [
    ("ECalBarrelSiHitsEven", "ECalBarrel"),
    ("ECalBarrelSiHitsOdd", "ECalBarrel"),
    ("ECalEndcapSiHitsEven", "ECalEndcap"),
    ("ECalEndcapSiHitsOdd", "ECalEndcap"),
    ("HcalBarrelRegCollection", "HCalBarrel"),
    ("HcalEndcapRingCollection", "HCalEndcap"),
]

# ── load ROOT file ─────────────────────────────────────────────────────────────
print("Loading ROOT file...")
root_file = uproot.open(ROOT_FILE)
event_tree = root_file["events"]

all_x, all_y, all_z, all_e = [], [], [], []

for col, section in COLLECTIONS:
    contrib = f"{col}Contributions"
    base = f"{contrib}/{contrib}."
    try:
        x = event_tree[base + "stepPosition.x"].array(entry_stop=N_EVENTS)
        y = event_tree[base + "stepPosition.y"].array(entry_stop=N_EVENTS)
        z = event_tree[base + "stepPosition.z"].array(entry_stop=N_EVENTS)
        e = event_tree[base + "energy"].array(entry_stop=N_EVENTS)
        n = ak.sum(ak.num(e))
        if n == 0:
            print(f"  SKIP (empty): {contrib}")
            continue
        print(f"  {contrib}: {n:,} hits")
        all_x.append(ak.flatten(x))
        all_y.append(ak.flatten(y))
        all_z.append(ak.flatten(z))
        all_e.append(ak.flatten(e))
    except Exception as err:
        print(f"  WARN: could not read {contrib}: {err}")

x1 = ak.to_numpy(ak.concatenate(all_x))
y1 = ak.to_numpy(ak.concatenate(all_y))
z1 = ak.to_numpy(ak.concatenate(all_z))
e1 = ak.to_numpy(ak.concatenate(all_e))
print(f"\nROOT total hits : {len(x1):,}")

# per-event total energy from ROOT
print("Computing per-event energy from ROOT...")
n_events_root = event_tree.num_entries if N_EVENTS is None else N_EVENTS
total_e1 = np.zeros(n_events_root)
for col, _ in COLLECTIONS:
    contrib = f"{col}Contributions"
    base = f"{contrib}/{contrib}."
    try:
        e = event_tree[base + "energy"].array(entry_stop=N_EVENTS)
        if ak.sum(ak.num(e)) == 0:
            continue
        total_e1 += ak.to_numpy(ak.sum(e, axis=1))
    except Exception:
        pass

# ── load HDF5 (step2point) ────────────────────────────────────────────────────
print("\nLoading step2point HDF5...")
with h5py.File(HDF5_FILE, "r") as f:
    ev_ids = f["steps/event_id"][:]
    pos2 = f["steps/position"][:]
    e2 = f["steps/energy"][:]

if N_EVENTS is not None:
    sel = ev_ids < N_EVENTS
    pos2 = pos2[sel]
    e2 = e2[sel]
    ev_ids = ev_ids[sel]

x2 = pos2[:, 0]
y2 = pos2[:, 1]
z2 = pos2[:, 2]
print(f"step2point hits : {len(x2):,}")

n_ev = n_events_root
total_e2 = np.array([e2[ev_ids == i].sum() for i in range(n_ev)])

# ── style ─────────────────────────────────────────────────────────────────────
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
COLOR1 = "#2196F3"  # blue  — ROOT (filled)
COLOR2 = "#E53935"  # red   — step2point (line)

# ── Figure 1: x, y, z, energy distributions ───────────────────────────────────
variables = [
    ("x  [mm]", x1, x2),
    ("y  [mm]", y1, y2),
    ("z  [mm]", z1, z2),
    ("Energy  [GeV]", e1, e2),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
fig.suptitle("Calorimeter Hit Distributions — ROOT vs step2point", fontsize=13, fontweight="bold")

for ax, (lbl, d1, d2) in zip(axes.flat, variables):
    lo = d1.min()  # min(d1.min(), d2.min())
    hi = d2.max()  # max(d1.max(), d2.max())
    edges = np.linspace(lo, hi, BINS + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # ROOT: filled + outline
    ax.hist(d1, bins=edges, density=True, color=COLOR1, alpha=0.4, label=f"ROOT  (n={len(d1):,})")
    ax.hist(d1, bins=edges, density=True, color=COLOR1, histtype="step", linewidth=1.2)

    # step2point: line only
    c2, _ = np.histogram(d2, bins=edges, density=True)
    ax.step(centres, c2, where="mid", color=COLOR2, linewidth=1.8, label=f"step2point  (n={len(d2):,})")

    ax.set_xlabel(lbl)
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

fig.savefig("martina_test/hit_distributions_xyze.png", dpi=150, bbox_inches="tight")
print("\nSaved: hit_distributions_xyze.png")

# ── Figure 2: per-event total energy ──────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
fig2.suptitle("Per-Event Total Deposited Energy — ROOT vs step2point", fontsize=13, fontweight="bold")

edges_e = np.linspace(min(total_e1.min(), total_e2.min()), max(total_e1.max(), total_e2.max()), 80)
centres_e = 0.5 * (edges_e[:-1] + edges_e[1:])

# histogram
axes2[0].hist(total_e1, bins=edges_e, density=True, color=COLOR1, alpha=0.4, label="ROOT")
axes2[0].hist(total_e1, bins=edges_e, density=True, color=COLOR1, histtype="step", linewidth=1.2)
c_e2, _ = np.histogram(total_e2, bins=edges_e, density=True)
axes2[0].step(centres_e, c_e2, where="mid", color=COLOR2, linewidth=1.8, label="step2point")
axes2[0].set_xlabel("Total Event Energy [GeV]")
axes2[0].set_ylabel("Density")
axes2[0].set_title("Distribution")
axes2[0].legend()

# scatter
axes2[1].scatter(total_e1, total_e2, s=8, alpha=0.5, color="#555555", linewidths=0)
lims = [min(total_e1.min(), total_e2.min()), max(total_e1.max(), total_e2.max())]
axes2[1].plot(lims, lims, color=COLOR2, linewidth=1.2, linestyle="--", label="y = x")
axes2[1].set_xlabel("ROOT total E [GeV]")
axes2[1].set_ylabel("step2point total E [GeV]")
axes2[1].set_title("Correlation")
axes2[1].legend()

fig2.savefig("martina_test/total_energy_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: total_energy_comparison.png")

plt.show()
