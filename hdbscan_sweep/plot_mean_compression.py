"""Mean compression ratio of the HDBSCAN sweep, on the full 0 (fully merged) to
1 (uncompressed) scale. Reads the same per-run summary files, with the same
filters, as plot_compression_ratios() in
hdbscan_sweep/sweep_hdbscan.py."""

import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sweep_style import EXCLUDE_MS, MS_COLORS, REF_COLORS, apply_style

CC3_DIR = os.environ.get("CC3_DIR", "/eos/user/m/mamozzan/CaloClouds-3")
# sweep outputs written by sweep_hdbscan.py
SWEEP = Path(os.environ.get("SWEEP_DIR", f"{CC3_DIR}/outputs/hdbscan_sweep"))
REFS = [
    (
        "/eos/project/f/fast/step2point_files/pipeline2_merge_within_cell/test_small/compression_summary_merge_within_cell.txt",
        "merge within cell",
    ),
    (
        "/eos/project/f/fast/step2point_files/pipeline2_merge_within_regular_subcell/test_small/"
        "compression_summary_merge_within_regular_subcell.txt",
        "merge within regular subcell",
    ),
]


def read_summary(path):
    row = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            try:
                row[k] = float(v)
            except ValueError:
                pass
    return row


rows = []
for folder in sorted(SWEEP.glob("hdbscan_mcs*_ms*")):
    m = re.search(r"mcs(\d+)_ms(\d+)(?:_eps([\d.]+))?", folder.name)
    if m is None or (m.group(3) is not None and float(m.group(3)) != 0) or int(m.group(2)) in {4} | EXCLUDE_MS:
        continue
    txt = folder / "compression_summary_hdbscan.txt"
    if txt.exists():
        rows.append({"mcs": int(m.group(1)), "ms": int(m.group(2)), **read_summary(txt)})
df = pd.DataFrame(rows)
df = df[df.mcs <= 40]  # x axis stops at m_cs = 40
print(f"{len(df)} settings, ms = {sorted(df.ms.unique())}")

apply_style()
fig, ax = plt.subplots(figsize=(5.0, 3.5))

ms_vals = sorted(df.ms.unique())
for ms in ms_vals:
    g = df[df.ms == ms].sort_values("mcs")
    ax.plot(g.mcs, g.mean_compression_ratio, marker="o", ms=4, lw=1.6, color=MS_COLORS[ms], label=f"$m_s = {ms}$")
for path, label in REFS:
    r = read_summary(path)
    ax.axhline(r["mean_compression_ratio"], color=REF_COLORS[label], ls="--", lw=1.2, label=label)

ax.set_ylim(0, 1)
ax.set_xlim(0, df.mcs.max() + 3)
ax.set_xlabel(r"minimum cluster size $m_{cs}$")
ax.set_ylabel(r"mean $N_\mathrm{after}/N_\mathrm{before}$")
ax.grid(True, alpha=0.3)
ax.legend(ncol=2, frameon=False, loc="upper right")
fig.tight_layout()

for out in sys.argv[1:]:
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)
