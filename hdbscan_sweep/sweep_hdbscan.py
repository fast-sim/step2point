"""
HDBSCAN (min_cluster_size, min_samples) sweep with step2point. Only tested with --preset ild.

    python hdbscan_sweep/sweep_hdbscan.py --preset ild

Writes one folder per run, summary.csv and plots/ to --output-dir. See the README.
"""

import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from plot_sweep_hdbscan import _read_summary, make_all_plots

REPO_ROOT = Path(__file__).resolve().parents[1]  # step2point repo root

# ─────────────────────────────────────────────
# DETECTOR PRESETS
# ─────────────────────────────────────────────
# Only "ild" exists and has been tested. Command-line options override preset
# values. To add a detector, add an entry with the same keys:
#   references: {algorithm: extra pipeline arguments} run once on the same input
#       into <output dir>/references/<algorithm>/ and drawn as reference curves;
#       one whose --compact-xml is not reachable is skipped
PRESETS = {
    "ild": {
        "input": str(REPO_ROOT / "tests/data/ILD/test_sweep_hdbscan_ild.edm4hep.root"),
        "collections": ["EcalBarrelCollection"],
        "cell_id_encoding": "system:5,module:3,stave:4,tower:4,layer:6,wafer:6,slice:4,cellX:32:-16,cellY:-16",
        "merge_scope": "cell_id",
        "use_time": True,
        # bottom of the 30 sensitive layers of the top barrel segment, global y [mm]
        # (Metadata().layer_bottom_pos_global of the ILD preprocessing), for the longitudinal profile
        "layer_positions": [
            1811.3402,
            1814.4651,
            1823.8101,
            1826.9351,
            1836.28,
            1839.405,
            1848.75,
            1851.875,
            1861.22,
            1864.345,
            1873.6899,
            1876.8149,
            1886.16,
            1889.285,
            1898.63,
            1901.755,
            1911.1,
            1914.225,
            1923.5699,
            1926.6949,
            1938.14,
            1943.365,
            1954.8101,
            1960.035,
            1971.48,
            1976.705,
            1988.15,
            1993.375,
            2004.8199,
            2010.045,
        ],
        "references": {
            "identity": [],
            "merge_within_cell": [],
            # needs the regular-ECal ILD geometry, only on EOS for now
            "merge_within_regular_subcell": [
                "--compact-xml",
                "/eos/user/p/pmckeown/Step2Point_Clustering/ILD_Regular_Detector/ILD/compact/ILD_l5_o1_v02/ILD_l5_o1_v02.xml",
                "--collection-name",
                "EcalBarrelCollection",
                "--grid-x",
                "5",
                "--grid-y",
                "5",
                "--position-mode",
                "weighted",
            ],
        },
    },
}

DEFAULT_MIN_CLUSTER_SIZES = [5, 10, 15, 20, 25, 40, 60, 80]
DEFAULT_MIN_SAMPLES = [3, 5, 8, 12, 20, 40, 60]
DEFAULT_EPSILON = [0.0]

# Set by configure() from the command line and the preset
INPUT = None
COLLECTIONS = []
CELL_ID_ENCODING = None
COMPACT_XML = None
COLLECTION_NAMES = []
MERGE_SCOPE = None
USE_TIME = False
REFERENCES = {}
REFERENCE_DIR = None
LAYER_POSITIONS = None
BASE_OUTPUT = REPO_ROOT / "outputs/hdbscan_sweep"


def run_pipeline(mcs: int, ms: int, epsilon: float) -> Path:
    out_dir = BASE_OUTPUT / f"hdbscan_mcs{mcs}_ms{ms}"
    if epsilon != 0:
        out_dir = Path(str(out_dir) + f"_eps{epsilon}")

    algo_args = ["--min-cluster-size", str(mcs), "--min-samples", str(ms), "--epsilon", str(epsilon)]
    if CELL_ID_ENCODING:
        algo_args += ["--cell-id-encoding", CELL_ID_ENCODING]
    if COMPACT_XML:
        algo_args += ["--compact-xml", COMPACT_XML, "--collection-name", *COLLECTION_NAMES]
    if MERGE_SCOPE:
        algo_args += ["--merge-scope", MERGE_SCOPE]
    if USE_TIME:
        algo_args.append("--use-time")

    print(f"  → mcs={mcs:2d}, ms={ms} epsilon={epsilon} ...", end=" ", flush=True)
    return run_step2point(out_dir, "hdbscan", algo_args)


def make_references():
    """Run the reference algorithms on INPUT into REFERENCE_DIR/<algorithm>/ (existing outputs are kept)."""
    for algorithm, algo_args in REFERENCES.items():
        if "--compact-xml" in algo_args:
            xml = Path(algo_args[algo_args.index("--compact-xml") + 1])
            if not xml.exists():
                print(f"  [skip] reference {algorithm}: geometry {xml} not reachable")
                continue
        print(f"  → reference {algorithm} ...", end=" ", flush=True)
        run_step2point(REFERENCE_DIR / algorithm, algorithm, algo_args)


def run_step2point(out_dir: Path, algorithm: str, algo_args: list) -> Path:
    """Run run_step2point_pipeline.py with one algorithm into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "examples/run_step2point_pipeline.py"),
        "--input",
        INPUT,
        "--algorithm",
        algorithm,
        "--output",
        str(out_dir),
        *algo_args,
    ]
    if COLLECTIONS:
        cmd += ["--collections", *COLLECTIONS]

    log_path = out_dir / "run.log"
    file_dir = out_dir / f"compressed_{algorithm}.h5"
    if file_dir.exists():
        print(f"  [skip] output exists at {out_dir}")
    else:
        with open(log_path, "w") as log:
            print(f"Running command:\n{' '.join(cmd)}\nLogging to {log_path}")
            result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
            status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
            print(status)
    return out_dir


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="HDBSCAN (min_cluster_size, min_samples) sweep with step2point. Only tested with --preset ild."
    )
    p.add_argument("--preset", choices=sorted(PRESETS), help="detector preset (only 'ild', the tested one)")
    p.add_argument("--input", help="input file for run_step2point_pipeline.py")
    p.add_argument("--collections", nargs="+", help="readout collection(s) to read (EDM4hep input)")
    p.add_argument("--cell-id-encoding", help="DD4hep cell-ID encoding string, passed to the pipeline as --cell-id-encoding")
    p.add_argument("--compact-xml", help="DD4hep compact XML to read the encoding from (with --collection-name)")
    p.add_argument("--collection-name", nargs="+", help="readout name(s) in --compact-xml")
    p.add_argument("--merge-scope", help="HDBSCAN merge scope (pipeline default if not given)")
    p.add_argument("--use-time", action="store_true", help="include time as a clustering feature")
    p.add_argument("--min-cluster-sizes", type=int, nargs="+", default=DEFAULT_MIN_CLUSTER_SIZES)
    p.add_argument("--min-samples", type=int, nargs="+", default=DEFAULT_MIN_SAMPLES)
    p.add_argument("--epsilon", type=float, nargs="+", default=DEFAULT_EPSILON)
    p.add_argument(
        "--layer-positions",
        type=float,
        nargs="+",
        help="bottom of each layer along global y [mm], for the longitudinal profile plot",
    )
    p.add_argument("--profile-mcs", type=int, default=20, help="m_cs of the point-energy and profile plots (default 20)")
    p.add_argument(
        "--output-dir",
        type=Path,
        help="default: $SWEEP_DIR, else <repo>/outputs/hdbscan_sweep",
    )
    p.add_argument(
        "--reference-dir",
        type=Path,
        help="folder for the reference runs, <dir>/<algorithm>/ (default: <output dir>/references with a preset)",
    )
    return p.parse_args()


def configure(args):
    """Fill the module-level settings from the preset, overridden by the command line."""
    global INPUT, COLLECTIONS, CELL_ID_ENCODING, COMPACT_XML, COLLECTION_NAMES, MERGE_SCOPE
    global USE_TIME, REFERENCES, REFERENCE_DIR, LAYER_POSITIONS, BASE_OUTPUT
    cfg = dict(PRESETS[args.preset]) if args.preset else {}
    for key in ("input", "collections", "cell_id_encoding", "merge_scope"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if not cfg.get("input"):
        sys.exit("Pass --input (and --collections for EDM4hep input), or use --preset ild.")
    if not cfg.get("cell_id_encoding") and not (args.compact_xml and args.collection_name):
        sys.exit("HDBSCAN needs --cell-id-encoding, or --compact-xml together with --collection-name.")

    INPUT = cfg["input"]
    COLLECTIONS = cfg.get("collections") or []
    CELL_ID_ENCODING = cfg.get("cell_id_encoding")
    COMPACT_XML = args.compact_xml
    COLLECTION_NAMES = args.collection_name or []
    MERGE_SCOPE = cfg.get("merge_scope")
    USE_TIME = args.use_time or cfg.get("use_time", False)
    REFERENCES = cfg.get("references", {})
    LAYER_POSITIONS = args.layer_positions or cfg.get("layer_positions")
    if args.output_dir:
        BASE_OUTPUT = args.output_dir
    elif os.environ.get("SWEEP_DIR"):
        BASE_OUTPUT = Path(os.environ["SWEEP_DIR"])
    else:
        BASE_OUTPUT = REPO_ROOT / "outputs/hdbscan_sweep"
    REFERENCE_DIR = args.reference_dir or (BASE_OUTPUT / "references" if REFERENCES else None)


def main():
    args = parse_args()
    configure(args)
    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)
    if REFERENCES:
        print("── Running the reference algorithms ──────────────")
        make_references()

    combos = [
        (mcs, ms, epsilon)
        for mcs, ms, epsilon in itertools.product(args.min_cluster_sizes, args.min_samples, args.epsilon)
        if ms <= mcs  # skip invalid HDBSCAN combos
    ]

    print(f"── Running {len(combos)} configurations ──────────────────────")
    rows = []
    for mcs, ms, epsilon in combos:
        out_dir = run_pipeline(mcs, ms, epsilon)
        # the pipeline's own key=value summary (compression numbers)
        summary = out_dir / "compression_summary_hdbscan.txt"
        rows.append({"mcs": mcs, "ms": ms, "epsilon": epsilon, **(_read_summary(summary) if summary.exists() else {})})

    df = pd.DataFrame(rows)
    csv_path = BASE_OUTPUT / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n── Summary saved to {csv_path}")

    make_all_plots(BASE_OUTPUT, REFERENCE_DIR, args.profile_mcs, LAYER_POSITIONS)


if __name__ == "__main__":
    main()
