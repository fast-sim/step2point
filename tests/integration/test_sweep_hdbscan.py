import os
import subprocess
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")  # the sweep's optional dependency (.[sweep])

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data"
GENERIC_PLOTS = {"compression_ratios.png", "mean_compression_ratio.png", "points_per_event_overlay.png", "point_energy.png"}


def _run_sweep(out, *args):
    # src/ on PYTHONPATH like pytest's own pythonpath setting: sys.executable may be a Python
    # without step2point installed (e.g. Key4hep's when `pytest` is not the venv's)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")]))}
    subprocess.run(
        [sys.executable, "hdbscan_sweep/sweep_hdbscan.py", "--output-dir", str(out), *args], check=True, cwd=ROOT, env=env
    )


def _check_run(out, mcs, ms):
    assert (out / f"hdbscan_mcs{mcs}_ms{ms}" / "compressed_hdbscan.h5").exists()
    summary = pd.read_csv(out / "summary.csv")
    assert len(summary) == 1
    assert 0 < summary["mean_compression_ratio"].iloc[0] <= 1
    return {p.name for p in (out / "plots").iterdir()}


def test_sweep_generic_odd(tmp_path):
    _run_sweep(
        tmp_path,
        "--input",
        str(DATA / "ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV.h5"),
        "--cell-id-encoding",
        "system:8,barrel:3,module:4,stave:1,layer:6,slice:5,x:32:-16,y:-16",
        "--use-time",
        "--min-cluster-sizes",
        "5",
        "--min-samples",
        "3",
        "--profile-mcs",
        "5",
    )
    plots = _check_run(tmp_path, 5, 3)
    assert GENERIC_PLOTS <= plots
    assert "longitudinal_profile.png" not in plots  # no layer positions without a preset


@pytest.mark.root
def test_sweep_ild_preset(tmp_path):
    pytest.importorskip("podio.root_io")
    _run_sweep(tmp_path, "--preset", "ild", "--min-cluster-sizes", "20", "--min-samples", "5")
    plots = _check_run(tmp_path, 20, 5)
    assert GENERIC_PLOTS | {"longitudinal_profile.png"} <= plots
    for algo in ("identity", "merge_within_cell"):  # merge_within_regular_subcell is skipped without --regular-compact-xml
        assert (tmp_path / "references" / algo / f"compressed_{algo}.h5").exists()
