import subprocess
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "tiny_showers.h5"


def test_generate_validation_plots_compare_mode(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    pipeline_out = tmp_path / "pipeline_out"
    compare_out = tmp_path / "compare_plots"

    subprocess.run(
        [
            sys.executable,
            "examples/run_step2point_pipeline.py",
            "--input",
            str(DATA),
            "--algorithm",
            "identity",
            "--output",
            str(pipeline_out),
        ],
        check=True,
        cwd=repo_root,
    )

    subprocess.run(
        [
            sys.executable,
            "examples/generate_validation_plots.py",
            "--input",
            str(DATA),
            str(pipeline_out / "compressed_identity.h5"),
            "--outdir",
            str(compare_out),
        ],
        check=True,
        cwd=repo_root,
    )

    assert (compare_out / "energy_ratio.png").exists()
    assert (compare_out / "longitudinal_profile_overlay.png").exists()
