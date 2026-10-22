import json
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
            "--ratio-ylim",
            "0.9",
            "1.1",
            "--outdir",
            str(compare_out),
        ],
        check=True,
        cwd=repo_root,
    )

    assert (compare_out / "energy_ratio.png").exists()
    assert (compare_out / "longitudinal_profile_overlay.png").exists()
    summary = json.loads((compare_out / "validation_summary.json").read_text())
    assert summary["reference"]["label"] == "pre"
    assert summary["comparisons"][0]["label"] == "post"
    assert "n_points_post" in summary["comparisons"][0]["distributions"]


def test_generate_validation_plots_summary_only(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    pipeline_out = tmp_path / "pipeline_out"
    compare_out = tmp_path / "compare_summary_only"

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
            "--summary-only",
            "--outdir",
            str(compare_out),
        ],
        check=True,
        cwd=repo_root,
    )

    assert (compare_out / "validation_summary.json").exists()
    assert not (compare_out / "energy_ratio.png").exists()


def test_generate_validation_plots_multi_compare_mode(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    identity_out = tmp_path / "identity_out"
    merge_out = tmp_path / "merge_out"
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
            str(identity_out),
        ],
        check=True,
        cwd=repo_root,
    )
    subprocess.run(
        [
            sys.executable,
            "examples/run_step2point_pipeline.py",
            "--input",
            str(DATA),
            "--algorithm",
            "merge_within_cell",
            "--output",
            str(merge_out),
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
            str(identity_out / "compressed_identity.h5"),
            str(merge_out / "compressed_merge_within_cell.h5"),
            "--label",
            "pre",
            "identity",
            "merge_within_cell",
            "--outdir",
            str(compare_out),
        ],
        check=True,
        cwd=repo_root,
    )

    assert (compare_out / "energy_ratio.png").exists()
    assert (compare_out / "longitudinal_profile_overlay.png").exists()
    assert not any(path.is_dir() for path in compare_out.iterdir())
    summary = json.loads((compare_out / "validation_summary.json").read_text())
    assert [entry["label"] for entry in summary["comparisons"]] == ["identity", "merge_within_cell"]


def test_generate_validation_plots_compare_mode_no_ratio(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    identity_out = tmp_path / "identity_out"
    merge_out = tmp_path / "merge_out"
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
            str(identity_out),
        ],
        check=True,
        cwd=repo_root,
    )
    subprocess.run(
        [
            sys.executable,
            "examples/run_step2point_pipeline.py",
            "--input",
            str(DATA),
            "--algorithm",
            "merge_within_cell",
            "--output",
            str(merge_out),
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
            str(identity_out / "compressed_identity.h5"),
            str(merge_out / "compressed_merge_within_cell.h5"),
            "--label",
            "pre",
            "identity",
            "merge_within_cell",
            "--no-ratio",
            "--outdir",
            str(compare_out),
        ],
        check=True,
        cwd=repo_root,
    )

    assert (compare_out / "n_points.png").exists()
    assert (compare_out / "longitudinal_profile_overlay.png").exists()


def test_plot_validation_summary_trends(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    pipeline_out = tmp_path / "pipeline_out"
    compare_out = tmp_path / "compare_plots"
    trend_out = tmp_path / "trend_plots"

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

    subprocess.run(
        [
            sys.executable,
            "examples/plot_validation_summary_trends.py",
            "--summary",
            str(compare_out / "validation_summary.json"),
            "--metric",
            "n_points_post",
            "point_count_ratio",
            "--outdir",
            str(trend_out),
        ],
        check=True,
        cwd=repo_root,
    )

    assert (trend_out / "n_points_post_vs_energy.png").exists()


def test_plot_validation_summary_trends_reference_only(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    pipeline_out = tmp_path / "pipeline_out"
    compare_out = tmp_path / "compare_plots"
    trend_out = tmp_path / "trend_plots"

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

    subprocess.run(
        [
            sys.executable,
            "examples/plot_validation_summary_trends.py",
            "--summary",
            str(compare_out / "validation_summary.json"),
            "--metric",
            "n_points_post",
            "--reference-only",
            "--outdir",
            str(trend_out),
        ],
        check=True,
        cwd=repo_root,
    )

    assert (trend_out / "n_points_post_vs_energy.png").exists()
