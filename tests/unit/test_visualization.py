from pathlib import Path

import numpy as np

from step2point.core.shower import Shower
from step2point.validation.benchmark_plots import generate_observables_matrix
from step2point.vis import plot_shower_distributions, plot_shower_overview, plot_shower_projections


def _toy_shower() -> Shower:
    return Shower(
        shower_id=7,
        x=np.array([0.0, 1.0, 2.0, 3.0]),
        y=np.array([0.0, 0.5, 1.0, 1.5]),
        z=np.array([0.0, 1.0, 2.0, 3.0]),
        E=np.array([1.0, 2.0, 1.5, 0.5]),
        t=np.array([0.0, 0.1, 0.2, 0.3]),
        pdg=np.array([11, 11, 22, 22]),
        metadata={"subdetector": np.array([0, 0, 1, 1])},
    )


def test_visualization_helpers_write_png_files(tmp_path: Path):
    shower = _toy_shower()
    plot_shower_projections(shower, tmp_path / "projections.png")
    plot_shower_distributions(shower, tmp_path / "distributions.png")
    plot_shower_overview(shower, tmp_path / "overview.png", axis_override=[0.0, 0.0, 1.0])
    generate_observables_matrix([shower, shower.copy()], tmp_path / "matrix.png", selected_index=0)
    assert (tmp_path / "projections.png").exists()
    assert (tmp_path / "distributions.png").exists()
    assert (tmp_path / "overview.png").exists()
    assert (tmp_path / "matrix.png").exists()
