from __future__ import annotations

import numpy as np


def energy_ratio(pre, post) -> float:
    e_pre = float(np.sum(pre.E, dtype=np.float64))
    e_post = float(np.sum(post.E, dtype=np.float64))
    return e_post / e_pre if e_pre > 0.0 else np.nan


def log_point_energy(shower, eps: float = 1e-12):
    return np.log10(np.maximum(shower.E, eps))


def aggregate_cell_energy(shower):
    if shower.cell_id is None:
        raise ValueError("cell_id is required to aggregate within cell.")
    unique_cells, inverse = np.unique(shower.cell_id, return_inverse=True)
    e = np.bincount(inverse, weights=shower.E, minlength=len(unique_cells))
    return unique_cells, e


def log_cell_energy(shower, eps: float = 1e-12):
    _, cell_energy = aggregate_cell_energy(shower)
    return np.log10(np.maximum(cell_energy, eps))
