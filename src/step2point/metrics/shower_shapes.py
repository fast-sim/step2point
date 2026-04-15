from __future__ import annotations

import numpy as np

from step2point.metrics.spatial import longitudinal_radial_phi


def weighted_moment(values, weights, order: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    norm = np.sum(weights)
    if norm <= 0:
        return float("nan")
    return float(np.sum((values**order) * weights) / norm)


def shower_moments(shower):
    long, radial, _ = longitudinal_radial_phi(shower)
    w = shower.E
    return {
        "longitudinal_m1": weighted_moment(long, w, 1),
        "longitudinal_m2": weighted_moment(long, w, 2),
        "radial_m1": weighted_moment(radial, w, 1),
        "radial_m2": weighted_moment(radial, w, 2),
    }
