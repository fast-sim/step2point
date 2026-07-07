from __future__ import annotations

import numpy as np

from step2point.metrics.spatial import longitudinal_radial_phi

def get_axis_override(shower):
    if "momentum" in shower.primary:
        p = np.asarray(shower.primary["momentum"], dtype=float)
        norm = np.linalg.norm(p)
        if norm > 0:
            return p / norm
    return None

def weighted_moment(values, weights, order: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    norm = np.sum(weights)
    if norm <= 0:
        return float("nan")
    return float(np.sum((values**order) * weights) / norm)


def shower_moments(shower):
    axis_override = get_axis_override(shower)
    long, radial, _ = longitudinal_radial_phi(shower, axis_override=axis_override)
    w = shower.E
    return {
        "longitudinal_m1": weighted_moment(long, w, 1),
        "longitudinal_m2": weighted_moment(long, w, 2),
        "radial_m1": weighted_moment(radial, w, 1),
        "radial_m2": weighted_moment(radial, w, 2),
    }
