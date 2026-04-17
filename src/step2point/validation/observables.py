from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from step2point.metrics.spatial import estimate_shower_axis, longitudinal_radial_phi

SCALAR_KEYS = ("mean_long", "mean_r", "var_long", "var_r", "total_energy", "num_steps")


def _weighted_moment(values: np.ndarray, weights: np.ndarray, order: int) -> float:
    if values.size == 0 or np.sum(weights) <= 0.0:
        return float("nan")
    return float(np.average(values**order, weights=weights))


def _safe_log10(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log10(np.clip(np.asarray(values, dtype=np.float64), eps, None))


def compute_shower_observables(shower, *, axis_override=None) -> dict[str, object]:
    """Compute summary observables for one shower.

    The default axis comes from PCA-based shower-axis estimation. Pass
    `axis_override` only when a manual physics-motivated direction should
    replace that default.
    """
    centroid, axis = estimate_shower_axis(shower, axis_override=axis_override)
    long, radial, phi = longitudinal_radial_phi(
        shower,
        centroid=centroid,
        axis=axis,
        longitudinal_origin="first_deposit",
    )
    weights = np.asarray(shower.E, dtype=np.float64)
    mean_long = _weighted_moment(long, weights, 1)
    mean_r = _weighted_moment(radial, weights, 1)
    var_long = float(np.average((long - mean_long) ** 2, weights=weights)) if long.size else float("nan")
    var_r = float(np.average((radial - mean_r) ** 2, weights=weights)) if radial.size else float("nan")
    return {
        "long_values": long,
        "radial_values": radial,
        "phi_values": phi,
        "log_energy_values": _safe_log10(weights),
        "weights": weights,
        "long_profile": np.histogram(long, bins=30, weights=weights),
        "r_profile": np.histogram(radial, bins=50, weights=weights),
        "phi_profile": np.histogram(phi, bins=np.linspace(-np.pi, np.pi, 51), weights=weights),
        "log_energy": np.histogram(_safe_log10(weights), bins=50),
        "mean_long": mean_long,
        "mean_r": mean_r,
        "var_long": var_long,
        "var_r": var_r,
        "total_energy": float(np.sum(weights)),
        "num_steps": int(len(weights)),
        "axis": axis,
        "centroid": centroid,
    }


def aggregate_observables(observables: Iterable[dict[str, object]]) -> dict[str, list[float]]:
    summary = {key: [] for key in SCALAR_KEYS}
    for row in observables:
        for key in SCALAR_KEYS:
            summary[key].append(float(row[key]))
    return summary
