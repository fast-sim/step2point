from __future__ import annotations

import numpy as np


def _normalized_axis(axis) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    if axis.shape != (3,):
        raise ValueError(f"Axis must be a length-3 vector, got shape {axis.shape}.")
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Axis must have finite, non-zero norm.")
    return axis / norm


def _preferred_axis_orientation(shower) -> np.ndarray:
    momentum = shower.primary.get("momentum")
    if momentum is not None:
        momentum = np.asarray(momentum, dtype=np.float64)
        norm = np.linalg.norm(momentum)
        if momentum.shape == (3,) and np.isfinite(norm) and norm > 0.0:
            return momentum / norm
    return np.array([1.0, 1.0, 1.0], dtype=np.float64)


def _orient_axis(axis: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if np.dot(axis, reference) < 0.0:
        return -axis
    return axis


def _transverse_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(ref, axis)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = ref - np.dot(ref, axis) * axis
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return e1, e2


def estimate_shower_axis(
    shower,
    *,
    axis_override=None,
    refine: bool = True,
    trim_percentiles: tuple[float, float] = (10.0, 99.0),
):
    """Estimate the shower axis.

    By default this uses an energy-weighted PCA axis, optionally refined after
    trimming extreme longitudinal outliers. A manual `axis_override` can be
    supplied for detector-specific studies where the PCA direction should be
    replaced explicitly.
    """
    coords = np.stack([shower.x, shower.y, shower.z], axis=1)
    weights = np.asarray(shower.E, dtype=np.float64)
    centroid = np.average(coords, axis=0, weights=weights)
    if axis_override is not None:
        return centroid, _normalized_axis(axis_override)

    centered = coords - centroid
    cov = np.cov(centered.T, aweights=weights)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = _normalized_axis(axis)
    axis = _orient_axis(axis, _preferred_axis_orientation(shower))

    if refine and len(coords) >= 3:
        long0 = centered @ axis
        q_low, q_high = np.percentile(long0, trim_percentiles)
        mask = (long0 >= q_low) & (long0 <= q_high)
        if np.count_nonzero(mask) >= 3:
            refined_coords = centered[mask]
            refined_weights = weights[mask]
            cov_refined = np.cov(refined_coords.T, aweights=refined_weights)
            eigvals, eigvecs = np.linalg.eigh(cov_refined)
            axis = eigvecs[:, np.argmax(eigvals)]
            axis = _normalized_axis(axis)
            axis = _orient_axis(axis, _preferred_axis_orientation(shower))
    return centroid, axis


def longitudinal_radial_phi(
    shower,
    centroid=None,
    axis=None,
    *,
    axis_override=None,
    longitudinal_origin: str = "centroid",
    shift_longitudinal_min: bool = False,
):
    """Return cylindrical shower coordinates around the shower axis.

    If `centroid` and `axis` are not provided, the axis is estimated with
    `estimate_shower_axis`, which means PCA is the default and
    `axis_override` is optional.
    """
    if centroid is None or axis is None:
        centroid, axis = estimate_shower_axis(shower, axis_override=axis_override)
    else:
        axis = _normalized_axis(axis)
    coords = np.stack([shower.x, shower.y, shower.z], axis=1)
    rel = coords - centroid
    long = rel @ axis
    radial_vec = rel - np.outer(long, axis)
    radial = np.linalg.norm(radial_vec, axis=1)
    e1, e2 = _transverse_basis(axis)
    phi = np.arctan2(radial_vec @ e2, radial_vec @ e1)
    if shift_longitudinal_min:
        longitudinal_origin = "min_projection"
    if longitudinal_origin not in {"centroid", "first_deposit", "min_projection"}:
        raise ValueError(f"Unsupported longitudinal_origin: {longitudinal_origin}")
    if longitudinal_origin == "first_deposit" and long.size:
        long = long - long[0]
    if longitudinal_origin == "min_projection" and long.size:
        long = long - np.min(long)
    return long, radial, phi
