from __future__ import annotations

import numpy as np


def estimate_shower_axis(shower):
    coords = np.stack([shower.x, shower.y, shower.z], axis=1)
    weights = np.asarray(shower.E, dtype=np.float64)
    centroid = np.average(coords, axis=0, weights=weights)
    centered = coords - centroid
    cov = np.cov(centered.T, aweights=weights)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / np.linalg.norm(axis)
    return centroid, axis


def longitudinal_radial_phi(shower, centroid=None, axis=None):
    if centroid is None or axis is None:
        centroid, axis = estimate_shower_axis(shower)
    coords = np.stack([shower.x, shower.y, shower.z], axis=1)
    rel = coords - centroid
    long = rel @ axis
    radial_vec = rel - np.outer(long, axis)
    radial = np.linalg.norm(radial_vec, axis=1)
    phi = np.arctan2(radial_vec[:, 1], radial_vec[:, 0])
    return long, radial, phi
