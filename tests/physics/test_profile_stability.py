from __future__ import annotations

from pathlib import Path

import numpy as np

from step2point.algorithms.identity import IdentityCompression
from step2point.algorithms.merge_within_cell import MergeWithinCell
from step2point.io.step2point_hdf5 import Step2PointHDF5Reader
from step2point.metrics.spatial import estimate_shower_axis, longitudinal_radial_phi

DATA = Path(__file__).resolve().parents[1] / "data" / "tiny_showers.h5"
DATA_GAMMA = Path(__file__).resolve().parents[1] / "data" / "ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV.h5"


def _hist(values, weights, bins):
    hist, _ = np.histogram(values, bins=bins, weights=weights)
    return hist.astype(float)


def _normalized_l1(pre, post):
    denom = max(np.sum(np.abs(pre)), 1.0)
    return float(np.sum(np.abs(pre - post)) / denom)


def _profile_distance(shower, other):
    centroid, axis = estimate_shower_axis(shower)
    long0, radial0, _ = longitudinal_radial_phi(shower, centroid=centroid, axis=axis)
    long1, radial1, _ = longitudinal_radial_phi(other, centroid=centroid, axis=axis)
    long_bins = np.histogram_bin_edges(long0, bins=8)
    radial_bins = np.histogram_bin_edges(radial0, bins=8)
    d_long = _normalized_l1(_hist(long0, shower.E, long_bins), _hist(long1, other.E, long_bins))
    d_rad = _normalized_l1(_hist(radial0, shower.E, radial_bins), _hist(radial1, other.E, radial_bins))
    return d_long, d_rad


def test_identity_profile_distance_is_negligible():
    for shower in Step2PointHDF5Reader(str(DATA)).iter_showers():
        out = IdentityCompression().compress(shower).shower
        d_long, d_rad = _profile_distance(shower, out)
        assert d_long < 1e-12
        assert d_rad < 1e-12


def test_merge_within_cell_profile_distance_is_small_on_tiny_sample():
    algo = MergeWithinCell()
    for shower in Step2PointHDF5Reader(str(DATA)).iter_showers():
        out = algo.compress(shower).shower
        d_long, d_rad = _profile_distance(shower, out)
        assert d_long < 2.0
        assert d_rad < 1.0


def test_hdbscan_profile_distance_is_bounded():
    from step2point.algorithms.hdbscan_clustering import HDBSCANClustering

    algo = HDBSCANClustering(min_cluster_size=5, min_samples=3, use_time=True)
    for shower in Step2PointHDF5Reader(str(DATA_GAMMA), shower_limit=3).iter_showers():
        out = algo.compress(shower).shower
        d_long, d_rad = _profile_distance(shower, out)
        assert d_long < 3.0
        assert d_rad < 2.0


def test_cluster_within_cell_profile_distance_is_bounded():
    from sklearn.cluster import AgglomerativeClustering

    from step2point.algorithms.cluster_within_cell import ClusterWithinCell

    algo = ClusterWithinCell(
        clusterer=AgglomerativeClustering(n_clusters=None, distance_threshold=1.0),
    )
    for shower in Step2PointHDF5Reader(str(DATA_GAMMA), shower_limit=3).iter_showers():
        out = algo.compress(shower).shower
        d_long, d_rad = _profile_distance(shower, out)
        assert d_long < 3.0
        assert d_rad < 2.0
