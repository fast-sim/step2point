from importlib.metadata import PackageNotFoundError, version

from step2point.algorithms.identity import IdentityCompression
from step2point.algorithms.merge_within_cell import MergeWithinCell
from step2point.algorithms.merge_within_regular_subcell import MergeWithinRegularSubcell
from step2point.core.shower import Shower

from step2point.algorithms.hdbscan_clustering import HDBSCANClustering

try:
    __version__ = version("step2point")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    "Shower",
    "IdentityCompression",
    "MergeWithinCell",
    "MergeWithinRegularSubcell",
    "HDBSCANClustering",
]
