from __future__ import annotations

from dataclasses import dataclass

from step2point.core.results import CompressionResult
from step2point.core.shower import Shower

try:
    import _step2point_cpp as _cpp
except ImportError:  # pragma: no cover - optional backend
    _cpp = None


def cpp_available() -> bool:
    return _cpp is not None


def to_cpp_shower(shower: Shower):
    if _cpp is None:
        raise RuntimeError("_step2point_cpp is not available")
    out = _cpp.CppShower()
    out.shower_id = int(shower.shower_id)
    out.x = shower.x.tolist()
    out.y = shower.y.tolist()
    out.z = shower.z.tolist()
    out.E = shower.E.tolist()
    out.t = None if shower.t is None else shower.t.tolist()
    out.cell_id = None if shower.cell_id is None else shower.cell_id.tolist()
    return out


def from_cpp_shower(shower, *, primary=None, metadata=None) -> Shower:
    return Shower(
        shower_id=int(shower.shower_id),
        x=shower.x,
        y=shower.y,
        z=shower.z,
        E=shower.E,
        t=shower.t,
        cell_id=shower.cell_id,
        primary={} if primary is None else dict(primary),
        metadata={} if metadata is None else dict(metadata),
    )


def merge_within_cell(shower: Shower) -> CompressionResult:
    if _cpp is None:
        raise RuntimeError("_step2point_cpp is not available")
    cpp_result = _cpp.merge_within_cell(to_cpp_shower(shower))
    out = from_cpp_shower(
        cpp_result.shower,
        primary=shower.primary,
        metadata={**shower.metadata, "algorithm": "merge_within_cell", "backend": "cpp"},
    )
    return CompressionResult(
        shower=out,
        algorithm=cpp_result.algorithm,
        stats={
            "n_points_before": cpp_result.stats.n_points_before,
            "n_points_after": cpp_result.stats.n_points_after,
            "compression_ratio": cpp_result.stats.compression_ratio,
            "energy_before": cpp_result.stats.energy_before,
            "energy_after": cpp_result.stats.energy_after,
        },
    )
