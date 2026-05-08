from __future__ import annotations

from pathlib import Path

import numpy as np

from step2point.algorithms.base import CompressionAlgorithm
from step2point.core.results import CompressionResult
from step2point.core.shower import Shower
from step2point.geometry.dd4hep.bitfield import decode_dd4hep_cell_id
from step2point.geometry.dd4hep.factory_geometry import (
    BarrelLayout,
    barrel_module_basis,
    barrel_sensitive_plane_center_xy,
    build_barrel_layout_from_collection,
)


def _subcell_indices(offset: np.ndarray, pitch: float, bins: int) -> np.ndarray:
    sub_pitch = pitch / bins
    raw = np.floor((offset + 0.5 * pitch) / sub_pitch).astype(np.int32)
    return np.clip(raw, 0, bins - 1)


def _subcell_center(parent_index: np.ndarray, sub_index: np.ndarray, pitch: float, bins: int) -> np.ndarray:
    sub_pitch = pitch / bins
    return parent_index.astype(np.float64) * pitch + (-0.5 * pitch + (sub_index.astype(np.float64) + 0.5) * sub_pitch)


class MergeWithinRegularSubcell(CompressionAlgorithm):
    """Subdivide each detector cell into a regular x/y grid before merging.

    The detector cell is split into ``x_bins * y_bins`` subcells covering the
    full cell area. Deposits are grouped by ``(cell_id, sub_x, sub_y)``.
    """

    name = "merge_within_regular_subcell"

    def __init__(
        self,
        x_bins: int = 2,
        y_bins: int = 2,
        position_mode: str = "weighted",
        *,
        layout: BarrelLayout | None = None,
        compact_xml: str | Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        if x_bins <= 0 or y_bins <= 0:
            raise ValueError("x_bins and y_bins must be positive integers.")
        if position_mode not in {"weighted", "center"}:
            raise ValueError("position_mode must be 'weighted' or 'center'.")
        if layout is None:
            if compact_xml is None or collection_name is None:
                raise ValueError(
                    "MergeWithinRegularSubcell requires either a prebuilt layout or both compact_xml and collection_name."
                )
            layout = build_barrel_layout_from_collection(compact_xml, collection_name)
        if layout.segmentation_type != "CartesianGridXY":
            raise NotImplementedError("MergeWithinRegularSubcell currently supports only barrel CartesianGridXY layouts.")
        self.x_bins = int(x_bins)
        self.y_bins = int(y_bins)
        self.position_mode = position_mode
        self.layout = layout

    def compress(self, shower: Shower) -> CompressionResult:
        if shower.cell_id is None:
            raise ValueError("MergeWithinRegularSubcell requires cell_id.")
        if shower.n_points == 0:
            out = Shower(
                shower_id=shower.shower_id,
                x=shower.x.copy(),
                y=shower.y.copy(),
                z=shower.z.copy(),
                E=shower.E.copy(),
                t=None if shower.t is None else shower.t.copy(),
                cell_id=shower.cell_id.copy(),
                primary=shower.primary,
                metadata={
                    **shower.metadata,
                    "algorithm": self.name,
                    "position_mode": self.position_mode,
                    "x_bins": self.x_bins,
                    "y_bins": self.y_bins,
                },
            )
            return CompressionResult(
                shower=out,
                algorithm=self.name,
                debug_data={"cluster_label": np.empty(0, dtype=np.int64)},
            )
        return self._compress_barrel_xy(shower)

    def _compress_barrel_xy(self, shower: Shower) -> CompressionResult:
        n_points = shower.n_points
        decoded = [decode_dd4hep_cell_id(int(cell_id), self.layout.cell_id_encoding) for cell_id in shower.cell_id]
        modules = np.asarray([item["module"] for item in decoded], dtype=np.int32)
        layers = np.asarray([item["layer"] for item in decoded], dtype=np.int32)
        cell_x = np.asarray([item["x"] for item in decoded], dtype=np.int32)
        cell_y = np.asarray([item["y"] for item in decoded], dtype=np.int32)

        sub_x = np.empty(n_points, dtype=np.int32)
        sub_y = np.empty(n_points, dtype=np.int32)
        center_x = np.empty(n_points, dtype=np.float64)
        center_y = np.empty(n_points, dtype=np.float64)
        center_z = np.empty(n_points, dtype=np.float64)

        xy = np.stack([shower.x, shower.y], axis=1).astype(np.float64)
        unique_ml = np.unique(np.stack([modules, layers], axis=1), axis=0)
        for module_index, layer_index in unique_ml:
            mask = (modules == module_index) & (layers == layer_index)
            layer = self.layout.layers[layer_index - 1]
            sensitive_center_xy = barrel_sensitive_plane_center_xy(self.layout, int(layer_index), int(module_index))
            _, _, tangent = barrel_module_basis(self.layout, int(layer_index), int(module_index))

            tangent_local = (xy[mask] - sensitive_center_xy) @ tangent
            long_local = shower.z[mask].astype(np.float64)

            parent_tangent = cell_x[mask].astype(np.float64) * layer.pitch_tangent_mm
            parent_long = cell_y[mask].astype(np.float64) * layer.pitch_z_mm

            sub_x_mask = _subcell_indices(
                tangent_local - parent_tangent,
                layer.pitch_tangent_mm,
                self.x_bins,
            )
            sub_y_mask = _subcell_indices(
                long_local - parent_long,
                layer.pitch_z_mm,
                self.y_bins,
            )
            sub_x[mask] = sub_x_mask
            sub_y[mask] = sub_y_mask

            sub_tangent_center = _subcell_center(cell_x[mask], sub_x_mask, layer.pitch_tangent_mm, self.x_bins)
            sub_long_center = _subcell_center(cell_y[mask], sub_y_mask, layer.pitch_z_mm, self.y_bins)

            center_xy_mask = sensitive_center_xy + sub_tangent_center[:, None] * tangent[None, :]
            center_x[mask] = center_xy_mask[:, 0]
            center_y[mask] = center_xy_mask[:, 1]
            center_z[mask] = sub_long_center

        key_dtype = np.dtype([("cell_id", np.uint64), ("sub_x", np.int32), ("sub_y", np.int32)])
        keys = np.empty(n_points, dtype=key_dtype)
        keys["cell_id"] = shower.cell_id
        keys["sub_x"] = sub_x
        keys["sub_y"] = sub_y
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        n_out = len(unique_keys)

        e_sum = np.bincount(inverse, weights=shower.E, minlength=n_out)
        safe_e = np.where(e_sum > 0.0, e_sum, 1.0)
        if self.position_mode == "weighted":
            x_out = np.bincount(inverse, weights=shower.x * shower.E, minlength=n_out) / safe_e
            y_out = np.bincount(inverse, weights=shower.y * shower.E, minlength=n_out) / safe_e
            z_out = np.bincount(inverse, weights=shower.z * shower.E, minlength=n_out) / safe_e
        else:
            first_indices = np.full(n_out, -1, dtype=np.int32)
            for point_index, group_index in enumerate(inverse):
                if first_indices[group_index] < 0:
                    first_indices[group_index] = point_index
            x_out = center_x[first_indices]
            y_out = center_y[first_indices]
            z_out = center_z[first_indices]

        t_out = None
        if shower.t is not None:
            t_out = np.bincount(inverse, weights=shower.t * shower.E, minlength=n_out) / safe_e

        out = Shower(
            shower_id=shower.shower_id,
            x=x_out.astype(np.float32),
            y=y_out.astype(np.float32),
            z=z_out.astype(np.float32),
            E=e_sum.astype(np.float32),
            t=None if t_out is None else t_out.astype(np.float32),
            cell_id=unique_keys["cell_id"].astype(np.uint64),
            primary=shower.primary,
            metadata={
                **shower.metadata,
                "algorithm": self.name,
                "position_mode": self.position_mode,
                "x_bins": self.x_bins,
                "y_bins": self.y_bins,
                "collection_name": self.layout.collection_name,
            },
        )
        return CompressionResult(
            shower=out,
            algorithm=self.name,
            parameters={
                "x_bins": self.x_bins,
                "y_bins": self.y_bins,
                "position_mode": self.position_mode,
                "collection_name": self.layout.collection_name,
            },
            stats={
                "n_points_before": shower.n_points,
                "n_points_after": out.n_points,
                "compression_ratio": out.n_points / max(shower.n_points, 1),
                "energy_before": shower.total_energy,
                "energy_after": out.total_energy,
            },
            debug_data={"cluster_label": inverse.astype(np.int64, copy=False)},
        )
