from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CellIDField:
    name: str
    offset: int
    width: int
    signed: bool


def parse_dd4hep_id_encoding(id_encoding: str) -> tuple[CellIDField, ...]:
    fields: list[CellIDField] = []
    next_offset = 0
    for raw_field in id_encoding.split(","):
        token = raw_field.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) == 2:
            name, width_spec = parts
            offset = next_offset
        elif len(parts) == 3:
            name, offset_spec, width_spec = parts
            offset = int(offset_spec)
        else:
            raise ValueError(f"Unsupported DD4hep id field specification: {token!r}")
        signed = int(width_spec) < 0
        width = abs(int(width_spec))
        fields.append(CellIDField(name=name, offset=offset, width=width, signed=signed))
        next_offset = offset + width
    return tuple(fields)


def decode_dd4hep_cell_id(cell_id: int, id_encoding: str) -> dict[str, int]:
    decoded: dict[str, int] = {}
    for field in parse_dd4hep_id_encoding(id_encoding):
        raw_value = (int(cell_id) >> field.offset) & ((1 << field.width) - 1)
        if field.signed and raw_value >= (1 << (field.width - 1)):
            raw_value -= 1 << field.width
        decoded[field.name] = int(raw_value)
    return decoded


def extract_field(cell_ids: np.ndarray, id_encoding: str, field_name: str = "layer") -> np.ndarray:
    """Extract a single field from an array of cell IDs (vectorized).

    Parameters
    ----------
    cell_ids : np.ndarray
        Array of cell IDs.
    id_encoding : str
        DD4hep cell ID encoding string.
    field_name : str
        Name of the field to extract.
    """
    fields = parse_dd4hep_id_encoding(id_encoding)
    for f in fields:
        if f.name == field_name:
            return (cell_ids.astype(np.int64) >> f.offset) & ((1 << f.width) - 1)
    raise ValueError(f"Field {field_name!r} not found in encoding. Available: {[f.name for f in fields]}")
