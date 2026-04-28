from __future__ import annotations

import ast
import operator
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_UNITS = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "um": 1e-3,
    "nm": 1e-6,
    "tesla": 1.0,
    "deg": np.pi / 180.0,
    "rad": 1.0,
}


def _eval_expr(expr: str, names: dict[str, float]) -> float:
    node = ast.parse(expr, mode="eval")

    def _visit(current: ast.AST) -> float:
        if isinstance(current, ast.Expression):
            return _visit(current.body)
        if isinstance(current, ast.Constant):
            return float(current.value)
        if isinstance(current, ast.Name):
            if current.id in names:
                return float(names[current.id])
            if current.id in _UNITS:
                return _UNITS[current.id]
            raise KeyError(current.id)
        if isinstance(current, ast.BinOp) and type(current.op) in _BIN_OPS:
            return _BIN_OPS[type(current.op)](_visit(current.left), _visit(current.right))
        if isinstance(current, ast.UnaryOp) and type(current.op) in _UNARY_OPS:
            return _UNARY_OPS[type(current.op)](_visit(current.operand))
        raise ValueError(f"Unsupported expression: {expr!r}")

    return float(_visit(node))


@dataclass(frozen=True, slots=True)
class XMLNodeRef:
    path: Path
    element: ET.Element


@dataclass(frozen=True, slots=True)
class LayerSpec:
    repeat: int
    thickness_mm: float
    sensitive_thickness_mm: float
    sensitive_center_offset_mm: float


@dataclass(frozen=True, slots=True)
class BarrelLayerGeometry:
    layer_index: int
    layer_center_radius_mm: float
    half_thickness_mm: float
    sensitive_radius_mm: float
    sensitive_half_thickness_mm: float
    half_tangent_mm: float
    half_z_mm: float
    pitch_tangent_mm: float
    pitch_z_mm: float
    modules: int
    module_angles_rad: tuple[float, ...]
    module_centers_xy_mm: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class BarrelLayout:
    collection_name: str
    detector_name: str
    readout_xml_path: str
    detector_xml_path: str
    segmentation_type: str
    numsides: int
    det_z_mm: float
    rmin_mm: float
    gap_mm: float
    total_thickness_mm: float
    inner_angle_rad: float
    inner_face_length_mm: float
    outer_face_length_mm: float
    sect_center_radius_mm: float
    envelope_rotation_z_rad: float
    pitch_tangent_mm: float
    pitch_z_mm: float
    cell_id_encoding: str
    layers: tuple[BarrelLayerGeometry, ...]


class DD4hepResolver:
    def __init__(self, main_xml: str | Path):
        self.main_xml = Path(main_xml).resolve()
        self._roots: dict[Path, ET.Element] = {}
        self._node_paths: list[XMLNodeRef] = []
        self.constants = self._collect_all()

    def _load_recursive(self, path: Path) -> None:
        path = path.resolve()
        if path in self._roots:
            return
        root = ET.parse(path).getroot()
        self._roots[path] = root
        self._node_paths.append(XMLNodeRef(path=path, element=root))
        for include in root.findall(".//include"):
            ref = include.attrib.get("ref")
            if ref:
                self._load_recursive((path.parent / ref).resolve())

    def _collect_all(self) -> dict[str, float]:
        self._load_recursive(self.main_xml)
        pending: dict[str, str] = {}
        for root in self._roots.values():
            for const in root.findall(".//define/constant"):
                pending[const.attrib["name"]] = const.attrib["value"]

        constants: dict[str, float] = {}
        while pending:
            progress = False
            for name, expr in list(pending.items()):
                try:
                    constants[name] = _eval_expr(expr.replace("^", "**"), constants)
                except KeyError:
                    continue
                del pending[name]
                progress = True
            if not progress:
                raise ValueError(f"Unresolved constants: {sorted(pending)}")
        return constants

    def find_readout(self, name: str) -> XMLNodeRef:
        for path, root in self._roots.items():
            for readout in root.findall(".//readouts/readout"):
                if readout.attrib.get("name") == name:
                    return XMLNodeRef(path=path, element=readout)
        raise KeyError(f"Readout {name!r} not found under {self.main_xml}")

    def find_detector_for_readout(self, readout_name: str) -> XMLNodeRef:
        for path, root in self._roots.items():
            for detector in root.findall(".//detectors/detector"):
                if detector.attrib.get("readout") == readout_name:
                    return XMLNodeRef(path=path, element=detector)
        raise KeyError(f"Detector using readout {readout_name!r} not found under {self.main_xml}")


def _rotation_matrix_xyz(z: float, y: float, x: float) -> np.ndarray:
    cz, sz = np.cos(z), np.sin(z)
    cy, sy = np.cos(y), np.sin(y)
    cx, sx = np.cos(x), np.sin(x)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx


def _layer_specs(detector: ET.Element, constants: dict[str, float]) -> list[LayerSpec]:
    specs: list[LayerSpec] = []
    for layer_elem in detector.findall("layer"):
        repeat = int(layer_elem.attrib["repeat"])
        thickness = 0.0
        sensitive_thickness = 0.0
        sensitive_center = 0.0
        offset = 0.0
        for slice_elem in layer_elem.findall("slice"):
            slice_thickness = _eval_expr(slice_elem.attrib["thickness"], constants)
            if slice_elem.attrib.get("sensitive") == "yes":
                sensitive_thickness = slice_thickness
                sensitive_center = offset + 0.5 * slice_thickness
            offset += slice_thickness
            thickness += slice_thickness
        specs.append(
            LayerSpec(
                repeat=repeat,
                thickness_mm=thickness,
                sensitive_thickness_mm=sensitive_thickness,
                sensitive_center_offset_mm=sensitive_center,
            )
        )
    return specs


def build_barrel_layout_from_collection(main_xml: str | Path, collection_name: str) -> BarrelLayout:
    resolver = DD4hepResolver(main_xml)
    readout_ref = resolver.find_readout(collection_name)
    detector_ref = resolver.find_detector_for_readout(collection_name)
    readout = readout_ref.element
    detector = detector_ref.element

    if detector.attrib.get("type") != "ODDPolyhedraBarrelCalorimeter":
        raise NotImplementedError(
            f"Only ODDPolyhedraBarrelCalorimeter is implemented in this prototype, got {detector.attrib.get('type')!r}"
        )

    seg = readout.find("segmentation")
    if seg is None or seg.attrib.get("type") != "CartesianGridXY":
        raise NotImplementedError("This prototype currently supports only barrel CartesianGridXY readouts")
    id_encoding = (readout.findtext("id") or "").strip()
    if not id_encoding:
        raise ValueError(f"Readout {collection_name!r} does not define an <id> encoding")

    pitch_tangent = _eval_expr(seg.attrib["grid_size_x"], resolver.constants)
    pitch_z = _eval_expr(seg.attrib["grid_size_y"], resolver.constants)

    dim = detector.find("dimensions")
    if dim is None:
        raise ValueError("Detector dimensions not found")
    numsides = int(_eval_expr(dim.attrib["numsides"], resolver.constants))
    det_z = _eval_expr(dim.attrib["z"], resolver.constants)
    rmin = _eval_expr(dim.attrib["rmin"], resolver.constants)
    gap = _eval_expr(detector.attrib.get("gap", "0"), resolver.constants)

    specs = _layer_specs(detector, resolver.constants)
    total_thickness = sum(spec.repeat * spec.thickness_mm for spec in specs)

    inner_angle = 2.0 * np.pi / numsides
    half_inner_angle = inner_angle / 2.0
    inner_face_len = rmin * np.tan(half_inner_angle) * 2.0
    outer_face_len = (rmin + total_thickness) * np.tan(half_inner_angle) * 2.0
    layer_inner_angle = np.pi / 2.0 - (np.pi - inner_angle) / 2.0
    layer_dim_x = inner_face_len / 2.0 - gap * 2.0
    layer_pos_z = -(total_thickness / 2.0)

    sect_center_radius = rmin + total_thickness / 2.0
    rot_y = inner_angle / 2.0
    module_angles: list[float] = []
    module_centers: list[tuple[float, float]] = []
    for _module in range(1, numsides + 1):
        pos_x = -sect_center_radius * np.sin(rot_y)
        pos_y = sect_center_radius * np.cos(rot_y)
        translation = np.array([-pos_x, -pos_y, 0.0], dtype=np.float64)
        module_angles.append(rot_y)
        module_centers.append((float(translation[0]), float(translation[1])))
        rot_y -= inner_angle

    layers: list[BarrelLayerGeometry] = []
    layer_index = 1
    for spec in specs:
        for _ in range(spec.repeat):
            layer_pos_z += spec.thickness_mm / 2.0
            layer_center_radius = sect_center_radius + layer_pos_z
            sensitive_local_z = layer_pos_z - spec.thickness_mm / 2.0 + spec.sensitive_center_offset_mm
            sensitive_radius = sect_center_radius + sensitive_local_z
            layers.append(
                BarrelLayerGeometry(
                    layer_index=layer_index,
                    layer_center_radius_mm=float(layer_center_radius),
                    half_thickness_mm=float(spec.thickness_mm / 2.0),
                    sensitive_radius_mm=float(sensitive_radius),
                    sensitive_half_thickness_mm=float(spec.sensitive_thickness_mm / 2.0),
                    half_tangent_mm=float(layer_dim_x),
                    half_z_mm=float(det_z / 2.0),
                    pitch_tangent_mm=float(pitch_tangent),
                    pitch_z_mm=float(pitch_z),
                    modules=numsides,
                    module_angles_rad=tuple(module_angles),
                    module_centers_xy_mm=tuple(module_centers),
                )
            )
            layer_dim_x += spec.thickness_mm * np.tan(layer_inner_angle)
            layer_pos_z += spec.thickness_mm / 2.0
            layer_index += 1

    return BarrelLayout(
        collection_name=collection_name,
        detector_name=detector.attrib["name"],
        readout_xml_path=str(readout_ref.path),
        detector_xml_path=str(detector_ref.path),
        segmentation_type=seg.attrib["type"],
        numsides=numsides,
        det_z_mm=float(det_z),
        rmin_mm=float(rmin),
        gap_mm=float(gap),
        total_thickness_mm=float(total_thickness),
        inner_angle_rad=float(inner_angle),
        inner_face_length_mm=float(inner_face_len),
        outer_face_length_mm=float(outer_face_len),
        sect_center_radius_mm=float(sect_center_radius),
        envelope_rotation_z_rad=float(np.pi / numsides),
        pitch_tangent_mm=float(pitch_tangent),
        pitch_z_mm=float(pitch_z),
        cell_id_encoding=id_encoding,
        layers=tuple(layers),
    )


def barrel_module_basis(
    layout: BarrelLayout,
    layer_index: int,
    module_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layer = layout.layers[layer_index - 1]
    raw_center_xy = np.array(layer.module_centers_xy_mm[module_index - 1], dtype=np.float64)
    cz = np.cos(layout.envelope_rotation_z_rad)
    sz = np.sin(layout.envelope_rotation_z_rad)
    envelope_rotation = np.array([[cz, -sz], [sz, cz]], dtype=np.float64)
    center_xy = envelope_rotation @ raw_center_xy
    radial = center_xy / np.linalg.norm(center_xy)
    tangent = np.array([-radial[1], radial[0]], dtype=np.float64)
    return center_xy, radial, tangent


def barrel_sensitive_plane_center_xy(
    layout: BarrelLayout,
    layer_index: int,
    module_index: int,
) -> np.ndarray:
    layer = layout.layers[layer_index - 1]
    center_xy, radial, _ = barrel_module_basis(layout, layer_index, module_index)
    radial_offset = layer.sensitive_radius_mm - layout.sect_center_radius_mm
    return center_xy + radial_offset * radial


def barrel_cell_center(
    layout: BarrelLayout,
    layer_index: int,
    module_index: int,
    cell_x: int,
    cell_y: int,
) -> np.ndarray:
    layer = layout.layers[layer_index - 1]
    sensitive_center_xy = barrel_sensitive_plane_center_xy(layout, layer_index, module_index)
    _, _, tangent = barrel_module_basis(layout, layer_index, module_index)
    xy = sensitive_center_xy + float(cell_x) * layer.pitch_tangent_mm * tangent
    z = float(cell_y) * layer.pitch_z_mm
    return np.array([xy[0], xy[1], z], dtype=np.float64)


def module_grid_lines_xy_zy(
    layout: BarrelLayout,
    layer_index: int,
    module_index: int | None = None,
    sensitive_only: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    layer = layout.layers[layer_index - 1]
    x_lines = np.arange(-layer.half_tangent_mm, layer.half_tangent_mm + 0.5 * layer.pitch_tangent_mm, layer.pitch_tangent_mm)
    z_lines = np.arange(-layer.half_z_mm, layer.half_z_mm + 0.5 * layer.pitch_z_mm, layer.pitch_z_mm)

    xy_segments: list[np.ndarray] = []
    zy_segments: list[np.ndarray] = []
    module_pairs = list(zip(layer.module_angles_rad, layer.module_centers_xy_mm))
    if module_index is not None:
        module_pairs = [module_pairs[module_index - 1]]
    cz = np.cos(layout.envelope_rotation_z_rad)
    sz = np.sin(layout.envelope_rotation_z_rad)
    envelope_rotation = np.array([[cz, -sz], [sz, cz]], dtype=np.float64)

    for _, raw_center_xy in module_pairs:
        center_xy = envelope_rotation @ np.array(raw_center_xy, dtype=np.float64)
        radial = center_xy / np.linalg.norm(center_xy)
        tangent = np.array([-radial[1], radial[0]], dtype=np.float64)
        radial_offset = layer.sensitive_radius_mm - layout.sect_center_radius_mm
        sensitive_center_xy = center_xy + radial_offset * radial
        radial_half_extent = layer.sensitive_half_thickness_mm if sensitive_only else layer.half_thickness_mm
        radial_center_xy = (
            sensitive_center_xy
            if sensitive_only
            else center_xy + (layer.layer_center_radius_mm - layout.sect_center_radius_mm) * radial
        )

        for x in x_lines:
            p0_xy = radial_center_xy + x * tangent - radial_half_extent * radial
            p1_xy = radial_center_xy + x * tangent + radial_half_extent * radial
            xy_segments.append(np.array([[p0_xy[0], p0_xy[1]], [p1_xy[0], p1_xy[1]]], dtype=np.float64))

        for z in z_lines:
            p0_xy = sensitive_center_xy - layer.half_tangent_mm * tangent
            p1_xy = sensitive_center_xy + layer.half_tangent_mm * tangent
            zy_segments.append(np.array([[z, p0_xy[1]], [z, p1_xy[1]]], dtype=np.float64))

    return xy_segments, zy_segments


def module_cell_strip_polygons_xy(
    layout: BarrelLayout,
    layer_index: int,
    module_index: int | None = None,
    sensitive_only: bool = False,
) -> list[np.ndarray]:
    layer = layout.layers[layer_index - 1]

    polygons: list[np.ndarray] = []
    module_pairs = list(zip(layer.module_angles_rad, layer.module_centers_xy_mm))
    if module_index is not None:
        module_pairs = [module_pairs[module_index - 1]]

    cz = np.cos(layout.envelope_rotation_z_rad)
    sz = np.sin(layout.envelope_rotation_z_rad)
    envelope_rotation = np.array([[cz, -sz], [sz, cz]], dtype=np.float64)

    radial_half_extent = layer.sensitive_half_thickness_mm if sensitive_only else layer.half_thickness_mm

    max_x_index = int(np.ceil(layer.half_tangent_mm / layer.pitch_tangent_mm - 0.5))
    x_centers = np.arange(-max_x_index, max_x_index + 1, dtype=np.int32) * layer.pitch_tangent_mm

    for _, raw_center_xy in module_pairs:
        center_xy = envelope_rotation @ np.array(raw_center_xy, dtype=np.float64)
        radial = center_xy / np.linalg.norm(center_xy)
        tangent = np.array([-radial[1], radial[0]], dtype=np.float64)
        radial_center_xy = (
            center_xy + (layer.sensitive_radius_mm - layout.sect_center_radius_mm) * radial
            if sensitive_only
            else center_xy + (layer.layer_center_radius_mm - layout.sect_center_radius_mm) * radial
        )

        for x_center in x_centers:
            x0 = max(float(x_center - 0.5 * layer.pitch_tangent_mm), -layer.half_tangent_mm)
            x1 = min(float(x_center + 0.5 * layer.pitch_tangent_mm), layer.half_tangent_mm)
            if x1 <= x0:
                continue
            polygons.append(
                np.array(
                    [
                        radial_center_xy + x0 * tangent - radial_half_extent * radial,
                        radial_center_xy + x1 * tangent - radial_half_extent * radial,
                        radial_center_xy + x1 * tangent + radial_half_extent * radial,
                        radial_center_xy + x0 * tangent + radial_half_extent * radial,
                    ],
                    dtype=np.float64,
                )
            )

    return polygons


def module_cell_strip_polygons_zy(
    layout: BarrelLayout,
    layer_index: int,
    module_index: int | None = None,
    sensitive_only: bool = False,
) -> list[np.ndarray]:
    layer = layout.layers[layer_index - 1]

    polygons: list[np.ndarray] = []
    module_pairs = list(zip(layer.module_angles_rad, layer.module_centers_xy_mm))
    if module_index is not None:
        module_pairs = [module_pairs[module_index - 1]]

    cz = np.cos(layout.envelope_rotation_z_rad)
    sz = np.sin(layout.envelope_rotation_z_rad)
    envelope_rotation = np.array([[cz, -sz], [sz, cz]], dtype=np.float64)

    radial_half_extent = layer.sensitive_half_thickness_mm if sensitive_only else layer.half_thickness_mm

    max_z_index = int(np.ceil(layer.half_z_mm / layer.pitch_z_mm - 0.5))
    z_centers = np.arange(-max_z_index, max_z_index + 1, dtype=np.int32) * layer.pitch_z_mm

    for _, raw_center_xy in module_pairs:
        center_xy = envelope_rotation @ np.array(raw_center_xy, dtype=np.float64)
        radial = center_xy / np.linalg.norm(center_xy)
        tangent = np.array([-radial[1], radial[0]], dtype=np.float64)
        radial_center_xy = (
            center_xy + (layer.sensitive_radius_mm - layout.sect_center_radius_mm) * radial
            if sensitive_only
            else center_xy + (layer.layer_center_radius_mm - layout.sect_center_radius_mm) * radial
        )

        corners_xy = np.array(
            [
                radial_center_xy - layer.half_tangent_mm * tangent - radial_half_extent * radial,
                radial_center_xy + layer.half_tangent_mm * tangent - radial_half_extent * radial,
                radial_center_xy + layer.half_tangent_mm * tangent + radial_half_extent * radial,
                radial_center_xy - layer.half_tangent_mm * tangent + radial_half_extent * radial,
            ],
            dtype=np.float64,
        )
        ymin = float(np.min(corners_xy[:, 1]))
        ymax = float(np.max(corners_xy[:, 1]))

        for z_center in z_centers:
            z0 = max(float(z_center - 0.5 * layer.pitch_z_mm), -layer.half_z_mm)
            z1 = min(float(z_center + 0.5 * layer.pitch_z_mm), layer.half_z_mm)
            if z1 <= z0:
                continue
            polygons.append(
                np.array(
                    [
                        [z0, ymin],
                        [z1, ymin],
                        [z1, ymax],
                        [z0, ymax],
                    ],
                    dtype=np.float64,
                )
            )

    return polygons


def module_cell_strip_polygons_xz(
    layout: BarrelLayout,
    layer_index: int,
    module_index: int | None = None,
    sensitive_only: bool = False,
) -> list[np.ndarray]:
    layer = layout.layers[layer_index - 1]

    polygons: list[np.ndarray] = []
    module_pairs = list(zip(layer.module_angles_rad, layer.module_centers_xy_mm))
    if module_index is not None:
        module_pairs = [module_pairs[module_index - 1]]

    cz = np.cos(layout.envelope_rotation_z_rad)
    sz = np.sin(layout.envelope_rotation_z_rad)
    envelope_rotation = np.array([[cz, -sz], [sz, cz]], dtype=np.float64)

    radial_half_extent = layer.sensitive_half_thickness_mm if sensitive_only else layer.half_thickness_mm

    max_x_index = int(np.ceil(layer.half_tangent_mm / layer.pitch_tangent_mm - 0.5))
    x_centers = np.arange(-max_x_index, max_x_index + 1, dtype=np.int32) * layer.pitch_tangent_mm
    max_z_index = int(np.ceil(layer.half_z_mm / layer.pitch_z_mm - 0.5))
    z_centers = np.arange(-max_z_index, max_z_index + 1, dtype=np.int32) * layer.pitch_z_mm

    for _, raw_center_xy in module_pairs:
        center_xy = envelope_rotation @ np.array(raw_center_xy, dtype=np.float64)
        radial = center_xy / np.linalg.norm(center_xy)
        tangent = np.array([-radial[1], radial[0]], dtype=np.float64)
        radial_center_xy = (
            center_xy + (layer.sensitive_radius_mm - layout.sect_center_radius_mm) * radial
            if sensitive_only
            else center_xy + (layer.layer_center_radius_mm - layout.sect_center_radius_mm) * radial
        )

        for x_center in x_centers:
            x0 = max(float(x_center - 0.5 * layer.pitch_tangent_mm), -layer.half_tangent_mm)
            x1 = min(float(x_center + 0.5 * layer.pitch_tangent_mm), layer.half_tangent_mm)
            if x1 <= x0:
                continue
            p00 = radial_center_xy + x0 * tangent - radial_half_extent * radial
            p01 = radial_center_xy + x1 * tangent - radial_half_extent * radial
            p11 = radial_center_xy + x1 * tangent + radial_half_extent * radial
            p10 = radial_center_xy + x0 * tangent + radial_half_extent * radial
            for z_center in z_centers:
                z0 = max(float(z_center - 0.5 * layer.pitch_z_mm), -layer.half_z_mm)
                z1 = min(float(z_center + 0.5 * layer.pitch_z_mm), layer.half_z_mm)
                if z1 <= z0:
                    continue
                polygons.append(
                    np.array(
                        [
                            [p00[0], z0],
                            [p01[0], z0],
                            [p11[0], z1],
                            [p10[0], z1],
                        ],
                        dtype=np.float64,
                    )
                )

    return polygons


def module_layer_outline_xy_xz_zy(
    layout: BarrelLayout,
    layer_index: int,
    module_index: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    layer = layout.layers[layer_index - 1]
    xy_segments: list[np.ndarray] = []
    xz_segments: list[np.ndarray] = []
    zy_segments: list[np.ndarray] = []
    cross_section = np.array(
        [
            [-layer.half_tangent_mm, -layer.half_thickness_mm],
            [+layer.half_tangent_mm, -layer.half_thickness_mm],
            [+layer.half_tangent_mm, +layer.half_thickness_mm],
            [-layer.half_tangent_mm, +layer.half_thickness_mm],
        ],
        dtype=np.float64,
    )
    module_pairs = list(zip(layer.module_angles_rad, layer.module_centers_xy_mm))
    if module_index is not None:
        module_pairs = [module_pairs[module_index - 1]]

    cz = np.cos(layout.envelope_rotation_z_rad)
    sz = np.sin(layout.envelope_rotation_z_rad)
    envelope_rotation = np.array([[cz, -sz], [sz, cz]], dtype=np.float64)

    for _, raw_center_xy in module_pairs:
        center_xy = envelope_rotation @ np.array(raw_center_xy, dtype=np.float64)
        radial = center_xy / np.linalg.norm(center_xy)
        tangent = np.array([-radial[1], radial[0]], dtype=np.float64)
        radial_offset = layer.layer_center_radius_mm - layout.sect_center_radius_mm
        layer_center_xy = center_xy + radial_offset * radial

        xy_corners: list[np.ndarray] = []
        front_face_xz: list[np.ndarray] = []
        back_face_xz: list[np.ndarray] = []
        front_face: list[np.ndarray] = []
        back_face: list[np.ndarray] = []
        for tangent_local, radial_local in cross_section:
            xy = layer_center_xy + tangent_local * tangent + radial_local * radial
            xy_corners.append(xy)
            front_face_xz.append(np.array([xy[0], +layer.half_z_mm], dtype=np.float64))
            back_face_xz.append(np.array([xy[0], -layer.half_z_mm], dtype=np.float64))
            front_face.append(np.array([+layer.half_z_mm, xy[1]], dtype=np.float64))
            back_face.append(np.array([-layer.half_z_mm, xy[1]], dtype=np.float64))

        for i in range(4):
            p0_xy = xy_corners[i]
            p1_xy = xy_corners[(i + 1) % 4]
            xy_segments.append(np.array([[p0_xy[0], p0_xy[1]], [p1_xy[0], p1_xy[1]]], dtype=np.float64))

            p0_front = front_face[i]
            p1_front = front_face[(i + 1) % 4]
            p0_back = back_face[i]
            p1_back = back_face[(i + 1) % 4]
            p0_front_xz = front_face_xz[i]
            p1_front_xz = front_face_xz[(i + 1) % 4]
            p0_back_xz = back_face_xz[i]
            p1_back_xz = back_face_xz[(i + 1) % 4]
            xz_segments.append(np.array([[p0_front_xz[0], p0_front_xz[1]], [p1_front_xz[0], p1_front_xz[1]]], dtype=np.float64))
            xz_segments.append(np.array([[p0_back_xz[0], p0_back_xz[1]], [p1_back_xz[0], p1_back_xz[1]]], dtype=np.float64))
            xz_segments.append(np.array([[p0_back_xz[0], p0_back_xz[1]], [p0_front_xz[0], p0_front_xz[1]]], dtype=np.float64))
            zy_segments.append(np.array([[p0_front[0], p0_front[1]], [p1_front[0], p1_front[1]]], dtype=np.float64))
            zy_segments.append(np.array([[p0_back[0], p0_back[1]], [p1_back[0], p1_back[1]]], dtype=np.float64))
            zy_segments.append(np.array([[p0_back[0], p0_back[1]], [p0_front[0], p0_front[1]]], dtype=np.float64))
    return xy_segments, xz_segments, zy_segments


def module_envelope_outline_xy_xz_zy(layout: BarrelLayout) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    xy_segments: list[np.ndarray] = []
    xz_segments: list[np.ndarray] = []
    zy_segments: list[np.ndarray] = []
    inner_half = layout.inner_face_length_mm / 2.0
    outer_half = layout.outer_face_length_mm / 2.0
    half_thickness = layout.total_thickness_mm / 2.0
    half_z = layout.det_z_mm / 2.0

    cross_section = np.array(
        [
            [-inner_half, -half_thickness],
            [+inner_half, -half_thickness],
            [+outer_half, +half_thickness],
            [-outer_half, +half_thickness],
        ],
        dtype=np.float64,
    )

    cz = np.cos(layout.envelope_rotation_z_rad)
    sz = np.sin(layout.envelope_rotation_z_rad)
    envelope_rotation = np.array([[cz, -sz], [sz, cz]], dtype=np.float64)

    for raw_center_xy in layout.layers[0].module_centers_xy_mm:
        center_xy = envelope_rotation @ np.array(raw_center_xy, dtype=np.float64)
        radial = center_xy / np.linalg.norm(center_xy)
        tangent = np.array([-radial[1], radial[0]], dtype=np.float64)

        xy_corners: list[np.ndarray] = []
        front_face_xz: list[np.ndarray] = []
        back_face_xz: list[np.ndarray] = []
        front_face: list[np.ndarray] = []
        back_face: list[np.ndarray] = []
        for tangent_local, radial_local in cross_section:
            xy = center_xy + tangent_local * tangent + radial_local * radial
            xy_corners.append(xy)
            front_face_xz.append(np.array([xy[0], +half_z], dtype=np.float64))
            back_face_xz.append(np.array([xy[0], -half_z], dtype=np.float64))
            front_face.append(np.array([+half_z, xy[1]], dtype=np.float64))
            back_face.append(np.array([-half_z, xy[1]], dtype=np.float64))

        for i in range(4):
            p0_xy = xy_corners[i]
            p1_xy = xy_corners[(i + 1) % 4]
            xy_segments.append(np.array([[p0_xy[0], p0_xy[1]], [p1_xy[0], p1_xy[1]]], dtype=np.float64))

            p0_front = front_face[i]
            p1_front = front_face[(i + 1) % 4]
            p0_back = back_face[i]
            p1_back = back_face[(i + 1) % 4]
            p0_front_xz = front_face_xz[i]
            p1_front_xz = front_face_xz[(i + 1) % 4]
            p0_back_xz = back_face_xz[i]
            p1_back_xz = back_face_xz[(i + 1) % 4]
            xz_segments.append(np.array([[p0_front_xz[0], p0_front_xz[1]], [p1_front_xz[0], p1_front_xz[1]]], dtype=np.float64))
            xz_segments.append(np.array([[p0_back_xz[0], p0_back_xz[1]], [p1_back_xz[0], p1_back_xz[1]]], dtype=np.float64))
            xz_segments.append(np.array([[p0_back_xz[0], p0_back_xz[1]], [p0_front_xz[0], p0_front_xz[1]]], dtype=np.float64))
            zy_segments.append(np.array([[p0_front[0], p0_front[1]], [p1_front[0], p1_front[1]]], dtype=np.float64))
            zy_segments.append(np.array([[p0_back[0], p0_back[1]], [p1_back[0], p1_back[1]]], dtype=np.float64))
            zy_segments.append(np.array([[p0_back[0], p0_back[1]], [p0_front[0], p0_front[1]]], dtype=np.float64))
    return xy_segments, xz_segments, zy_segments


def barrel_layout_debug_report(layout: BarrelLayout, max_layers: int = 5) -> str:
    lines = [
        f"collection={layout.collection_name}",
        f"detector={layout.detector_name}",
        f"readout_xml={layout.readout_xml_path}",
        f"detector_xml={layout.detector_xml_path}",
        f"segmentation_type={layout.segmentation_type}",
        f"numsides={layout.numsides}",
        f"det_z_mm={layout.det_z_mm:.6f}",
        f"rmin_mm={layout.rmin_mm:.6f}",
        f"gap_mm={layout.gap_mm:.6f}",
        f"total_thickness_mm={layout.total_thickness_mm:.6f}",
        f"inner_angle_rad={layout.inner_angle_rad:.12f}",
        f"inner_angle_deg={np.degrees(layout.inner_angle_rad):.12f}",
        f"inner_face_length_mm={layout.inner_face_length_mm:.6f}",
        f"outer_face_length_mm={layout.outer_face_length_mm:.6f}",
        f"sect_center_radius_mm={layout.sect_center_radius_mm:.6f}",
        f"envelope_rotation_z_rad={layout.envelope_rotation_z_rad:.12f}",
        f"envelope_rotation_z_deg={np.degrees(layout.envelope_rotation_z_rad):.12f}",
        f"pitch_tangent_mm={layout.pitch_tangent_mm:.6f}",
        f"pitch_z_mm={layout.pitch_z_mm:.6f}",
        f"n_layers={len(layout.layers)}",
        "modules:",
    ]
    first_layer = layout.layers[0]
    for index, (angle, center) in enumerate(zip(first_layer.module_angles_rad, first_layer.module_centers_xy_mm), start=1):
        lines.append(
            f"  module={index:02d} angle_rad={angle:.12f} angle_deg={np.degrees(angle):.12f} "
            f"center_xy_mm=({center[0]:.6f}, {center[1]:.6f})"
        )
    lines.append("layers:")
    for layer in layout.layers[:max_layers]:
        lines.append(
            f"  layer={layer.layer_index:02d} sensitive_radius_mm={layer.sensitive_radius_mm:.6f} "
            f"layer_center_radius_mm={layer.layer_center_radius_mm:.6f} "
            f"half_thickness_mm={layer.half_thickness_mm:.6f} "
            f"sensitive_half_thickness_mm={layer.sensitive_half_thickness_mm:.6f} "
            f"half_tangent_mm={layer.half_tangent_mm:.6f} half_z_mm={layer.half_z_mm:.6f} "
            f"pitch_tangent_mm={layer.pitch_tangent_mm:.6f} pitch_z_mm={layer.pitch_z_mm:.6f}"
        )
    if len(layout.layers) > max_layers:
        lines.append("  ...")
        tail = layout.layers[-1]
        lines.append(
            f"  last_layer={tail.layer_index:02d} sensitive_radius_mm={tail.sensitive_radius_mm:.6f} "
            f"layer_center_radius_mm={tail.layer_center_radius_mm:.6f} "
            f"half_thickness_mm={tail.half_thickness_mm:.6f} "
            f"sensitive_half_thickness_mm={tail.sensitive_half_thickness_mm:.6f} "
            f"half_tangent_mm={tail.half_tangent_mm:.6f} half_z_mm={tail.half_z_mm:.6f}"
        )
    return "\n".join(lines)
