from __future__ import annotations

from pathlib import Path

import pytest

from step2point.geometry.dd4hep.bitfield import decode_dd4hep_cell_id, parse_dd4hep_id_encoding
from step2point.geometry.dd4hep.factory_geometry import (
    DD4hepResolver,
    build_barrel_layout_from_collection,
)

ODD_XML = Path("../OpenDataDetector/xml/OpenDataDetector.xml")


@pytest.mark.skipif(not ODD_XML.exists(), reason="OpenDataDetector XML checkout is not available")
def test_resolver_follows_includes():
    resolver = DD4hepResolver(ODD_XML)
    readout = resolver.find_readout("ECalBarrelCollection")
    detector = resolver.find_detector_for_readout("ECalBarrelCollection")
    assert readout.element.attrib["name"] == "ECalBarrelCollection"
    assert detector.element.attrib["name"] == "ECalBarrel"


@pytest.mark.skipif(not ODD_XML.exists(), reason="OpenDataDetector XML checkout is not available")
def test_build_barrel_layout_from_collection():
    layout = build_barrel_layout_from_collection(ODD_XML, "ECalBarrelCollection")
    assert layout.detector_name == "ECalBarrel"
    assert layout.numsides == 16
    assert len(layout.layers) == 48
    assert layout.layers[0].sensitive_radius_mm > 1250.0


def test_parse_dd4hep_id_encoding_signed_fields():
    fields = parse_dd4hep_id_encoding("system:8,barrel:3,module:4,layer:6,slice:5,x:32:-16,y:-16")
    assert [field.name for field in fields] == ["system", "barrel", "module", "layer", "slice", "x", "y"]
    assert fields[-2].offset == 32
    assert fields[-2].width == 16
    assert fields[-2].signed is True
    assert fields[-1].offset == 48
    assert fields[-1].width == 16
    assert fields[-1].signed is True


def test_decode_dd4hep_cell_id_signed_values():
    encoding = "system:8,barrel:3,module:4,layer:6,slice:5,x:32:-16,y:-16"
    cell_id = (
        10
        | (1 << 8)
        | (9 << 11)
        | (12 << 15)
        | (3 << 21)
        | ((5 & 0xFFFF) << 32)
        | ((-7 & 0xFFFF) << 48)
    )
    decoded = decode_dd4hep_cell_id(cell_id, encoding)
    assert decoded["system"] == 10
    assert decoded["barrel"] == 1
    assert decoded["module"] == 9
    assert decoded["layer"] == 12
    assert decoded["slice"] == 3
    assert decoded["x"] == 5
    assert decoded["y"] == -7
