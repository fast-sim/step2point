from __future__ import annotations

from pathlib import Path

from step2point.geometry.dd4hep.factory_geometry import DD4hepResolver, build_barrel_layout_from_collection


def test_resolver_follows_includes():
    resolver = DD4hepResolver(Path("../OpenDataDetector/xml/OpenDataDetector.xml"))
    readout = resolver.find_readout("ECalBarrelCollection")
    detector = resolver.find_detector_for_readout("ECalBarrelCollection")
    assert readout.element.attrib["name"] == "ECalBarrelCollection"
    assert detector.element.attrib["name"] == "ECalBarrel"


def test_build_barrel_layout_from_collection():
    layout = build_barrel_layout_from_collection(Path("../OpenDataDetector/xml/OpenDataDetector.xml"), "ECalBarrelCollection")
    assert layout.detector_name == "ECalBarrel"
    assert layout.numsides == 16
    assert len(layout.layers) == 48
    assert layout.layers[0].sensitive_radius_mm > 1250.0
