from __future__ import annotations

from pathlib import Path

import pytest

from step2point.io import EDM4hepRootReader

DATA = Path(__file__).resolve().parents[1] / "data" / "ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_edm4hep.root"


def _has_podio_root_io() -> bool:
    try:
        from podio import root_io  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.root
@pytest.mark.skipif(not DATA.exists(), reason="ODD EDM4hep ROOT sample is not available")
@pytest.mark.skipif(not _has_podio_root_io(), reason="podio.root_io is not available in this environment")
def test_edm4hep_root_reader_optional():
    reader = EDM4hepRootReader(
        str(DATA),
        collections=("ECalBarrelCollection", "ECalEndcapCollection", "HCalBarrelCollection", "HCalEndcapCollection"),
        shower_limit=1,
    )
    shower = next(reader.iter_showers())
    assert shower.n_points > 0
    assert shower.cell_id is not None
    assert shower.t is not None
