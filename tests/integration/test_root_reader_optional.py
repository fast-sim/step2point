from __future__ import annotations

from pathlib import Path

import pytest

from step2point.core.edm4hep_root import EDM4hepRootReader


DATA = Path(__file__).resolve().parents[1] / "data" / "tiny_showers.root"


@pytest.mark.root
@pytest.mark.skipif(not DATA.exists(), reason="tiny EDM4hep ROOT sample not committed yet")
def test_edm4hep_root_reader_tiny_optional():
    reader = EDM4hepRootReader(str(DATA), shower_limit=1)
    shower = next(reader.iter_showers())
    assert shower.n_points > 0
    assert shower.cell_id is not None
