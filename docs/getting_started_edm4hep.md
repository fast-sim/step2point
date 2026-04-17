# EDM4hep ROOT Input

This page covers the additional setup needed to read EDM4hep ROOT files directly.

For the default HDF5 workflow, see [Getting started](getting_started.md).

## Key4hep environment

The ROOT reader depends on `podio` and `edm4hep`, so it must run inside a Key4hep environment.

Create a local virtual environment from the Key4hep Python:

```bash
source /cvmfs/sw.hsf.org/key4hep/setup.sh
bash scripts/setup_key4hep_venv.sh
source /cvmfs/sw.hsf.org/key4hep/setup.sh
source .venv-key4hep/bin/activate
```

This matters because a regular virtual environment created before sourcing Key4hep will not see the right ROOT-stack Python packages.

The helper script installs `step2point` in editable mode with the `dev` extras into `.venv-key4hep`.

Run the ROOT integration test:

```bash
pytest -q tests/integration/test_root_reader_optional.py
```

## Minimal ROOT-reader example

```python
from step2point.io import EDM4hepRootReader

reader = EDM4hepRootReader(
    "tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_edm4hep.root",
    collections=(
        "ECalBarrelCollection",
        "ECalEndcapCollection",
        "HCalBarrelCollection",
        "HCalEndcapCollection",
    ),
    shower_limit=1,
)

shower = next(reader.iter_showers())
print(shower.n_points, shower.cell_id is not None, shower.t is not None)
```

## ROOT inspection example

```bash
python examples/inspect_showers.py \
  --input tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_edm4hep.root \
  --collections ECalBarrelCollection ECalEndcapCollection HCalBarrelCollection HCalEndcapCollection \
  --shower-index 5 \
  --axis 0 1 0 \
  --outdir outputs/inspect_root
```
