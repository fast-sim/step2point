# step2point

Library for converting simulated calorimeter showers into compressed point-cloud representations that can be used as input to ML-based fast simulation.

## Why step2point was created

Detailed Geant4 calorimeter simulations produce either cell-level (merged) deposits which can be too coarse and too detector-specific, or granular step-level deposits that are too large for efficient ML training. step2point is designed to turn those detailed step-level deposits into a compact point-cloud shower representation while preserving the calorimeter observables that matter for later physics analysis and ML training.

The core task is:

**reduce the number of points without distorting the relevant shower physics.**

## Physics goal

Typical observables to preserve:
- total shower energy
- longitudinal profile
- radial profile
- phi profile
- first and second shower moments
- cell-energy spectrum
- number of active detector cells

Quantities expected to change by construction:
- point-energy spectrum
- number of points

## Core design

### Per-shower processing

The primitive operation is a single-shower transformation:

```text
read one shower -> compress one shower -> validate one shower
```

This keeps the library suitable for:
- offline dataset preparation
- streaming/event-wise workflows
- future on-the-fly clustering inside a C++ event-processing framework

### Minimal canonical data model

The core object is a flat `Shower` container of arrays:
- `x, y, z, E`
- optional `t`
- optional `cell_id`
- optional provenance arrays such as `pdg` or `track_id`
- `shower_id`

The internal representation avoids nested object graphs so it can be mapped cleanly to a future C++ implementation.

### Stateless algorithms

Compression algorithms take one `Shower` and return a new `Shower`.
They do not mutate the input and do not depend on file-format details.

### Strict separation of sub-directories

```text
core/        data model, interfaces, pipeline
algorithms/  compression kernels
metrics/     numerical computations
validation/  physics comparisons and reports
vis/         plotting only
io/          file-format adapters
cpp/         future C++ core and Python bindings
```

## Current features

- canonical `Shower` object
- HDF5 reader compatible with the step2point dataset layout
- direct `edm4hep.root` reader
- baseline algorithms:
    - `identity`
    - `merge_within_cell`
    - `merge_within_regular_subcell`
- metrics / validation / plotting
- small HDF5 regression sample and GitHub Actions CI
- MkDocs documentation site
- C++ core skeleton with an example of the `merge_within_cell` implementation

## Repository layout

```text
step2point/
├── pyproject.toml
├── README.md
├── mkdocs.yml
├── docs/
├── src/step2point/
│   ├── core/
│   ├── io/
│   ├── algorithms/
│   ├── metrics/
│   ├── validation/
│   └── vis/
├── cpp/
│   ├── include/step2point/
│   ├── src/
│   ├── tests/
│   └── bindings/
├── tests/
├── examples/
└── .github/workflows/
```

Contribution guidelines for common extension tasks are in [CONTRIBUTING.md](/home/anna/Workspace/step2point/CONTRIBUTING.md:1).

## Quick start

Install:

```bash
pip install -e .[dev]
```

### Produce Output HDF5

If you want to turn an input shower file into a new output HDF5 file with a chosen algorithm, use [examples/run_step2point_pipeline.py](/home/anna/Workspace/step2point/examples/run_step2point_pipeline.py:1).

Note:
`PYTHONPATH=src` is only needed when running directly from a source checkout without installing the package first. If you already ran `pip install -e .[dev]`, you can drop that prefix and use `python ...` directly.

Write an output file with no changes (`identity`):

```bash
PYTHONPATH=src python examples/run_step2point_pipeline.py \
  --input tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV.h5 \
  --algorithm identity \
  --output outputs/pipeline_identity
```

Write an output file with cell-wise merging:

```bash
PYTHONPATH=src python examples/run_step2point_pipeline.py \
  --input tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV.h5 \
  --algorithm merge_within_cell \
  --output outputs/pipeline_merge_within_cell
```

Write an output file with a regular `2x2` subgrid inside each detector cell:

```bash
PYTHONPATH=src python examples/run_step2point_pipeline.py \
  --input tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV.h5 \
  --algorithm merge_within_regular_subcell \
  --compact-xml ../OpenDataDetector/xml/OpenDataDetector.xml \
  --collection-name ECalBarrelCollection \
  --grid-x 2 \
  --grid-y 2 \
  --position-mode weighted \
  --output outputs/pipeline_regular_grid
```

Each run writes:

- `compressed_<algorithm>.h5`
- `compression_summary_<algorithm>.txt`

Minimal usage in application:

```python
from step2point.io.step2point_hdf5 import Step2PointHDF5Reader
from step2point.algorithms.merge_within_cell import MergeWithinCell

reader = Step2PointHDF5Reader("tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV.h5")
algorithm = MergeWithinCell()

for shower in reader.iter_showers():
    compressed = algorithm.compress(shower).shower
```
### How to use Key4hep + step2point

For the EDM4hep ROOT reader, create the virtual environment from the Key4hep Python, not from the system Python:

```bash
source /cvmfs/sw.hsf.org/key4hep/setup.sh
bash scripts/setup_key4hep_venv.sh
source /cvmfs/sw.hsf.org/key4hep/setup.sh
source .venv-key4hep/bin/activate
```

The helper script installs `step2point` in editable mode with the `dev` extras into `.venv-key4hep`.

Run the ROOT integration test:

```bash
pytest -q tests/integration/test_root_reader_optional.py
```

Minimal ROOT-reader smoke test:

```bash
python - <<'PY'
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
PY
```

Run tests:

```bash
pytest -q
```

## Inspection and visualization

An example basic inspection of shower observables can be done with [examples/inspect_showers.py](/home/anna/Workspace/step2point/examples/inspect_showers.py:1) which produces plots for manual validation.

Note:
`PYTHONPATH=src` is only needed when running directly from a source checkout without installing the package first. If you already ran `pip install -e .[dev]`, you can drop that prefix and use `python ...` directly.

Dataset-level only:

```bash
PYTHONPATH=src python examples/inspect_showers.py \
  --input tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV.h5 \
  --outdir outputs/inspect_gamma
```

Dataset plus single-shower plots:

```bash
PYTHONPATH=src python examples/inspect_showers.py \
  --input tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV.h5 \
  --shower-index 0 \
  --outdir outputs/inspect_gamma
```

Optional override of shower direction (so axis is not calculated from principal component analysis):

```bash
PYTHONPATH=src python examples/inspect_showers.py \
  --input tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV.h5 \
  --shower-index 0 \
  --axis 0 1 0 \
  --outdir outputs/inspect_gamma_axis
```

Library supports reading from HDF5 files (with structure as defined in [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17199427.svg)](https://doi.org/10.5281/zenodo.17199427)) as well as from the EDM4hep ROOT files:

```bash
python examples/inspect_showers.py \
  --input tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_edm4hep.root \
  --collections ECalBarrelCollection ECalEndcapCollection HCalBarrelCollection HCalEndcapCollection \
  --shower-index 5 \
  --axis 0 1 0 \
  --outdir outputs/inspect_root
```


Typical outputs are:
- `dataset_observables.png`
- `shower_<id>_projections.png`
- `shower_<id>_distributions.png`
- `shower_<id>_overview.png`

For detector-aware geometry inspection and shower overlays on module/layer/cell views, use [examples/plot_detector_cells.py](examples/plot_detector_cells.py:1). It writes `XY`, `XZ`, and `ZY` detector projections. The detailed workflow and example screenshots are documented in [docs/validation.md](docs/validation.md:1).

## [WIP] C++ backend

`merge_within_cell` is structured so it can later run from a shared C++ kernel. The Python algorithm already supports backend selection:

```python
MergeWithinCell(backend="python")
MergeWithinCell(backend="cpp")
MergeWithinCell(backend="auto")
```

Today, `auto` falls back to the Python implementation unless the optional `_step2point_cpp` extension has been built.

## Documentation site

A MkDocs Material site is included. Local preview:

```bash
pip install -e .[docs]
mkdocs serve
```

Build static docs:

```bash
mkdocs build
```

## CI

GitHub Actions includes:
- `ci.yml` for lint + tests on the tiny HDF5 sample
- `regression.yml` for generating validation plots as workflow artifacts
- `docs.yml` for documentation deployment to GitHub Pages

Every pull request runs install/import checks, unit and integration tests, a small physics sanity suite, docs build, and the C++/CMake test target.
