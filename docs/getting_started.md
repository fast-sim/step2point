# Getting started

## Installation

Install the Python package in editable mode:

```bash
pip install -e .[dev]
```

## First example

The simplest pipeline is deliberately organized around one shower at a time.

```python
from step2point.io.step2point_hdf5 import Step2PointHDF5Reader
from step2point.algorithms.merge_within_cell import MergeWithinCell

reader = Step2PointHDF5Reader("tests/data/ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV.h5")
algorithm = MergeWithinCell()

for shower in reader.iter_showers():
    compressed = algorithm.compress(shower)
```

## File reader

Readers convert external formats into the canonical `Shower` representation used everywhere else in the library.

That means the algorithms themselves do not need to know whether the original source was:

- EDM4hep ROOT
- step2point-style HDF5
- or any future custom source

## Algorithms

Algorithms are intended to be simple:

```python
compressed = algorithm.compress(shower)
```

- the input `Shower` should not be modified in place
- the output should be a new `Shower`
- configuration lives in the algorithm object, not in hidden global state

## Running tests

```bash
pytest -q
```

The test suite is meant to catch both software regressions and early physics regressions on a tiny frozen sample.

## Running the documentation site

To work on the documentation site locally:

```bash
pip install -e .[docs]
mkdocs serve
```

Then open the local URL shown by MkDocs.
