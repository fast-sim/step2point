# Architecture

The architecture is designed to remain simple and to support the evolution of the project, which ismeant to stay usable both as a Python library now and as a future shared Python/C++ toolkit later.

## Directory roles

```text
core/        data model, pipeline, interfaces
algorithms/  compression methods
metrics/     numerical building blocks
validation/  physics-facing comparisons and summaries
vis/         plotting and presentation
io/          readers and writers
```

## The canonical Shower object

The central object is a single `Shower` with flat arrays such as:

- `x`
- `y`
- `z`
- `E`
- optional `t`
- optional `cell_id`
- `shower_id`

This should remain language-neutral and easy to map to a C++ struct or class later.

The Shower object can be extended with other (optional) characteristics needed by other algorithms.

## Metrics and validation

### `metrics/`
Contains reusable numerical computations, for example:

- total energy
- centroids
- shower moments
- profile coordinates
- binned distributions

### `validation/`
Uses those computations to answer physics questions, for example:

- was energy preserved?
- did the longitudinal profile move?
- how much did the number of cells change?

This makes it easy to move low-level numerics into C++ later without pulling plotting and reporting logic with them.

## Algorithms

Algorithms should behave like pure per-shower transformations:

```python
compressed = algorithm.compress(shower)
```

That means:

- the input shower is not mutated
- the output is a fresh shower
- the algorithm object only stores configuration

## C++-friendliness

The future C++ path is straightforward as Python library already now is designed to behave as if it was calling an external kernel:

- a `Shower` is a flat data container
- an algorithm consumes one shower and returns one shower
- metrics are numerical and modular
- plotting stays outside the core
