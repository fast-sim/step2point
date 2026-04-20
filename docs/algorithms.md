
# Algorithms

step2point library implements different algorithms for optimal data representation, which may differ depending on the target detector.
It also allows new algorithms to be developed and tested.
Contributions are welcome!

## Identity algorithm

> identity

A fake compression that maintains a shower. Meant as an example and base for sanity checks.

## Merging with a cell

> merge_within_cell

Requires:
- `cell_id` defined for each deposit with a `Shower`

This algorithm compresses all deposits that fall within the same detector cell (defined by a `cell_id`). This behaviour reproduces a typical simulation of calorimeters that aggregates deposits within a cell, avoiding enormous output files.

TODO:
- [ ] Implement a configurable time window that merges deposits only within this window to reproduce a first basic digitisation step.

## Merging within a regular grid inside a cell

> merge_within_regular_subcell

Requires:
- `cell_id` defined for each deposit with a `Shower`
- a DD4hep compact XML and readout collection name, or a prebuilt barrel layout

This algorithm first subdivides each detector cell into a regular `x/N` by `y/M` grid in the local segmentation plane and then merges deposits within each subcell. The output position can be either:

- `weighted` (default): energy-weighted barycenter of the deposits in the subcell
- `center`: geometric center of the subcell on the sensitive surface
