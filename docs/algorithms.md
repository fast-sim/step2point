
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