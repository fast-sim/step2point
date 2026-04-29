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

## HDBSCAN clustering

> hdbscan

HDBSCAN stands for **Hierarchical Density-Based Spatial Clustering of Applications with Noise**.
It is a density-based clustering method: instead of merging points by fixed detector bins, it groups deposits that form locally dense structures and treats sparse outliers as noise. In `step2point`, this is useful when you want compression that follows shower structure rather than only detector cell boundaries.

Requires:

- `cell_id` defined for each deposit
- a cell-id decoding rule that can be provided as either:
  - a cell-id encoding string
  - or a compact XML plus one or more readout collection names

Optional:

- `t` (time): when used (and present), used as a third clustering feature alongside x and y

This algorithm assumes a `cell_id` can be decoded to define the unmergeable points. It first groups deposits according to a configurable `merge_scope`, and then runs HDBSCAN independently inside each group. The grouping fields are extracted from the cell ID through an explicit cell-id decoding rule, either from a supplied encoding string or derived from the compact XML for one or more readout collections. Within each group the x/y/z coordinates are divided by a spatial scale (default 5 mm, roughly one cell width). If time is used (and available), it is expressed relative to the group median and divided by a temporal scale (default 1 ns), and HDBSCAN clusters on the four scaled coordinates `(x, y, z, t)`; otherwise it clusters on `(x, y, z)` only. Deposits that HDBSCAN labels as noise (label -1) are reassigned to their nearest cluster, ensuring energy conservation.

`merge_scope` defines which deposits are allowed to cluster together:

- `none`: no detector boundary is enforced; all deposits are clustered together
- `layer`: deposits can merge only within the same decoded layer
- `system_layer`: deposits can merge only within the same decoded system and layer
- `cell_id`: deposits can merge only within the same exact `cell_id`
- `cell_id_neighbour`: deposits can merge only within connected occupied-cell neighbourhoods built from decoded `cell_id` bins. Neighbours are cells with the same decoded system/layer and either `x` or `y` differing by `+-1` (or `x` and `z` when that is the available second cell-bin field). Wider groups can still form through chains of such neighbouring cells.

Parameters:

- `min_cluster_size`: HDBSCAN minimum cluster size (required)
- `min_samples`: HDBSCAN minimum samples for core points (required)
- `cluster_selection_epsilon`: HDBSCAN builds a hierarchy of clusters at different density levels and by default (epsilon=0) picks the most persistent ones, which can produce many small, high-density clusters. When epsilon > 0, clusters separated by a distance below this threshold are merged rather than split, producing fewer, larger clusters. A small value (e.g. 0.5 - 1.0 in scaled feature space) prevents over-fragmenting dense shower cores while still separating distinct deposits (default: 0)
- `xy_scale`: divide x, y, z coordinates by this before clustering. This normalises spatial distances so that 1.0 in scaled space corresponds to roughly one cell width. When `use_time` is True, this also ensures spatial and temporal coordinates are on comparable magnitudes. The value is detector-specific (default: 5.0 mm, matching the ODD calorimeter cell size)
- `t_scale`: divide time (relative to the layer median) by this before clustering. Normalises the temporal dimension so it contributes meaningfully alongside the scaled spatial coordinates. Only used when `t` is present and `use_time` is True (default: 1.0 ns)
- `use_time`: whether to include time as a clustering feature (default: False). When True, time must be present in the input shower or an error is raised
- `merge_scope`: detector boundary that HDBSCAN is not allowed to cross. Supported values are `none`, `layer`, `system_layer`, `cell_id`, and `cell_id_neighbour` (default: `system_layer`)
- `cell_id_encoding`: cell-id encoding string, or one string per input collection / system slot when clustering across multiple readout collections
- `algorithm`: internal neighbour-search method used by scikit-learn's HDBSCAN - `"auto"` (default), `"brute"`, `"kd_tree"`, or `"ball_tree"`. Use `"brute"` for the most reproducible reference outputs across machines. `"auto"` may choose different methods depending on the environment, which can slightly change cluster boundaries and therefore the compressed output
- `n_jobs`: number of parallel jobs for HDBSCAN and nearest-neighbour queries. `-1` uses all cores (default). `1` forces single-threaded execution, which improves reproducibility across runs

TODO:

- [ ] C++ backend. The core HDBSCAN clustering is handled by scikit-learn (Cython/C), so the Python path already gets compiled performance for the hot loop. A native C++ implementation would require either reimplementing HDBSCAN or linking a C++ library.
- [ ] GPU acceleration via scikit-learn's [Array API support](https://scikit-learn.org/stable/modules/array_api.html). This would allow HDBSCAN to run on GPU arrays (e.g. CuPy, PyTorch) for larger datasets. Needs investigation into whether the step2point backends API (currently Python and C++ only) should be extended to cover compute backends, or if this should be handled transparently within the Python algorithm.
