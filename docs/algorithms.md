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

> hdbscan_clustering

Requires:

- `cell_id` defined for each deposit (for layer extraction)

Optional:

- `t` (time): when present, used as a third clustering feature alongside x and y

This algorithm clusters deposits using HDBSCAN within each (subdetector, layer) partition. Deposits are first partitioned by (subdetector, layer) where the layer is extracted from the cell ID using a bit-extraction function (defaulting to the ODD layout: 9 bits at bit 19). Within each partition the x/y coordinates are divided by a spatial scale (default 5 mm, roughly one cell width). If time is available, it is expressed relative to the layer median and divided by a temporal scale (default 1 ns), and HDBSCAN clusters on all three scaled features (x, y, t); otherwise it clusters on (x, y) only. Deposits that HDBSCAN labels as noise (label -1) are reassigned to their nearest cluster, ensuring energy conservation.

Parameters:

- `min_cluster_size`: HDBSCAN minimum cluster size (required)
- `min_samples`: HDBSCAN minimum samples for core points (required)
- `cluster_selection_epsilon`: HDBSCAN builds a hierarchy of clusters at different density levels and by default (epsilon=0) picks the most persistent ones, which can produce many small, high-density clusters. When epsilon > 0, clusters separated by a distance below this threshold are merged rather than split, producing fewer, larger clusters. A small value (e.g. 0.5 - 1.0 in scaled feature space) prevents over-fragmenting dense shower cores while still separating distinct deposits (default: 0)
- `xy_scale`: divide x, y coordinates by this before clustering. This normalises spatial distances so that 1.0 in scaled space corresponds to roughly one cell width. When `use_time` is True, this also ensures spatial and temporal features are on comparable magnitudes. The value is detector-specific (default: 5.0 mm, matching the ODD calorimeter cell size)
- `t_scale`: divide time (relative to the layer median) by this before clustering. Normalises the temporal dimension so it contributes meaningfully alongside the scaled spatial features. Only used when `t` is present and `use_time` is True (default: 1.0 ns)
- `use_time`: whether to include time as a clustering feature (default: False). When True, time must be present in the input shower or an error is raised
- `layer_extractor`: how to extract layer IDs from cell IDs. Can be a callable `f(cell_ids) -> layers`, a DD4hep ID encoding string (e.g. `"system:8,barrel:3,layer:19:9"`), or `None` to use the ODD default `(cell_id >> 19) & 0x1FF`
- `algorithm`: internal neighbour-search method used by scikit-learn's HDBSCAN - `"auto"` (default), `"brute"`, `"kd_tree"`, or `"ball_tree"`. Use `"brute"` for the most reproducible reference outputs across machines. `"auto"` may choose different methods depending on the environment, which can slightly change cluster boundaries and therefore the compressed output
- `n_jobs`: number of parallel jobs for HDBSCAN and nearest-neighbour queries. `-1` uses all cores (default). `1` forces single-threaded execution, which improves reproducibility across runs

TODO:

- [ ] C++ backend. The core HDBSCAN clustering is handled by scikit-learn (Cython/C), so the Python path already gets compiled performance for the hot loop. A native C++ implementation would require either reimplementing HDBSCAN or linking a C++ library.
- [ ] GPU acceleration via scikit-learn's [Array API support](https://scikit-learn.org/stable/modules/array_api.html). This would allow HDBSCAN to run on GPU arrays (e.g. CuPy, PyTorch) for larger datasets. Needs investigation into whether the step2point backends API (currently Python and C++ only) should be extended to cover compute backends, or if this should be handled transparently within the Python algorithm.
