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
- `t` (time) defined for each deposit (used as a clustering feature)
- `scikit-learn` installed (`pip install step2point[hdbscan]`)

This algorithm clusters deposits using HDBSCAN within each (subdetector, layer) partition. Features are scaled x, y coordinates and time relative to the layer median. Each cluster is merged into a single point: energy-weighted centroid position, summed energy, minimum time.

The partitioning by subdetector and layer, the feature scaling, and the noise-handling strategies are ported from the LUMEN project's HDBSCAN implementation.

Parameters:

- `min_cluster_size` -- HDBSCAN minimum cluster size (required)
- `min_samples` -- HDBSCAN minimum samples for core points (required)
- `cluster_selection_epsilon` -- values > 0 add DBSCAN-like behaviour (default: 0)
- `xy_scale` -- divide x, y coordinates by this before clustering (default: 5.0 mm)
- `t_scale` -- divide time by this before clustering (default: 1.0 ns)
- `layer_extractor` -- callable to extract layer from cell_id (default: ODD bit layout)

Noise handling strategies (`noise_handle`):

- `nn` -- reassign noise to the nearest cluster (default, energy-conserving)
- `singleton` -- each noise point becomes its own cluster (energy-conserving)
- `layer` -- all noise in a layer is bundled into one cluster (energy-conserving)
- `drop` -- discard noise points (loses energy)

TODO:

- [ ] C++ backend. The core HDBSCAN clustering is handled by scikit-learn (Cython/C), so the Python path already gets compiled performance for the hot loop. A native C++ implementation would require either reimplementing HDBSCAN or linking a C++ library.
- [ ] GPU acceleration via scikit-learn's [Array API support](https://scikit-learn.org/stable/modules/array_api.html). This would allow HDBSCAN to run on GPU arrays (e.g. CuPy, PyTorch) for larger datasets. Needs investigation into whether the step2point backends API (currently Python and C++ only) should be extended to cover compute backends, or if this should be handled transparently within the Python algorithm.
