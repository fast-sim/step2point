# Test data

- `tiny_showers.h5` is a tiny regression sample used in CI for sanity checks (content is not the real showers).
  - based on step2point dataset (https://arxiv.org/pdf/2509.22340, https://gitlab.cern.ch/fastsim/step2point#dataset-file-structure)
- `ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV.h5` is the small ODD gamma HDF5 sample used for examples and inspection.
- `ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_merge_within_cell_reference.h5` is the committed reference output for `merge_within_cell` on the ODD gamma sample, used by CI to check writer output stability.
- `ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_merge_within_regular_subcell_3x3_weighted_reference.h5` is the committed reference output for `merge_within_regular_subcell` with `--grid-x 3 --grid-y 3 --position-mode weighted` on the ODD gamma sample.
- `ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_merge_within_regular_subcell_3x3_center_reference.h5` is the committed reference output for `merge_within_regular_subcell` with `--grid-x 3 --grid-y 3 --position-mode center` on the ODD gamma sample.
  - algorithm regressions follow the rule: `identity` is checked against the original input, while every non-identity algorithm should have its own committed reference file and regression case
- `ODD_gamma_10ev_theta90deg_phi90deg_posX0mmY1250mmZ0mm_100GeV_merge_within_regular_subcell_5x5_weighted_debug.h5` is a debug example that keeps the original points and adds `steps/cluster_label` for a `merge_within_regular_subcell` `5x5` weighted clustering run.
- `ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_hdbscan_reference.h5` is the committed reference output for `hdbscan` on the ODD gamma sample with:
  - `min_cluster_size=5`
  - `min_samples=3`
  - `--use-time`
  - cell-id decoding rule:
    `system:8,barrel:3,module:4,stave:1,layer:6,slice:5,x:32:-16,y:-16`
  - the blocking loose regression compares the with-time result's shower IDs, compression, energy-weighted centroids,
    moments, and longitudinal/radial/time profiles with narrow tolerances instead of comparing point rows
    - point counts: 0.5% total and 2% per shower
    - total energy per shower: `1e-7` absolute tolerance
    - centroids: 0.01 mm spatial and 0.001 ns time absolute tolerance
    - first/second moments: 0.5% relative tolerance
    - normalized 8-bin profile L1 distance: at most 0.02
  - the without-time loose case uses broader invariant/range checks because it has no committed reference
  - the point-by-point strict reference check runs as a non-blocking CI diagnostic because HDBSCAN cluster boundaries can
    vary slightly across platforms; its output artifacts remain available for investigation
- `ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_cluster_within_cell_reference.h5` is the committed reference output for `cluster_within_cell` on the ODD gamma sample with:
  - `AgglomerativeClustering`
    - `distance_threshold=1.`
- `ODD_pionM_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV.h5` is the corresponding ODD pion HDF5 sample.
- `ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_edm4hep.root` is the matching ODD gamma EDM4hep ROOT sample.
- `ODD_pionM_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_edm4hep.root` is the matching ODD pion EDM4hep ROOT sample.
ROOT files:
  - they contain `MCParticles` with MC truth particle data
  - and calorimeter hit collections are:
    `ECalBarrelCollection`, `ECalEndcapCollection`, `HCalBarrelCollection`,
    `HCalEndcapCollection`
