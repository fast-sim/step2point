# Test data

- `tiny_showers.h5` is a tiny regression sample used in CI for sanity checks (content is not the real showers).
  - based on step2point dataset (https://arxiv.org/pdf/2509.22340, https://gitlab.cern.ch/fastsim/step2point#dataset-file-structure)
- `ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV.h5` is the small ODD gamma HDF5 sample used for examples and inspection.
- `ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_merge_within_cell_reference.h5` is the committed reference output for `merge_within_cell` on the ODD gamma sample, used by CI to check writer output stability.
  - algorithm regressions follow the rule: `identity` is checked against the original input, while every non-identity algorithm should have its own committed reference file and regression case
- `ODD_pionM_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV.h5` is the corresponding ODD pion HDF5 sample.
- `ODD_gamma_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_edm4hep.root` is the matching ODD gamma EDM4hep ROOT sample.
- `ODD_pionM_10ev_theta90deg_phi0deg_posX0mmY1250mmZ0mm_10GeV_edm4hep.root` is the matching ODD pion EDM4hep ROOT sample.
ROOT files:
  - they contain `MCParticles` with MC truth particle data
  - and calorimeter hit collections are:
    `ECalBarrelCollection`, `ECalEndcapCollection`, `HCalBarrelCollection`,
    `HCalEndcapCollection`
