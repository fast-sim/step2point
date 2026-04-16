# Test data

- `tiny_showers.h5` is a tiny regression sample used in CI.
  - based on step2point dataset (https://arxiv.org/pdf/2509.22340, https://gitlab.cern.ch/fastsim/step2point#dataset-file-structure)
- `CLD_gamma_10GeV_posY2150mm_dirY1_10ev_sim_detailed_tchandler.h5` is a realistic but small (10 events) gamma sample intended for examples and inspection.
- `CLD_pionM_10GeV_posY2150mm_dirY1_10ev_sim_detailed_tchandler.h5` is the corresponding pion sample.
- `tiny_showers.root` is not included here (yet)
  - ROOT sample properties:
  - 10 showers
  - contains `MCParticles`
  - contains the four calorimeter hit collections used in the current repo:
    `ECalBarrelCollection`, `ECalEndcapCollection`, `HCalBarrelCollection`,
    `HCalEndcapCollection`
