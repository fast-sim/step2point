# Test data

- `tiny_showers.h5` is the committed tiny regression sample used in CI.
  - based on step2point dataset (https://arxiv.org/pdf/2509.22340, https://gitlab.cern.ch/fastsim/step2point#dataset-file-structure)
- `tiny_showers.root` is not included here (yet)
  - ROOT sample properties:
  - 10 showers
  - contains `MCParticles`
  - contains the four calorimeter hit collections used in the current repo:
    `ECalBarrelCollection`, `ECalEndcapCollection`, `HCalBarrelCollection`,
    `HCalEndcapCollection`