# Validation

Compression is only useful if we can say clearly what it preserves and what it changes.

For that reason, `step2point` separates validation into two categories.

## Quantities that should stay unchanged

These are the observables that define whether the compressed shower still behaves like the original shower.

### Total shower energy

A natural diagnostic is the per-shower ratio:

```text
E_post / E_pre
```

Using a ratio is usually more informative than looking at raw energies directly, because it factors out the broad physical energy range in the dataset.

### Shower profiles

The compressed shower should preserve the broad shape of the shower:

- longitudinal profile
- radial profile
- phi profile

### Shower moments

Useful compact summaries are:

- first longitudinal moment
- second longitudinal moment
- first radial moment
- second radial moment

### Detector-aware quantities

If `cell_id` is present, detector-aware checks become especially important:

- distribution of `log(cell_energy)`
- ratio of the number of cells before and after compression

## Quantities that are expected to change

Some changes are not only acceptable but are the whole point of compression.

### Point-energy spectrum

The distribution of individual point energies will change because points are being merged.

### Number of points

The ratio

```text
N_points_post / N_points_pre
```

is one of the central performance indicators of a compression algorithm.

## Example inspection workflow

For shower inspection, use `examples/inspect_showers.py`. The script always produces a dataset-level observables, and it also produces single-shower plots if `--shower-index` is given.

Dataset-level only:

```bash
PYTHONPATH=src python examples/inspect_showers.py \
  --input tests/data/CLD_gamma_10GeV_posY2150mm_dirY1_10ev_sim_detailed_tchandler.h5 \
  --input-format hdf5 \
  --axis 0 1 0 \
  --outdir outputs/inspect_gamma
```

Dataset plus single-shower plots:

```bash
PYTHONPATH=src python examples/inspect_showers.py \
  --input tests/data/CLD_gamma_10GeV_posY2150mm_dirY1_10ev_sim_detailed_tchandler.h5 \
  --input-format hdf5 \
  --shower-index 0 \
  --axis 0 1 0 \
  --outdir outputs/inspect_gamma
```

Recommended axis override for the CLD front-face samples:

```bash
PYTHONPATH=src python examples/inspect_showers.py \
  --input tests/data/CLD_gamma_10GeV_posY2150mm_dirY1_10ev_sim_detailed_tchandler.h5 \
  --input-format hdf5 \
  --shower-index 0 \
  --axis 0 1 0 \
  --outdir outputs/inspect_gamma
```

Expected outputs:

- `dataset_observables.png`
- `shower_<id>_projections.png`
- `shower_<id>_distributions.png`
- `shower_<id>_overview.png`

# Validation results

## EM showers

Animation below shows an example of a single electromagnetic shower:

![gamma](assets/images/animation_gamma.gif){ .animation-gamma }

Single-shower inspection outputs produced with `--shower-index 0` on the gamma sample:

### Projections

![gamma projections](assets/images/inspect_gamma/shower_0_projections.png)

### Distributions

![gamma distributions](assets/images/inspect_gamma/shower_0_distributions.png)

### Overview

![gamma overview](assets/images/inspect_gamma/shower_0_overview.png)

### Dataset observables matrix

![gamma matrix](assets/images/inspect_gamma/dataset_observables.png)

## hadronic showers

Animation below shows an example of a single hadronic shower:

![pion](assets/images/animation_pion.gif){ .animation-pion }

Single-shower inspection outputs produced with `--shower-index 0 --axis 0 1 0` on the pion sample:

### Projections

![pion projections](assets/images/inspect_pion/shower_0_projections.png)

### Distributions

![pion distributions](assets/images/inspect_pion/shower_0_distributions.png)

### Overview

![pion overview](assets/images/inspect_pion/shower_0_overview.png)

### Dataset observables

![pion matrix](assets/images/inspect_pion/dataset_observables.png)
