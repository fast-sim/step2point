#!/bin/bash
# shellcheck source=/dev/null
source /cvmfs/sw.hsf.org/key4hep/setup.sh

cd /eos/user/m/mamozzan/step2point/ || exit

# shellcheck source=/dev/null
source .venv-key4hep/bin/activate

# geometry file in $k4geo_DIR/ILD/compact/ILD_l5_o1_v02/ILD_l5_o1_v02.xml
# for loop on seed
export algo=merge_within_regular_subcell #[identity, merge_within_cell, merge_within_regular_subcell, hdbscan]

extra_args="merge_within_regular_subcell"

# shellcheck source=/dev/null
if [[ "$algo" == "merge_within_regular_subcell" ]]; then
    extra_args="--compact-xml k4geo_DIR/ILD/compact/ILD_l5_o1_v02/ILD_l5_o1_v02.xml \
                --collection-name ECalBarrelCollection \
                --grid-x 5 \
                --grid-y 5 \
                --position-mode weighted \
                "
fi

for seed in {0..0}; do
    python examples/run_step2point_pipeline.py \
      --input /eos/user/m/mamozzan/photons_root/p22_th45-135_ph79-109_en5-130_seed"${seed}"_ip.edm4hep.root \
      --algorithm "$algo" \
      --output outputs/pipeline_ECalBarrelCollection_"${algo}"_"${seed}" \
      --collections ECalBarrelSiHitsEven ECalBarrelSiHitsOdd ECalEndcapSiHitsEven ECalEndcapSiHitsOdd HcalBarrelRegCollection HcalEndcapRingCollection \
      "$extra_args"
done

# testing
export seed=0
export algo=identity
python examples/generate_validation_plots.py \
  --input /eos/user/m/mamozzan/photons_root/p22_th45-135_ph79-109_en5-130_seed"${seed}"_ip.edm4hep.root \
  --algorithm "${algo}" \
  --outdir outputs/plots_"${algo}"_"${seed}" \
  --collections ECalBarrelSiHitsEven ECalBarrelSiHitsOdd ECalEndcapSiHitsEven ECalEndcapSiHitsOdd HcalBarrelRegCollection HcalEndcapRingCollection

# convert to cc format
for seed in {1..10}; do
    python martina_test/convert_to_cc3_format.py outputs/pipeline_identity/pipeline_identity_"${seed}"/compressed_identity.h5
done
