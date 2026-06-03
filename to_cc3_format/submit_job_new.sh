#!/bin/bash
# shellcheck source=/dev/null
source /cvmfs/sw.hsf.org/key4hep/setup.sh

cd /eos/user/m/mamozzan/k4EDM4hep2LcioConv || exit

for seed in {20..25}; do
    /eos/user/m/mamozzan/k4EDM4hep2LcioConv/install/bin/lcio2edm4hep /eos/user/m/mamozzan/bechmark_photons/sim-E1261AT600AP180-180_file_${seed}.slcio /eos/project/f/fast/edm4hep_frombenchmark/sim-E1261AT600AP180-180_file_${seed}.edm4hep.root
done

cd /eos/user/m/mamozzan/step2point/ || exit

# shellcheck source=/dev/null
source .venv-key4hep/bin/activate

# geometry file in $k4geo_DIR/ILD/compact/ILD_l5_o1_v02/ILD_l5_o1_v02.xml
# for loop on seed
export algo=merge_within_regular_subcell #[identity, merge_within_cell, merge_within_regular_subcell, hdbscan]
mkdir -p outputs/pipeline2_"${algo}"

# ----------------------------------------------------
# for subcell
# shellcheck source=/dev/null
if [[ "$algo" == "merge_within_regular_subcell" ]]; then
    extra_args=(
        --compact-xml "$k4geo_DIR/ILD/compact/ILD_l5_o1_v02/ILD_l5_o1_v02.xml"
        --collection-name ECalBarrelCollection
        --grid-x 5
        --grid-y 5
        --position-mode weighted
    )
else
    extra_args=()
fi

for seed in {0..0}; do
    python examples/run_step2point_pipeline.py \
      --input /eos/project/f/fast/edm4hep_frombenchmark/sim-E1261AT600AP180-180_file_${seed}.edm4hep.root \
      --algorithm "$algo" \
      --output outputs/pipeline2_"${algo}"/file_"${seed}" \
      --collections EcalBarrelCollection \
      "${extra_args[@]}"
done
# ----------------------------------------------------
# for identity or within cell
for seed in {2..3}; do
    python examples/run_step2point_pipeline.py \
      --input /eos/project/f/fast/edm4hep_frombenchmark/sim-E1261AT600AP180-180_file_${seed}.edm4hep.root \
      --algorithm "$algo" \
      --output outputs/pipeline2_"${algo}"/file_"${seed}" \
      --collections EcalBarrelCollection
done

python examples/run_step2point_pipeline.py \
      --input /eos/project/f/fast/edm4hep_frombenchmark/test_small.edm4hep.root \
      --algorithm "$algo" \
      --output outputs/pipeline2_"${algo}"/test_small \
      --collections EcalBarrelCollection

# for hdbscan
algo="hdbscan"
for seed in {0..0}; do
    python examples/run_step2point_pipeline.py \
      --input /eos/project/f/fast/edm4hep_frombenchmark/sim-E1261AT600AP180-180_file_${seed}.edm4hep.root \
      --algorithm ${algo} \
      --collections EcalBarrelCollection \
      --hdbscan-cell-id-encoding system:5,module:3,stave:4,tower:4,layer:6,wafer:6,slice:4,cellX:32:-16,cellY:-16 \
      --min-cluster-size 40 \
      --min-samples 8 \
      --epsilon 0.5 \
      --merge-scope cell_id \
      --use-time \
      --output outputs/pipeline2_"${algo}"/file_"${seed}"
done

# validation plots
export seed=0
export algo=hdbscan
python examples/generate_validation_plots.py \
  --input /eos/user/m/mamozzan/edm4hep_frombenchmark/sim-E1261AT600AP180-180_file_${seed}.edm4hep.root \
  --algorithm "${algo}" \
  --outdir outputs/pipeline2_"${algo}"/file_"${seed}"/plots_"${algo}"_"${seed}" \
  --collections EcalBarrelCollection

# shower display
export index=10
python examples/render_shower_display.py \
  --input outputs/pipeline2_"${algo}"/file_"${seed}"/compressed_"${algo}".h5 \
  --shower-index ${index} \
  --crop-percentile 80 \
  --out outputs/pipeline2_"${algo}"/file_"${seed}"/shower${index}.png

# shower display from root
python examples/render_shower_display.py \
  --input ../edm4hep_frombenchmark/sim-E1261AT600AP180-180_file_${seed}.edm4hep.root \
  --collections EcalBarrelCollection \
  --shower-index ${index} \
  --crop-percentile 80 \
  --out outputs/shower_fromroot${index}.png

# ----------------------------------------------------
# convert to cc3 format
for seed in {0..3}; do
    python martina_test/convert_to_cc3_format.py outputs/pipeline2_"${algo}"/file_"${seed}"/compressed_"${algo}".h5 outputs/pipeline2_"${algo}"/file_"${seed}"/compressed_"${algo}".input_global_cc3.h5
done
# plots as check
# python martina_test/plot_check_cc3_format.py outputs/pipeline_"${algo}"/file_0/compressed_"${algo}".input_cc3.h5
python martina_test/plot_check_cc3_format.py outputs/cc3input_"${algo}"/input_cc3_file_"${seed}".h5
