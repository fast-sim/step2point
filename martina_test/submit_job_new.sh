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
mkdir -p outputs/pipeline2_"${algo}"

export algo=hdbscan #[identity, merge_within_cell, merge_within_regular_subcell, hdbscan]
ms=8
mcs=40

# ----------------------------------------------------
# for subcell
# shellcheck source=/dev/null
name=""
if [[ "$algo" == "merge_within_regular_subcell" ]]; then
    extra_args=(
        --compact-xml "/eos/user/p/pmckeown/Step2Point_Clustering/ILD_Regular_Detector/ILD/compact/ILD_l5_o1_v02/ILD_l5_o1_v02.xml"
        --collection-name EcalBarrelCollection
        --grid-x 5
        --grid-y 5
        --position-mode weighted
    )
elif [[ "$algo" == "hdbscan" ]]; then
    name=_"ms${ms}"_"mcs${mcs}"
    extra_args=(
        --hdbscan-cell-id-encoding system:5,module:3,stave:4,tower:4,layer:6,wafer:6,slice:4,cellX:32:-16,cellY:-16 \
        --min-cluster-size $mcs \
        --min-samples $ms \
        --epsilon 0.0 \
        --merge-scope cell_id \
        --use-time \
    )
else
    extra_args=()
fi

for seed in {16..27}; do
    python examples/run_step2point_pipeline.py \
      --input /eos/project/f/fast/edm4hep_frombenchmark/sim-E1261AT600AP180-180_file_${seed}.edm4hep.root \
      --algorithm "$algo" \
      --output /eos/project/f/fast/step2point_files/pipeline2_"${algo}""${name}"_9x/file_"${seed}" \
      --collections EcalBarrelCollection \
      "${extra_args[@]}"
done

# file_"${seed}
# validation plots
export seed=0
export algo=hdbscan
python examples/generate_validation_plots.py \
  --input /eos/project/f/fast/edm4hep_frombenchmark/test_small.edm4hep.root \
  --algorithm "${algo}" \
  --outdir /eos/project/f/fast/step2point_files/pipeline2_"${algo}""${name}"/test_small/plots_"${algo}" \
  --collections EcalBarrelCollection \
  --min-cluster-size 40 \
  --min-samples 8 \
  --merge-scope cell_id \
  --hdbscan-cell-id-encoding system:5,module:3,stave:4,tower:4,layer:6,wafer:6,slice:4,cellX:32:-16,cellY:-16 \

# shower display
export index=10
python examples/render_shower_display.py \
  --input /eos/project/f/fast/step2point_files/pipeline2_"${algo}""${name}"/file_"${seed}"/compressed_"${algo}".h5 \
  --shower-index ${index} \
  --crop-percentile 80 \
  --out outputs/pipeline2_"${algo}""${name}"/file_"${seed}"/shower${index}.png

# ----------------------------------------------------
# convert to cc3 format
for seed in {1..15}; do
    python martina_test/convert_to_cc3_format.py /eos/project/f/fast/step2point_files/pipeline2_"${algo}""${name}"/file_"${seed}"/compressed_"${algo}".h5 /eos/project/f/fast/step2point_files/pipeline2_"${algo}""${name}"/file_"${seed}"/compressed_"${algo}".input_global_cc3.h5 --pc_save_folder /eos/user/m/mamozzan/step2point/outputs/cc3input_merge_within_regular_subcell_6kcut --6kcut
done
export seed=0

# test_small isn't seed-indexed (file_N), so convert_to_cc3_format.py's
# default output-path logic can't parse a seed out of the parent dir name -
# use --pc_save_folder to write into the same per-algorithm folder as the
# seeded runs; the output filename is derived from the input's parent dir
# name ("test_small" -> input_cc3_test_small.h5).
python martina_test/convert_to_cc3_format.py /eos/project/f/fast/step2point_files/pipeline2_"${algo}""${name}"/test_small/compressed_"${algo}".h5 --pc_save_folder outputs/cc3input_"${algo}""${name}"
# plots as check
python martina_test/plot_check_cc3_format.py outputs/cc3input_"${algo}""${name}"/input_cc3_test_small.h5
python martina_test/plot_check_cc3_format.py outputs/cc3input_"${algo}""${name}"/input_cc3_file_"${seed}".h5

python martina_test/plot_check_cc3_format.py /eos/user/m/mamozzan/step2point/outputs/cc3input_merge_within_regular_subcell_6kcut/input_cc3_file_0.h5
# convert to DDML format
export repo=merge_within_cell #option [identity, merge_within_cell, merge_within_regular_subcell, hdbscan_ms8_mcs40, hdbscan_ms3_mcs10, hdbscan_ms12_mcs12, hdbscan_ms40_mcs40]
python martina_test/convert_to_DDML_format.py outputs/cc3input_$repo/input_cc3_test_small.h5
cp outputs/cc3input_$repo/input_cc3_test_small_ddml_$repo.h5 /eos/user/m/mamozzan/DDML/models/

python martina_test/edm4hep_to_ddml_minimal.py \
  /eos/project/f/fast/edm4hep_frombenchmark/test_small.edm4hep.root \
  --uproot-reader \
  --collections EcalBarrelCollection \
  --algorithm identity \
  --no-shift-and-cut \
  --output outputs/local_frame/ddml_minimal_noSC_simHit_merged_test_small.h5

cp outputs/local_frame/ddml_minimal_noSC_simHit_merged_test_small.h5 /eos/user/m/mamozzan/DDML/models/
