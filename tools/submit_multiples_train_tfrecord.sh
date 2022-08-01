#!/bin/bash

experiments=(
    # New Baseline
    # /home/ir-borq1/experiments/regression/cnn_tfr/cnn_lst_cut.yml
    # /home/ir-borq1/experiments/regression/cnn_tfr/cnn_mst_cut.yml
    # /home/ir-borq1/experiments/regression/cnn_tfr/cnn_sst_cut.yml

    # /home/ir-borq1/experiments/regression/umonne_tfr/umonne_large_lst_cut.yml
    # /home/ir-borq1/experiments/regression/umonne_tfr/umonne_large_mst_cut.yml
    # /home/ir-borq1/experiments/regression/umonne_tfr/umonne_large_sst_cut.yml

    # /home/ir-borq1/experiments/regression/umonne_tfr/umonne_small_lst_cut.yml
    # /home/ir-borq1/experiments/regression/umonne_tfr/umonne_small_mst_cut.yml
    # /home/ir-borq1/experiments/regression/umonne_tfr/umonne_small_sst_cut.yml
  
  
    # /home/ir-borq1/experiments/regression/cnn_tfr/cnn_lst_full.yml
    /home/ir-borq1/experiments/regression/cnn_tfr/cnn_mst_full.yml
    /home/ir-borq1/experiments/regression/cnn_tfr/cnn_sst_full.yml

    /home/ir-borq1/experiments/regression/umonne_tfr/umonne_large_lst_full.yml
    /home/ir-borq1/experiments/regression/umonne_tfr/umonne_large_mst_full.yml
    /home/ir-borq1/experiments/regression/umonne_tfr/umonne_large_sst_full.yml
    
    # /home/ir-borq1/experiments/regression/umonne_tfr/umonne_small_lst_full.yml
    /home/ir-borq1/experiments/regression/umonne_tfr/umonne_small_mst_full.yml
    /home/ir-borq1/experiments/regression/umonne_tfr/umonne_small_sst_full.yml
)

# ----------------Comands--------------------------
echo "Enqueue multiples train_model.sh jobs:"

for experiment in "${experiments[@]}"; do
    echo "Experment config file: $experiment"
    sbatch --export=experiment=$experiment submit_train_tfrecords.sh
done
