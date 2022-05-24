#!/bin/bash

experiments=(
    # New Baseline
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/cnn/lst/20220510_051437_cnn_lst_co_regression
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/cnn/lst/20220510_051438_cnn_lst_pe_regression
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/cnn/mst/20220510_051441_cnn_mst_co_regression
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/cnn/mst/20220510_051437_cnn_mst_pe_regression
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/cnn/sst/20220510_051437_cnn_sst_co_regression
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/cnn/sst/20220510_051437_cnn_sst_pe_regression
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/umonne/lst/20220510_051437_umonne_lst_co_regression
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/umonne/lst/20220510_051437_umonne_lst_pe_regression
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/umonne/mst/20220510_051437_umonne_mst_co_regression
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/umonne/mst/20220510_051440_umonne_mst_pe_regression
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/umonne/sst/20220510_051437_umonne_sst_co_regression
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/umonne/sst/20220510_051438_umonne_sst_pe_regression
)

datasets=(
    test_gm_full
    test_gm_cut1000
    test_g_full
    test_g_cut1000
)

# ----------------Comands--------------------------
echo "Enqueue multiples submit_eval.sh jobs:"
for dataset in "${datasets[@]}"; do
    for experiment in "${experiments[@]}"; do
        echo "Experment config file: $experiment"
        sbatch --job-name=$dataset.run --export=experiment=$experiment,dataset=$dataset submit_eval.sh
    done
done
