#!/bin/bash

experiments=(
    # Baseline
    # Cut
    ## cnn
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/cnn/sst/20220712_142834_cnn_sst_cut_regression
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/cnn/mst/20220712_142834_cnn_mst_cut_regression
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/cnn/lst/20220712_142834_cnn_lst_cut_regression
    ## umonne large
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/umonne_large/sst/20220712_143810_umonne_large_sst_cut_regression
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/umonne_large/mst/20220712_143154_umonne_large_mst_cut_regression
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/umonne_large/lst/20220712_142834_umonne_large_lst_cut_regression
    ## umonne small
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/umonne_small/sst/20220712_145120_umonne_small_sst_cut_regression
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/umonne_small/mst/20220712_144829_umonne_small_mst_cut_regression
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/umonne_small/lst/20220712_144824_umonne_small_lst_cut_regression
    # Full
    ## cnn
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/cnn/lst/20220712_142834_cnn_lst_full_regression
    ## umonne large
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/umonne_large/lst/20220712_143154_umonne_large_lst_full_regression
    ## umonne small
    /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/umonne_small/lst/20220712_144826_umonne_small_lst_full_regression

    # Small experiments
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/cnn/lst/20220510_051438_cnn_lst_pe_regression
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/cnn/mst/20220510_051437_cnn_mst_pe_regression
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/cnn/sst/20220510_051437_cnn_sst_pe_regression
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/cnn/lst/20220616_164915_cnn_lst_pe_full_regression
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/umonne/lst/20220510_051437_umonne_lst_pe_regression
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/umonne/mst/20220510_051440_umonne_mst_pe_regression
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/umonne/sst/20220510_051438_umonne_sst_pe_regression
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/umonne/lst/20220616_164915_umonne_lst_pe_full_regression
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/umonne/lst/20220616_164941_umonne_lst_pe_big_full_regression

    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/cnn/lst/20220510_051437_cnn_lst_co_regression
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/cnn/mst/20220510_051441_cnn_mst_co_regression
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/cnn/sst/20220510_051437_cnn_sst_co_regression
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/umonne/lst/20220510_051437_umonne_lst_co_regression
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/umonne/mst/20220510_051437_umonne_mst_co_regression
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/new_baseline/umonne/sst/20220510_051437_umonne_sst_co_regression

    # Smoothing
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220604_072240_umonne_lst_full_l21000
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220604_072240_umonne_lst_full_l2100
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220604_072240_umonne_lst_full_l210
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220604_072240_umonne_lst_cut_l21000
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220604_072240_umonne_lst_cut_l2100
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220604_072240_umonne_lst_cut_l210

    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220607_061225_umonne_lst_full_e1000
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220607_060403_umonne_lst_full_e100
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220607_060403_umonne_lst_full_e10
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220607_060403_umonne_lst_cut_e1000
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220607_060403_umonne_lst_cut_e100
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220607_060403_umonne_lst_cut_e10

    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220607_060403_umonne_lst_full_ne1000
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220607_060403_umonne_lst_full_ne100
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220607_060403_umonne_lst_full_ne10
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220607_060403_umonne_lst_cut_ne1000
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220607_060403_umonne_lst_cut_ne100
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220607_060403_umonne_lst_cut_ne10
    
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220607_062759_umonne_lst_cut_l210_adam
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220607_062927_umonne_lst_full_l210_adam

    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220608_071536_umonne_lst_cut_zero_sgd
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220608_071537_umonne_lst_full_zero_sgd
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220608_071540_umonne_lst_cut_zero_adam
    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220608_071543_umonne_lst_full_zero_adam

    # /home/ir-borq1/rds/rds-iris-ip007/ir-borq1/smooth_experiments/20220610_054333_umonne_lst_full_l25_adam
)

datasets=(
    test_gd_full
    test_gd_cut1000
    #test_g_full
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
