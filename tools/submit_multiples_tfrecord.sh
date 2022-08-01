#!/bin/bash

# Full
#sbatch --export=experiment="/home/ir-borq1/experiments/regression/cnn/cnn_lst_pe_full.yml",output_dir="/home/ir-borq1/rds/rds-iris-ip007/ir-borq1/DL1_Prod5_GammaDiffuse_Train/all/tf_records/lst" submit_tfrecord.sh
sbatch --export=experiment="/home/ir-borq1/experiments/regression/cnn/cnn_mst_pe_full.yml",output_dir="/home/ir-borq1/rds/rds-iris-ip007/ir-borq1/DL1_Prod5_GammaDiffuse_Train/all/tf_records/mst" submit_tfrecord.sh
sbatch --export=experiment="/home/ir-borq1/experiments/regression/cnn/cnn_sst_pe_full.yml",output_dir="/home/ir-borq1/rds/rds-iris-ip007/ir-borq1/DL1_Prod5_GammaDiffuse_Train/all/tf_records/sst" submit_tfrecord.sh

# Cut
#sbatch --export=experiment="/home/ir-borq1/experiments/regression/cnn/cnn_lst_pe.yml",output_dir="/home/ir-borq1/rds/rds-iris-ip007/ir-borq1/DL1_Prod5_GammaDiffuse_Train/cut_hillas_intensity_1000/tf_records/lst" submit_tfrecord.sh
#sbatch --export=experiment="/home/ir-borq1/experiments/regression/cnn/cnn_mst_pe.yml",output_dir="/home/ir-borq1/rds/rds-iris-ip007/ir-borq1/DL1_Prod5_GammaDiffuse_Train/cut_hillas_intensity_1000/tf_records/mst" submit_tfrecord.sh
#sbatch --export=experiment="/home/ir-borq1/experiments/regression/cnn/cnn_sst_pe.yml",output_dir="/home/ir-borq1/rds/rds-iris-ip007/ir-borq1/DL1_Prod5_GammaDiffuse_Train/cut_hillas_intensity_1000/tf_records/sst" submit_tfrecord.sh

# Evaluation datasets