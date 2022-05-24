#!/bin/bash

#! Full path to application executable:
rds_dir="$HOME/rds/rds-iris-ip007"
gerumo_dir="$HOME/gerumo2"
singularity_sif="$HOME/rds/rds-iris-ip007/ir-borq1/gerumo2-fixed.sif"
script="$gerumo_dir/tools/train_net.py"

singularity exec $singularity_sif tensorboard --logdir $1
