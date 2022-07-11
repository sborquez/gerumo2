#!/bin/bash

experiments=(
    # New Baseline
)

# ----------------Comands--------------------------
echo "Enqueue multiples train_model.sh jobs:"

for experiment in "${experiments[@]}"; do
    echo "Experment config file: $experiment"
    sbatch --export=experiment=$experiment submit_train_tfrecord.sh
done
