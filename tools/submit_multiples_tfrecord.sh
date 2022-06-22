#!/bin/bash

# Full
sbatch --export=experiment=,output_dir= submit_tfrecord.sh
sbatch --export=experiment=,output_dir= submit_tfrecord.sh
sbatch --export=experiment=,output_dir= submit_tfrecord.sh

# Cut
sbatch --export=experiment=,output_dir= submit_tfrecord.sh
sbatch --export=experiment=,output_dir= submit_tfrecord.sh
sbatch --export=experiment=,output_dir= submit_tfrecord.sh

# Evaluation datasets