#!/bin/bash

# 1. Find the Docker image.
docker_image="gerumo2:latest"

# 2. Create a tarball of the Docker image.
tar_file="gerumo2.tar"
sudo docker save $docker_image -o $tar_file

# 3. Convert the tarball to a Singularity image.
sif_file="gerumo2.sif"
sudo singularity build $sif_file "docker-archive://$tar_file"
