#!/bin/bash
#
#SBATCH --mail-user=lamonts@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/lamonts/mpcs52072/mpcs52072-assignments/vector/%j.%N.stdout
#SBATCH --error=/home/lamonts/mpcs52072/mpcs52072-assignments/vector/%j.%N.stderr
#SBATCH --chdir=/home/lamonts/mpcs52072/mpcs52072-assignments/vector
#SBATCH --partition=gpu-all
#SBATCH --gres=gpu:1
#SBATCH --job-name=vec_1_nss_job

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH=/usr/local/cuda/lib

nvcc vec_1_n.cu -o vec_1_n
./vec_1_n
