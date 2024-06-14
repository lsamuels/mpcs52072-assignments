#!/bin/bash
#
#SBATCH --mail-user=lamonts@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/lamonts/mpcs52072/mpcs52072-assignments/sizes/%j.%N.stdout
#SBATCH --error=/home/lamonts/mpcs52072/mpcs52072-assignments/sizes/%j.%N.stderr
#SBATCH --chdir=/home/lamonts/mpcs52072/mpcs52072-assignments/sizes
#SBATCH --partition=gpu-all
#SBATCH --gres=gpu:1
#SBATCH --job-name=check_dim_gpu

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH=/usr/local/cuda/lib

nvcc checkDimension.cu -o checkDimension
./checkDimension