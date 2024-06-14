#!/bin/bash
#
#SBATCH --mail-user=lamonts@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/lamonts/mpcs52072/mpcs52072-assignments/hello/%j.%N.stdout
#SBATCH --error=/home/lamonts/mpcs52072/mpcs52072-assignments/hello/%j.%N.stderr
#SBATCH --chdir=/home/lamonts/mpcs52072/mpcs52072-assignments/hello
#SBATCH --partition=gpu-all
#SBATCH --gres=gpu:1
#SBATCH --job-name=hello_gpu

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH=/usr/local/cuda/lib

nvcc hello_gpu.cu -o hello_gpu 
./hello_gpu