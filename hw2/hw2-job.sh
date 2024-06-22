#!/bin/bash
#
#SBATCH --mail-user=lamonts@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/lamonts/mpcs52072/mpcs52072-assignments/hw2/%j.%N.stdout
#SBATCH --error=/home/lamonts/mpcs52072/mpcs52072-assignments/hw2/%j.%N.stderr
#SBATCH --chdir=/home/lamonts/mpcs52072/mpcs52072-assignments/hw2
#SBATCH --partition=gpu-all
#SBATCH --gres=gpu:1
#SBATCH --job-name=hw2_job

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH=/usr/local/cuda/lib

make 
./grayscale test_img.png test_img_out.png 
