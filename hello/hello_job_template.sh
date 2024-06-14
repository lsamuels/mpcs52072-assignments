#!/bin/bash
#
#SBATCH --mail-user=CNetID@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/PATH/TO/PLACE/STDOUT/%j.%N.stdout
#SBATCH --error=/PATH/TO/PLACE/STDERR/%j.%N.stderr
#SBATCH --chdir=/PATH/TO/CHANGE/TO/RUNNING/CUDA/PROG
#SBATCH --partition=gpu-all
#SBATCH --gres=gpu:1
#SBATCH --job-name=NAME_OF_JOB

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH=/usr/local/cuda/lib

### The compliation and execution of your CUDA program below #######