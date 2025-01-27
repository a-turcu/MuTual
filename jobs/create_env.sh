#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --output=create_env.out
#SBATCH --job-name=env

# # Run script from $HOME
module purge
module load 2022
module load Anaconda3/2022.05

cd MuTual
conda env create -f env.yml