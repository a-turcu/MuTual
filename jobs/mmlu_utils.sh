#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --output=mmlu_utils.out
#SBATCH --job-name=mmlu_utils

# Execute program located in $HOME
source activate dl4nlp

cd MuTual
srun python baseline/multi_choice/mmlu_utils.py