#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --output=rank_by_distance.out
#SBATCH --job-name=scores
# Execute program located in $HOME
source activate dl4nlp

srun python /gpfs/home2/scur0659/rank_by_distance.py