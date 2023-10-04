#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=1:30:00
#SBATCH --output=distilroberta_emb.out
#SBATCH --job-name=distilroberta_emb
# Execute program located in $HOME
source activate dl4nlp

srun python baseline/embeddings/save_embeddings.py