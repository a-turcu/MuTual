#!/usr/bin/bash

#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --job-name=bertMC_random_1
#SBATCH --output=/home/%u/job_logs/%x_%A_%u.out

# start environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate mutual-dl4nlp

cd MuTual
srun python -m similarity_augmentations.main fine-tune \
    --model-name bert-base-uncased \
    --model-save-dir finetuned_models/bert-base-uncased-random-1 \
    --scoring-strategy random \
    --percentage 1