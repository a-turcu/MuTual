#!/usr/bin/bash

#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --partition=gpu
#SBATCH --job-name=robertaMC_mutual_baseline
#SBATCH --output=/home/%u/job_logs/%x_%A_%u.out

# start environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate mutual-dl4nlp

cd MuTual
# - run everything: train, validate, test; logs on TensorBoard
# - 200gb in /home/$USER, try to avoid /scratch-* folders (they're temporary)
srun python -m similarity_augmentations.main fine-tune \
    --model-save-dir finetuned_models/roberta-base-v1
