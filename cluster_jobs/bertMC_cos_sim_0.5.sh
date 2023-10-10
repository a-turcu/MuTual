#!/usr/bin/bash

#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=10G
#SBATCH --partition=gpu
#SBATCH --job-name=bertMC_cos_sim_05
#SBATCH --output=/home/%u/job_logs/%x_%A_%u.out

# start environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate mutual-dl4nlp

cd MuTual
srun python -m similarity_augmentations.main fine-tune \
    --model-name bert-base-uncased \
    --model-save-dir finetuned_models/bert-base-uncased-cos-sim-05 \
    --scoring-strategy dot-product \
    --percentage .5
