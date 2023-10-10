#!/usr/bin/bash

#SBATCH --time=01:40:00
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=10G
#SBATCH --partition=gpu
#SBATCH --job-name=bertMC_cos_sim_03_nospeakers
#SBATCH --output=/home/%u/job_logs/%x_%A_%u.out

# start environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate mutual-dl4nlp

cd $HOME/MuTual
srun python -m similarity_augmentations.main fine-tune \
    --model-name bert-base-uncased \
    --model-save-dir finetuned_models/bert-base-uncased-cos-sim-03-nospeakers \
    --scoring-strategy dot-product \
    --percentage .3 \
    --faiss-index-dir vector_db \
    --mutual-index mutual_plus__train__all-distilroberta-v1__no_speaker_tags.faiss \
    --mmlu-index mmlu__auxiliary_train__all-distilroberta-v1.faiss \
    --epochs 10 \
    --no-speaker-tags
