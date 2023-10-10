#!/usr/bin/bash

#SBATCH --time=01:15:00
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=10G
#SBATCH --partition=gpu
#SBATCH --job-name=HF_robertaMC_mutual_baseline
#SBATCH --output=/home/%u/job_logs/%x_%A_%u.out

# start environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate mutual-dl4nlp

cd MuTual
# - run everything: train, validate, test; logs on TensorBoard
# - 200gb in /home/$USER, try to avoid /scratch-* folders (they're temporary)
# srun python -m similarity_augmentations.main fine-tune \
#     --model-save-dir finetuned_models/roberta-base-v1
srun python run_swag.py \
    --model_name_or_path roberta-base \
    --output_dir finetuned_models/roberta-base-hf-script \
    --do_train \
    --do_eval \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --weight_decay 0.01  # non-default
