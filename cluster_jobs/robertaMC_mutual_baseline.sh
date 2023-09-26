#!/usr/bin/bash

#SBATCH --time 02:00:00
#SBATCH --ntasks 1
#SBATCH --gpus 1
#SBATCH --partition gpu
#SBATCH --job-name=robertaMC_mutual_baseline
#SBATCH --output=/home/%u/job_logs/%x_%A_%u.out

# start environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate mutual-dl4nlp

# curiosity (spoiler: A100)
nvidia-smi

cd MuTual
# - run everything: train, validate, test; logs on TensorBoard
# - 200gb in /home/$USER, try to avoid /scratch-* folders (they're temporary)
srun python -m baseline.multi_choice.run_multiple_choice \
    --seed 33 \
    --data_dir data/mutual_plus \
    --output_dir output \
    --cache_dir cache \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --task_name mutual \
    --do_lower_case \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training
