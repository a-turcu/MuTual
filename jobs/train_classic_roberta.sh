#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --output=train_classic_roberta.out
#SBATCH --job-name=roberta_classic

# Execute program located in $HOME
source activate dl4nlp

cd MuTual
srun python -u baseline/multi_choice/run_multiple_choice.py --percentage 0.5 --data_dir "data/mutual_plus" --model_type "roberta" --model_name_or_path "roberta-base" --task_name "mutual" --output_dir "output/roberta/classic" --do_train --evaluate_during_training