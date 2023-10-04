#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=40:00:00
#SBATCH --output=train_emb_bert_preloaded.out
#SBATCH --job-name=bert_emb

# Execute program located in $HOME
source activate dl4nlp

cd MuTual
srun python -u baseline/multi_choice/run_multiple_choice.py --train_mode "embeddings_mix" --percentage 0.04 --data_dir "data/mutual_plus" --model_type "bert" --model_name_or_path "bert-base-uncased" --task_name "mutual" --output_dir "output/bert/emb_mix" --do_train --evaluate_during_training --do_lower_case --overwrite_output_dir --overwrite_cache --remove_speakers --preload_similarities