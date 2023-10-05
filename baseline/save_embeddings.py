from multi_choice.embeddings import create_embeddings

mutual_dir = "data/mutual_plus"
mmlu_dir = "data/mmlu"

create_embeddings(split="train", data_dir=mutual_dir, save_dir=f"{mutual_dir}/distilroberta_embeddings")
create_embeddings(split="auxiliary_train", data_dir=mmlu_dir, save_dir=f"{mmlu_dir}/distilroberta_embeddings")

