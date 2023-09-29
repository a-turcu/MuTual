import json, tqdm
import numpy as np
from utils_multiple_choice import MuTualProcessor


mutual_embeddings_path = "baseline/embeddings/train.json"
mmlu_embeddings_path = "baseline/embeddings/auxiliary_train.json"
save_rankings_path = 'baseline/embeddings/rankings.json'

# compute mean cosine similarity between one row and all rows in embeddings
def cosine_similarity(row, embeddings):
    row = np.array(row, dtype=np.float64)
    sim_scores = []
    for data_row in embeddings.values():
        data_row = np.array(data_row, dtype=np.float64)
        sim_score = np.dot(row, data_row) / (np.linalg.norm(row) * np.linalg.norm(data_row))
        sim_scores.append(sim_score)

    return np.mean(sim_scores)


with open(mutual_embeddings_path, 'r') as f:
    mtl_embeddings = json.load(f)

with open(mmlu_embeddings_path, 'r') as f:
    mmlu_embeddings = json.load(f)

mmlu_scores = {}

for row, embedding in mmlu_embeddings:
    # compare row with mtl_data
    mmlu_score = cosine_similarity(embedding, mtl_embeddings)
    mmlu_scores[row] = mmlu_score
    

mmlu_scores = sorted(mmlu_scores.items(), key=lambda x: x[1], reverse=True)

with open(save_rankings_path, 'w') as f:
    json.dump(mmlu_scores, f)