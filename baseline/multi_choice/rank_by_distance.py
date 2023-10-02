import json, tqdm
import numpy as np

mutual_embeddings_path = "/gpfs/home2/scur0659/train.json"
mmlu_embeddings_path = "/gpfs/home2/scur0659/auxiliary_train.json"

save_rankings_path = '/gpfs/home2/scur0659/mmlu_rankings.json'

mtl_scores_path = '/gpfs/home2/scur0659/mutual_inter_rankings.json'
mmlu_scores_path = '/gpfs/home2/scur0659/mmlu_inter_rankings.json'


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
    print("Loading mutual embeddings...")
    mtl_embeddings = json.load(f)

with open(mmlu_embeddings_path, 'r') as f:
    mmlu_embeddings = json.load(f)

mmlu_scores = {}

for row in tqdm.tqdm(mmlu_embeddings):
    embedding = mmlu_embeddings[row]
    # compare row with mtl_data
    score = cosine_similarity(embedding, mtl_embeddings)
    mmlu_scores[row] = score
    

sorted_scores = sorted(mmlu_scores.items(), key=lambda x: x[1], reverse=True)

with open(save_rankings_path, 'w') as f:
    json.dump(sorted_scores, f)


with open(mtl_scores_path, 'r') as f:
    print("Loading mutual scores...")
    mtl_scores = json.load(f)

with open(mmlu_scores_path, 'r') as f:
    print("Loading mmlu embeddings...")
    mmlu_scores = json.load(f)

# mtl_scores = [[id, score], [id, score], ...]
mtl_mu = np.mean([score for _, score in mtl_scores])
mtl_sd = np.std([score for _, score in mtl_scores])
mtl_min = np.min([score for _, score in mtl_scores])
mtl_max = np.max([score for _, score in mtl_scores])
print(f"Mutual scores: mean = {mtl_mu}, sd = {mtl_sd}, min={mtl_min}, max={mtl_max}")

# mmlu_scores = [[id, score], [id, score], ...]
mmlu_mu = np.mean([score for _, score in mmlu_scores])
mmlu_sd = np.std([score for _, score in mmlu_scores])
mmlu_min = np.min([score for _, score in mmlu_scores])
mmlu_max = np.max([score for _, score in mmlu_scores])
print(f"MMLU scores: mean = {mmlu_mu}, sd = {mmlu_sd}, min={mmlu_min}, max={mmlu_max}")


mmlu_numpy = np.asarray([score for _, score in mmlu_scores])
print("First element below mean mutual threshold has argmax ", np.argmax(mmlu_numpy<mtl_mu))
print("This many elements in mmlu have scores larger than mutual mean", np.count_nonzero(mmlu_numpy > mtl_mu))