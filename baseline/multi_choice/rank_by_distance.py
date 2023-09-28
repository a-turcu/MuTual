import os
from baseline.multi_choice.utils_multiple_choice import MuTualProcessor


mutual_embeddings_path = ""
mmlu_embeddings_path = ""
split = ""

mtl_p = MuTualProcessor()
mtl_data = mtl_p._read_txt(os.path.join(mutual_embeddings_path, split))

mmlu_p = MuTualProcessor()
mmlu_data = mmlu_p._read_txt(os.path.join(mmlu_embeddings_path, split))

# sim_scores = {}

for i, row in enumerate(mmlu_data):
    pass
    # compare row with mtl_data
    # sim_score = cosine_similarity(row, mtl_data)
    # sim_scores[i] = sim_score

# sim_scores = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)