import json, tqdm
import numpy as np

from pathlib import Path
from typing import List, Optional, Union

from sentence_transformers import util as st_util

from datasets import DatasetDict, load_dataset
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores import FAISS, VectorStore
from pathlib import Path


mutual_embeddings_path = "baseline/embeddings/mutual_plus__train__multi-qa-mpnet-base-dot-v1/mutual_plus__train__multi-qa-mpnet-base-dot-v1.faiss"
mmlu_embeddings_path = "baseline/embeddings/mmlu__train__multi-qa-mpnet-base-dot-v1/mmlu__train__multi-qa-mpnet-base-dot-v1.faiss"

save_rankings_path = '/gpfs/home2/scur0659/mmlu_rankings.json'

mtl_scores_path = 'baseline/embeddings/mutual_inter_rankings.json'
mmlu_scores_path = 'baseline/mmlu_rankings.json'


# compute mean cosine similarity between one row and all rows in embeddings
def cosine_similarity(row, embeddings):
	row = np.array(row, dtype=np.float64)
	sim_scores = []
	# for data_row in embeddings.values():
	for data_row in embeddings:
		# data_row = np.array(data_row, dtype=np.float64)
		row = embeddings.index.reconstruct(data_row)
		# sim_score = np.dot(row, data_row) / (np.linalg.norm(row) * np.linalg.norm(data_row))
		sim_score = st_util.cos_sim(row, data_row)
		sim_scores.append(sim_score)

	return np.mean(sim_scores)


def make_vectordb_dir_name(
	subset: str,
	split: str,
	model_name: str,
	vectordb_root: Union[Path, str] = Path("data")
) -> str:
	return str(vectordb_root / f"{subset}__{split}__{model_name}")


# TODO batching, large datasets cannot just use `FAISS.from_texts`
def create_or_load_vectordb(
	savedir: Union[str, Path],
	embedder: Embeddings,
	dataset: Optional[List[str]] = None,
	index_name: Optional[str] = None,
) -> VectorStore:
	savedir = Path(savedir)
	# when not given, assume index name is leaf folder name
	if index_name is None:
		index_name = savedir.stem
	if savedir.is_dir():
		print(f"Loading FAISS db with index {index_name!r} from {str(savedir)!r}")
		return FAISS.load_local(savedir, embedder, index_name)
	assert dataset is not None
	print(f"Creating FAISS db")
	db = FAISS.from_texts(dataset, embedder)
	db.save_local(savedir, index_name)
	print(f"Saved FAISS db with index {index_name!r} from {savedir!r}")
	return db



# embedding_model_name = "multi-qa-mpnet-base-dot-v1"
# embedder_dim = 768
# embedder = HuggingFaceEmbeddings(model_name=embedding_model_name)

# mmlu_db_name = make_vectordb_dir_name("mmlu", "train", embedding_model_name)
# mmlu_db = create_or_load_vectordb(savedir="baseline/embeddings/mmlu__train__multi-qa-mpnet-base-dot-v1", dataset=mmlu_db_name, embedder=embedder)

# mtl_db_name = make_vectordb_dir_name("mutual_plus", "train", embedding_model_name)
# mtl_db = create_or_load_vectordb(savedir="baseline/embeddings/mutual_plus__train__multi-qa-mpnet-base-dot-v1", dataset=mtl_db_name, embedder=embedder)

# with open(mutual_embeddings_path, 'r') as f:
#	 print("Loading mutual embeddings...")
#	 mtl_embeddings = json.load(f)

# with open(mmlu_embeddings_path, 'r') as f:
#	 mmlu_embeddings = json.load(f)

# mmlu_scores = {}


# #prepare index of mtl for similarity search
# Findex = FAISS.IndexFlatL2(embedder_dim)
# Findex.add(mtl_db)


# for row in tqdm.tqdm(mmlu_db.list()):
# 	embedding = mmlu_db.index.reconstruct(row)
# 	# compare row with mtl_data
# 	score = cosine_similarity(embedding, mtl_db)
# 	# get closest k embeddings
# 	# k = 5
# 	# distances, indices = Findex.search(embedding, k)
# 	# score = cosine_similarity(embedding, mtl_embeddings)
# 	mmlu_scores[row] = score
	

# sorted_scores = sorted(mmlu_scores.items(), key=lambda x: x[1], reverse=True)

# with open(save_rankings_path, 'w') as f:
# 	json.dump(sorted_scores, f)


with open(mtl_scores_path, 'r') as f:
	print("Loading mutual scores...")
	mtl_scores = json.load(f)

with open(mmlu_scores_path, 'r') as f:
	print("Loading mmlu scores...")
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
print(f"First element below mean mutual threshold has index {np.argmax(mmlu_numpy<mtl_mu)} of {len(mmlu_numpy)}")


# results for Bogdan's embeddings from 28/09:
# Mutual scores: mean = 0.12840557311490933, sd = 0.0446762604625268, min=-0.018264208600817598, max=0.24551262450001518
# MMLU scores: mean = 0.06182741853990571, sd = 0.04611496630337503, min=-0.07081724730108141, max=0.23270543666408808
# First element below mean mutual threshold has index 8744 of 99842