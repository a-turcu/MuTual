import json, tqdm, argparse
import numpy as np

from pathlib import Path
from typing import List, Optional, Union

from sentence_transformers import util as st_util

from datasets import DatasetDict, load_dataset
from pathlib import Path


mutual_embeddings_path = "data/mutual_plus/distilroberta_embeddings/train.json"
mmlu_embeddings_path = "data/mmlu/distilroberta_embeddings/auxiliary_train.json"

mtl_scores_path = 'baseline/embeddings/mutual_distil_rankings.json'
mmlu_scores_path = 'baseline/embeddings/mmlu_distil_rankings.json'


# compute mean similarity between one row and all rows in embeddings
def compute_similarity(row, embeddings):
	sim_scores = []
	for data_row in embeddings.values():
		# sim_score = np.dot(row, data_row) / (np.linalg.norm(row) * np.linalg.norm(data_row))
		sim_score = np.linalg.norm(np.asarray(row) - np.asarray(data_row))
		# sim_score = st_util.cos_sim(row, data_row)
		sim_scores.append(sim_score)

	return np.mean(sim_scores)


argParser = argparse.ArgumentParser()
argParser.add_argument("--mutual", action="store_true",help="Compute mutual scores",)
argParser.add_argument("--mmlu", action="store_true",help="Compute mmlu scores",)
argParser.add_argument("--compare", action="store_true",help="Compare the saved scores",)
args = argParser.parse_args()


with open(mutual_embeddings_path, 'r') as f:
	print("Loading mutual embeddings...")
	mtl_embeddings = json.load(f)
with open(mmlu_embeddings_path, 'r') as f:
	print("Loading mmlu embeddings...")
	mmlu_embeddings = json.load(f)

if args.mutual:
	mtl_scores = {}
	print("Computing mutual scores...")
	for row, emb in tqdm.tqdm(mtl_embeddings.items()):
		# compare row with mtl_data
		mtl_scores[row] = compute_similarity(emb, mtl_embeddings)
	sorted_mtl = sorted(mtl_scores.items(), key=lambda x: x[1], reverse=True)
	with open(mtl_scores_path, 'w') as f:
		json.dump(sorted_mtl, f)
		print("saved mtl scores")

if args.mmlu:
	mmlu_scores = {}
	print("Computing mmlu scores...")
	for row, emb in tqdm.tqdm(mmlu_embeddings.items()):
		# compare row with mtl_data
		mmlu_scores[row] = compute_similarity(emb, mtl_embeddings)
	sorted_mmlu = sorted(mmlu_scores.items(), key=lambda x: x[1], reverse=True)
	with open(mmlu_scores_path, 'w') as f:
		json.dump(sorted_mmlu, f)
		print("saved mmlu scores")

if args.compare:
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