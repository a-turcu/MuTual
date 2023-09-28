import os, json
import regex as re
import numpy as np
import itertools

from sentence_transformers import SentenceTransformer, util
from utils_multiple_choice import MuTualProcessor

def create_embeddings(model, split='train', data_dir='data/mutual_plus', save_dir='data/mutual_plus/embeddings'):

	p = MuTualProcessor()
	data = p._read_txt(os.path.join(data_dir, split))

	save_dict = {}
	for line in data:
		match = re.compile(r"\b([mfMF]) ?: ")
		context = match.sub("", line["article"])
		embedding = model.encode(context, convert_to_tensor=True)
		save_dict[line['id_emb']] = embedding.tolist()

	# check if save_dir exists
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	with open(f'{save_dir}/{split}.json', 'w') as f:
		json.dump(save_dict, f)


def compare_embeddings(mutual_dir, mmlu_dir):

	# keys are strings; values are lists
	emb_mutual = json.load(open(os.path.join(mutual_dir, 'train.json')))
	emb_mmlu = json.load(open(os.path.join(mmlu_dir, 'short.json')))

	scores = dict.fromkeys(emb_mmlu.keys(), 0)

	for key, val in emb_mmlu.items():
		prod = itertools.product([val], emb_mutual.values())
		for i, j in prod:
			scores[key] += util.pytorch_cos_sim(i, j)

	# divide all scores by the length of mutual
	len_mutual = len(emb_mutual)
	scores = {k: v/len_mutual for k, v in scores.items()}

	# get best k scores
	k = 10
	best_k = sorted(scores, key=scores.get, reverse=True)[:k]

	for key in best_k:
		print(key, scores[key])

	return best_k

compare_embeddings('data/mutual_plus/embeddings', 'data/mmlu/embeddings')

# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# create_embeddings(model, split='auxiliary_train', data_dir='data/mmlu', save_dir='data/mmlu/embeddings')
