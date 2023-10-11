import os, json
import regex as re
import numpy as np
import itertools
import logging
import faiss

from tqdm import trange
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


# create embeddings for either dataset using the distilroberta model
def create_embeddings(split='train', data_dir='data/mutual_plus', save_dir='data/mutual_plus/embeddings'):
	
	from utils_multiple_choice import MuTualProcessor
	
	if os.path.exists(os.path.join(save_dir, f'{split}.json')):
		print(f'{split}.json already exists in {save_dir}.')
		return

	model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

	p = MuTualProcessor()
	data = p._read_txt(os.path.join(data_dir, split))

	logger.info(f"Creating {save_dir}/{split} embeddings")
	print("Creating embeddings")
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
		logger.info(f"Saved {save_dir}/{split}.json")


# calculates which embeddings from mmlu are closest to the embeddings from mutual
def get_closest_embeddings(mutual_dir, mmlu_dir, percentage=0.04):

	# keys are strings; values are lists
	filename_mutual = "distilroberta_mutual.json"
	filename_mmlu = "distilroberta_mmlu.json"
	emb_mutual = json.load(open(os.path.join(mutual_dir, filename_mutual)))
	emb_mmlu = json.load(open(os.path.join(mmlu_dir, filename_mmlu)))

	keys_mmlu = emb_mmlu.keys()
	values_mmlu = emb_mmlu.values()
	distances = dict.fromkeys(keys_mmlu, (0, 0))
	keys_mmlu = list(keys_mmlu) 
	vectors = np.array(list(values_mmlu), dtype=np.float32)
	
	# create FAISS index
	dim = vectors.shape[1]
	index = faiss.IndexFlatL2(dim)
	faiss.normalize_L2(vectors)
	
	index.add(vectors)
	logger.info("Calculating euclidean distances")
	for _, val in emb_mutual.items():
		ref_val = np.array(val, dtype=np.float32)
		ref_val = ref_val.reshape(1, -1)
		faiss.normalize_L2(ref_val)
		k = int(0.3 * percentage * len(keys_mmlu))
		D, I = index.search(ref_val, k)

		for i, dist in zip(I[0], D[0]):
			key = keys_mmlu[i]
			value = (dist + distances[key][0], distances[key][1] + 1)
			distances[key] = value
	
	new_dict = {}
	for key, val in distances.items():
		if val[1] != 0:
			new_dict[key] = val[0] / val[1]

	trunc = min(int(percentage * len(keys_mmlu)), len(new_dict.keys()))
	
	# sort by value, return only keys
	new_dict = dict(sorted(new_dict.items(), key=lambda item: item[1], reverse=False)[:trunc])
	ids = list(new_dict.keys())

	# save list of ids as txt file to not calculate them every run
	with open(os.path.join(mmlu_dir, 'best_ids.txt'), 'w') as f:
		for item in ids:
			f.write("%s\n" % item)

	return ids

# read the embeddings to save time
def get_precomputed_closest_embeddings(scores_file):
	with open(scores_file, 'r') as f:
		best_k = f.read().splitlines()
	return best_k