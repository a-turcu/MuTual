import os, json
import regex as re
import numpy as np
import itertools
import logging

from tqdm import trange
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

def create_embeddings(split='train', data_dir='data/mutual_plus', save_dir='data/mutual_plus/embeddings'):
	
	from utils_multiple_choice import MuTualProcessor
	
	if os.path.exists(os.path.join(save_dir, f'{split}.json')):
		print(f'{split}.json already exists in {save_dir}.')
		return

	model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

	p = MuTualProcessor()
	data = p._read_txt(os.path.join(data_dir, split))

	logger.info(f"Creating {save_dir}/{split} embeddings")

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
  

def create_embeddings_faiss(split='train', data_dir='data/mutual_plus', save_dir='data/mutual_plus/embeddings'):

	from utils_multiple_choice import MuTualProcessor

	if os.path.exists(os.path.join(save_dir, f'{split}.json')):
		print(f'{split}.json already exists in {save_dir}.')
		return

	embedder = HuggingFaceEmbeddings(model_name="all-distilroberta-v1")

	p = MuTualProcessor()
	data = p._read_txt(os.path.join(data_dir, split))

	logger.info(f"Creating {save_dir}/{split} embeddings")

	context_db = [line["article"] for line in data]
	batch_size = int(1e4)
	mmlu_embeddings = np.zeros((len(context_db), embedder.client[1].word_embedding_dimension))

	for i in trange(0, len(context_db), batch_size):
		end = min(i + batch_size, len(context_db))
		mmlu_embeddings[i:end, :] = np.array(embedder.embed_documents(context_db[i:end]))

	# check if save_dir exists
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	with open(f'{save_dir}/{split}.json', 'w') as f:
		json.dump(mmlu_embeddings, f)
		logger.info(f"Saved {save_dir}/{split}.json")


def get_closest_embeddings(mutual_dir, mmlu_dir, percentage=0.04):

	# keys are strings; values are lists
	emb_mutual = json.load(open(os.path.join(mutual_dir, 'train.json')))
	emb_mmlu = json.load(open(os.path.join(mmlu_dir, 'auxiliary_train.json')))

	scores = dict.fromkeys(emb_mmlu.keys(), 0)

	logger.info("Calculating cosine similarity scores")
	for key, val in emb_mmlu.items():
		prod = itertools.product([val], emb_mutual.values())
		for i, j in prod:
			scores[key] += util.pytorch_cos_sim(i, j)

	# divide all scores by the length of mutual
	len_mutual = len(emb_mutual)
	scores = {key: v/len_mutual for key, v in scores.items()}

	k = int(percentage * len(scores))
	# get best k scores
	best_k = sorted(scores, key=scores.get, reverse=True)[:k]

	logger.info("Calculated best k scores")

	return best_k
