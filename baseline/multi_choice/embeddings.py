import os, json
import regex as re
import numpy as np

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


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

create_embeddings(model)

with open('data/mutual_plus/embeddings/train.json', 'r') as f:
	train = json.load(f)
	print(len(train))
	print(train.keys())
