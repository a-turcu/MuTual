from transformers import AutoTokenizer, Data2VecTextModel
import torch

from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#Compute embedding for both lists
embedding_1= model.encode(sentences[0], convert_to_tensor=True)
embedding_2 = model.encode(sentences[1], convert_to_tensor=True)
embedding_3 = model.encode(sentences[2], convert_to_tensor=True)
print(util.pytorch_cos_sim(embedding_1, embedding_2))
print(util.pytorch_cos_sim(embedding_1, embedding_3))
# tokenizer = AutoTokenizer.from_pretrained("facebook/data2vec-text-base")
# model = Data2VecTextModel.from_pretrained("facebook/data2vec-text-base")

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state

# # print(last_hidden_states)
# inputs = tokenizer("Hello, my cat is cute", return_tensors="pt")
# outputs = model(**inputs)

# last_hidden_states2 = outputs.last_hidden_state

# inputs = tokenizer("PULA PULA PIZDA PIZDA", return_tensors="pt")
# outputs = model(**inputs)

# last_hidden_states3 = outputs.last_hidden_state

# # cosine similarity
# cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
# # cos_sim = cos(last_hidden_states, last_hidden_states2)
# # cos_sim2 = cos(last_hidden_states, last_hidden_states3)
# # print(cos_sim)
# # print(cos_sim2)

# print(util.pytorch_cos_sim(last_hidden_states, last_hidden_states2))
# print(util.pytorch_cos_sim(last_hidden_states, last_hidden_states3))