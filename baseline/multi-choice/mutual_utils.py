import os

import regex as re

from datasets import load_dataset

path = os.path.join("data", "mutual")


mutual = load_dataset('lighteval/mutual_plus', 'test')

# select character to separate speakers' lines in the context
dialog_splitter = ""


def remove_speakers(split):
	folder = os.path.join(path, split)
	if not os.path.exists(folder):
		os.mkdir(folder)

	for i, row in enumerate(mutual[split]):
		# remove speaker labels including : and potentially spaces around :
		row = re.sub(r"\b([mfMF]) ?: ", dialog_splitter, str(row))

		file = os.path.join(path, split, f"{split}_{i + 1}.txt")
		
		with open(file, 'w', encoding='utf-8') as f:
			f.write(str(row))

remove_speakers("test")