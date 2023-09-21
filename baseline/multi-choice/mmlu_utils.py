import os

from datasets import load_dataset

path = os.path.join("data", "mmlu")

subset = "all"
mmlu = load_dataset('cais/mmlu', 'all')

# Save each row of the dataset from each split into a separate txt file containing a JSON
# representation of the row.

answer_labels = {0: "A", 1: "B", 2: "C", 3: "D"}

def save_split(split):
	
	
	folder = os.path.join(path, split)
	if not os.path.exists(folder):
		os.mkdir(folder)

	for i, row in enumerate(mmlu[split]):
		row["answers"] = row.pop("answer")
		row["answers"] = answer_labels[row["answers"]]
		row["article"] = row.pop("question")
		row["options"] = row.pop("choices")
		row["id"] = f"{split}_{i + 1}"	
		
		# replace with double quotes
		row = str(row).replace("'", '"')

		file = os.path.join(path, split, f"{split}_{i + 1}.txt")
		
		with open(file, 'w', encoding='utf-8') as f:
			f.write(str(row))

save_split("test")