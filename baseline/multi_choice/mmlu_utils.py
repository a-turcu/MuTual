import os
import json
from datasets import load_dataset

answer_labels = {0: "A", 1: "B", 2: "C", 3: "D"}

# Save each row of the dataset from each split into a separate txt file containing a JSON
# representation of the row.
def save_split(split, mmlu_dataset, dataset_path):
    folder = os.path.join(dataset_path, split)
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i, row in enumerate(mmlu_dataset[split]):
        row["answers"] = row.pop("answer")
        row["answers"] = answer_labels[row["answers"]]
        row["article"] = row.pop("question")
        row["options"] = row.pop("choices")
        row["id"] = f"{split}_{i + 1}"

        file = os.path.join(dataset_path, split, f"{split}_{i + 1}.txt")

        with open(file, "w", encoding="utf-8") as f:
            f.write(json.dumps(row))


if __name__ == "__main__":
    path = os.path.join("data", "mmlu")

    subset = "all"
    mmlu = load_dataset("cais/mmlu", "all")
    
    print(mmlu)

    save_split("auxiliary_train", mmlu, path)