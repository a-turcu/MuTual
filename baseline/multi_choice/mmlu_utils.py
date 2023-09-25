import os

from datasets import load_dataset

answer_labels = {0: "A", 1: "B", 2: "C", 3: "D"}


# Save each row of the dataset from each split into a separate txt file containing a JSON
# representation of the row.
def save_split(split, mmlu_dataset, dataset_path):
    folder = os.path.join(dataset_path, split)
    if not os.path.exists(folder):
        os.mkdir(folder)

    for i, row in enumerate(mmlu_dataset[split]):
        row["answers"] = row.pop("answer")
        row["answers"] = answer_labels[row["answers"]]
        row["article"] = row.pop("question")
        row["options"] = row.pop("choices")
        row["id"] = f"{split}_{i + 1}"

        # replace with double quotes
        row = str(row).replace("'", '"')

        file = os.path.join(dataset_path, split, f"{split}_{i + 1}.txt")

        with open(file, "w", encoding="utf-8") as f:
            f.write(str(row))


if __name__ == "__main__":
    path = os.path.join("data", "mmlu")

    subset = "all"
    mmlu = load_dataset("cais/mmlu", "all")

    save_split("test")
