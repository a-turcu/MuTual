# [MuTual](https://huggingface.co/datasets/lighteval/mutual_harness) + [MMLU](https://huggingface.co/datasets/cais/mmlu/viewer/all/auxiliary_train): Similarity Augmentations

## Installation
With Conda or (Micro)Mamba
```sh
micromamba install -f env.yml
```
If you also want to fine-tune a
[multiple choice Transformer model](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForMultipleChoice)
on MuTual(+) with a subset of similar datapoints from MMLU, install either `faiss-cpu` or `faiss-gpu`, e.g. for a GPU
system run:
```sh
micromamba activate mutual-dl4nlp
micromamba install -c conda-forge faiss-gpu
```
## Commands
The CLI is built with [`typer`](https://typer.tiangolo.com/). Commands are thoroughly documented and can be inspected by
running:
```sh
python -m similarity_augmentations.main --help
```
Commands and their subcommands can be inspected by running:
```sh
python -m similarity_augmentations.main <command> --help
python -m similarity_augmentations.main <command> <subcommand> --help
```

### Generating a FAISS vector database
We compare the performance of a multiclass classifier fine-tuned on MuTual alone to that of a model which saw data
augmentation. To this aim, we first create a
[FAISS](https://huggingface.co/datasets/lighteval/mutual_harness/viewer/mutual_plus/train) database for MMLU and MuTual,
which helps us to perform similarity search in embedding space. The minimal command to create e.g. a MuTual(+) database
is:
```sh
python -m similarity_augmentations.main embedding create-db mutual_plus train
```
This will create a FAISS database of MuTual(+) train split at `vectordb/default` using the pretrained encoder
[`all-distilroberta-v1`](https://www.sbert.net/docs/pretrained_models.html).
