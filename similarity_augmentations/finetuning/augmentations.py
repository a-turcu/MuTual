import math
from enum import Enum
from pathlib import Path
from typing import List

from datasets import Dataset
from langchain.embeddings import HuggingFaceEmbeddings
from numpy.random import Generator

from similarity_augmentations.embedding import crud


# NOTE copied from LangChain to add 'random'
class DataSelectionStrategy(str, Enum):
    euclidean_distance = "euclidean-distance"
    max_inner_product = "max-inner-product"
    dot_product = "dot-product"
    jaccard = "jaccard"
    cosine = "cosine"
    random = "random"


def random_augmentation(p: float, mmlu: Dataset, rng: Generator) -> List[int]:
    assert rng is not None, f"Missing RNG parameter"
    n_samples = int(math.ceil(len(mmlu) * p))
    assert n_samples <= len(
        mmlu
    ), f"Cannot pick {n_samples} samples from {len(mmlu)} total, decrease p: {p:.3f}"
    return list(rng.choice(len(mmlu), n_samples, replace=False))


def embedding_similarity_augmentation(
    p: float,
    strategy: DataSelectionStrategy,
    faiss_db_dir: Path,
    mutual_index: str,
    mmlu_index: str,
    model_name: str,
) -> List[int]:
    assert strategy != DataSelectionStrategy.random
    embedder = HuggingFaceEmbeddings(model_name=model_name)
    mutual_db = crud.create_or_load_faiss(faiss_db_dir, mutual_index, embedder)
    mmlu_db = crud.create_or_load_faiss(faiss_db_dir, mmlu_index, embedder)
    # now get the closest points from MMLU according to scoring_strategy
    return []
