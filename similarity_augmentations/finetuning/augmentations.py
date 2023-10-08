import math
from enum import Enum
from pathlib import Path
from typing import Iterable, List

from datasets import Dataset, concatenate_datasets
from langchain.embeddings import HuggingFaceEmbeddings
from numpy.random import Generator

from similarity_augmentations.embedding import crud, utils

logger = utils.get_logger(name=__name__)


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


def unify_mutual_mmlu_structure(mmlu: Dataset) -> Dataset:
    """Rename MMLU features to same ones of preprocessed MuTual."""
    if "subject" in mmlu.features:
        mmlu = mmlu.remove_columns("subject")  # unused column
    # normalize the feature names to MuTual's ones
    to_mutual_map = {"answer": "labels", "choices": "options", "question": "article"}
    return mmlu.rename_columns(to_mutual_map)


def merge_mutual_mmlu(
    mutual: Dataset, mmlu: Dataset, mmlu_merge_ids: Iterable[int] = -1
) -> Dataset:
    """
    Add specific datapoints from MMLU subset 'all' to the MuTual dataset.

    The two datasets must have same features.

    Arguments
    ---------
    `mutual`: A MuTual dataset split e.g. 'train'.

    `mmlu`: A MMLU dataset split e.g. 'auxiliary_train'.

    `mmlu_merge_ids`: The indices of dapoints in MMLU which should be added to
        MuTual. If negative, the two datasets are fully concatenated.

    Parameters
    ----------
    A `datasets.Dataset` with all datapoints from MuTual and selected datapoins
    from MMLU.
    """
    assert "labels" in mutual.features, "Call `utils.preprocess_mutual(mutual)` first"
    if isinstance(mmlu_merge_ids, int) and mmlu_merge_ids < 0:
        mmlu_merge_ids = range(len(mmlu))
    return concatenate_datasets([mutual, mmlu.select(mmlu_merge_ids)])
