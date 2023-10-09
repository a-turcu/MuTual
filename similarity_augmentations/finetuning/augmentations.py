import math
from enum import Enum
from functools import reduce
from itertools import chain
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from faiss import (GpuIndexFlatL2, IndexFlatL2, StandardGpuResources,
                   index_cpu_to_gpu)
from numpy.random import Generator

from similarity_augmentations import utils
from similarity_augmentations.embedding import faiss_utils

logger = utils.get_logger(name=__name__)


# NOTE copied from LangChain to add 'random'
class DataSelectionStrategy(str, Enum):
    euclidean_distance = "euclidean-distance"
    max_inner_product = "max-inner-product"
    dot_product = "dot-product"
    jaccard = "jaccard"
    cosine = "cosine"
    random = "random"


def to_mmlu_size(p: float, mmlu_len: int) -> int:
    n_samples = int(math.ceil(mmlu_len * p))
    assert (
        n_samples <= mmlu_len
    ), f"Cannot pick {n_samples} samples from {mmlu_len} total, percentage must be <= 1"
    return n_samples


def random_augmentation(p: float, mmlu: Dataset, rng: Generator) -> List[int]:
    assert rng is not None, f"Missing RNG parameter"
    return list(
        map(int, rng.choice(len(mmlu), to_mmlu_size(p, len(mmlu)), replace=False))
    )


# TODO check if other distance metrics can be used out of the box, they
# probably need a change of index to IndexFlatIP; this involves param
# higher_is_better
def embedding_similarity_augmentation(
    p: float,
    sim_at_ranks_strategy: str,
    mutual_index: IndexFlatL2,
    mmlu_index: IndexFlatL2,
    strategy: DataSelectionStrategy = None,
) -> List[int]:
    # assert strategy != DataSelectionStrategy.random
    n_samples = to_mmlu_size(p, mmlu_index.ntotal)
    # load or create similarity arrays and file
    if torch.cuda.is_available() and not isinstance(mmlu_index, GpuIndexFlatL2):
        logger.info("Moving MMLU index to GPU for faster K-NN search")
        res = StandardGpuResources()
        mmlu_index = index_cpu_to_gpu(res, 0, mmlu_index)
    D, I, _ = find_ranking_lower_bound_for_n_unique_samples(
        mutual_index, mmlu_index, n_samples
    )
    # transform L2 distance to cosine to have in [0, 1]
    # https://github.com/facebookresearch/faiss/wiki/FAQ/7f59d7b610e258df8c15dc808ca54b2325515375#how-can-i-index-vectors-for-cosine-distance
    D = (2 - D) / 2
    sorted_sim_mmlu_points = sort_similar_datapoints_at_ranks(
        I, D, prioritize=sim_at_ranks_strategy
    )
    return [int(d["id"]) for d in sorted_sim_mmlu_points[:n_samples]]


def find_ranking_lower_bound_for_n_unique_samples(
    query_index: IndexFlatL2,
    target_index: IndexFlatL2,
    n_desired_similar: int,
    start_max_rank: int = 5,
) -> Tuple[np.ndarray, np.ndarray, int]:
    start_max_rank = max(1, start_max_rank)
    logger.info(
        "Required %d similar points, search starts at rank %d",
        n_desired_similar,
        start_max_rank,
    )
    for k in range(start_max_rank, query_index.ntotal):
        D, I = faiss_utils.kth_similar_to_all_query_vectors(
            query_index, target_index, k
        )
        n_unique_at_ranks = len(np.unique(I))
        logger.info("From rank 1 to %d are %d unique points", k, n_unique_at_ranks)
        if n_unique_at_ranks >= n_desired_similar:
            break
    return D, I, k


def sort_similar_datapoints_at_ranks(
    I: np.ndarray,
    D: np.ndarray,
    prioritize: str = "size",
    higher_is_better: bool = True,
) -> List[Dict]:
    def merge_fn(acc: Dict, distance_d: List[Dict]) -> Dict:
        for dp_d in distance_d:
            id = dp_d["id"]
            distance = dp_d["distance"]
            similar_to = dp_d["similar_to"]
            if id in acc:
                acc[id]["distance"] = (acc[id]["distance"] + distance) / 2
                acc[id]["similar_to"] += similar_to
                acc[id]["similar_to"] = list(set(acc[id]["similar_to"]))
            else:
                acc[id] = {"distance": distance, "similar_to": similar_to}
        return acc

    mean_distances_at = [
        mean_distance_at_rank(I, D, k, prioritize, higher_is_better)
        for k in range(1, I.shape[1] + 1)
    ]
    merged = reduce(merge_fn, mean_distances_at, {})
    merged = [{"id": id, **d} for id, d in merged.items()]
    return sort_similarities(merged, prioritize, higher_is_better)


def mean_distance_at_rank(
    I: np.ndarray,
    D: np.ndarray,
    rank: int,
    prioritize: str,
    higher_is_better: bool,
) -> List[Dict[str, Union[int, List[int], float]]]:
    rank -= 1
    assert I.shape == D.shape and 0 <= rank <= I.shape[1]
    mean_distance_dicts = []
    for u in np.unique(I[:, rank]):
        distance_idxs = np.argwhere(I[:, rank] == u)
        mean_distance_dicts.append(
            {
                "id": u,
                "distance": D[distance_idxs, rank].mean(),
                "similar_to": list(chain(*distance_idxs.tolist())),
            }
        )
    return sort_similarities(mean_distance_dicts, prioritize, higher_is_better)


def sort_similarities(
    distance_at_rank: List[Dict],
    prioritize: str,
    higher_is_better: bool,
) -> List[Dict]:
    assert prioritize in [
        "size",
        "distance",
    ], "Can only sort similar datapoints by mean distance at ranks or neighborhood size"
    # default initializations
    sort_key, reverse = lambda d: len(d["similar_to"]), True
    if prioritize == "distance":
        sort_key = lambda d: d["distance"]
        if not higher_is_better:
            reverse = False
    return list(sorted(distance_at_rank, key=sort_key, reverse=reverse))


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
