from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from faiss import IndexFlat, IndexFlatL2, read_index, write_index
from sentence_transformers import SentenceTransformer

from similarity_augmentations import consts, utils

logger = utils.get_logger(name=__name__)


def is_faiss_gpu() -> bool:
    ret = False
    try:
        from faiss import GpuIndexFlat

        ret = True
    except:
        ...
    return ret


def try_move_index_to_gpu(index: IndexFlat, device: int = 0) -> Tuple[IndexFlat, bool]:
    moved = False
    if is_faiss_gpu():
        from faiss import GpuIndexFlat, StandardGpuResources, index_cpu_to_gpu

        if not isinstance(index, GpuIndexFlat):
            index = index_cpu_to_gpu(StandardGpuResources(), device, index)
            moved = True
    return index, moved


def check_move_index_to_cpu(index: IndexFlat) -> Tuple[IndexFlat, bool]:
    moved = False
    if is_faiss_gpu():
        from faiss import GpuIndexFlat, index_gpu_to_cpu

        if isinstance(index, GpuIndexFlat):
            logger.info("Moving GPU index to CPU")
            index, moved = index_gpu_to_cpu(index), True
    return index, moved


# NOTE bad design passing folder + names, single filename would be better
def create_or_load_faiss_index(
    savedir: Union[str, Path],
    index_name: str,
    embedder: Optional[Union[SentenceTransformer, str]] = None,
    dataset: Optional[List[str]] = None,
    overwrite: bool = False,
    batch_size: int = 32,
) -> IndexFlatL2:
    assert isinstance(index_name, str)
    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)
    faiss_index_path = savedir / f"{index_name.replace('.faiss', '')}.faiss"
    faiss_index_path_str = str(faiss_index_path)

    if faiss_index_path.is_file() and not overwrite:
        logger.info("Loading FAISS index '%s'", faiss_index_path_str)
        return read_index(faiss_index_path_str)

    assert dataset is not None, f"Pass a dataset for embedding database creation."
    logger.info("Creating FAISS index (overwrite: %s)", overwrite)
    if embedder is None:
        embedder = consts.DEFAULT_EMBEDDING_MODEL
    if isinstance(embedder, str):
        embedder = SentenceTransformer(embedder)
        logger.info("Created embedding model '%s'", consts.DEFAULT_EMBEDDING_MODEL)
    # replace newlines for legacy models https://github.com/openai/openai-python/issues/418
    dataset = [x.replace("\n", " ") for x in dataset]
    logger.info("Encode dataset size %d batch size %d", len(dataset), batch_size)
    embeddings = embedder.encode(
        dataset,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    index = IndexFlatL2(embedder[1].word_embedding_dimension)
    # embeddings are added sequentially, so index.reconstruct(n) == embedder.encode(dataset[n])
    index.add(embeddings)
    logger.info("Saving FAISS index '%s'", faiss_index_path_str)
    write_index(index, faiss_index_path_str)

    return index


def kth_similar_to_all_query_vectors(
    query_index: IndexFlat, target_index: IndexFlat, max_rank: int
) -> np.ndarray:
    if max_rank >= 2048:
        target_index, moved = check_move_index_to_cpu(target_index)
        if moved:
            logger.warning(
                "GpuIndexFlat max rank for K-NN search is 2048, required %d, moved target_index to CPU",
                max_rank,
            )
    # faiss does not fail if searched rank exceeds target_index.ntotal, just
    # returns 0s at exceeding ranks
    max_rank = min(max_rank, target_index.ntotal)
    return target_index.search(
        query_index.reconstruct_batch(range(query_index.ntotal)), max_rank
    )
