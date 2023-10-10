from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from faiss import (GpuIndexFlat, IndexFlat, IndexFlatL2, index_gpu_to_cpu,
                   read_index, write_index)
from sentence_transformers import SentenceTransformer

from similarity_augmentations import consts, utils

logger = utils.get_logger(name=__name__)


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
    embeddings = embedder.encode(dataset, batch_size=batch_size, show_progress_bar=True)

    index = IndexFlatL2(embedder[1].word_embedding_dimension)
    # embeddings are added sequentially, so index.reconstruct(n) == embedder.encode(dataset[n])
    index.add(embeddings)
    logger.info("Saving FAISS index '%s'", faiss_index_path_str)
    write_index(index, faiss_index_path_str)

    return index


# NOTE make sure each big index is on GPU first for 10X speedup (snellius)
def kth_similar_to_all_query_vectors(
    query_index: Union[IndexFlat, GpuIndexFlat],
    target_index: Union[IndexFlat, GpuIndexFlat],
    max_rank: int,
) -> Tuple[np.ndarray]:
    if isinstance(target_index, GpuIndexFlat) and max_rank >= 2048:
        logger.warning("GpuIndexFlat max K-NN search: 2048, moving target_index to CPU")
        target_index = index_gpu_to_cpu(target_index)
    # faiss does not fail if searched rank exceeds target_index.ntotal, just
    # returns 0s at exceeding ranks
    max_rank = min(max_rank, target_index.ntotal)
    return target_index.search(
        query_index.reconstruct_batch(range(query_index.ntotal)), max_rank
    )
