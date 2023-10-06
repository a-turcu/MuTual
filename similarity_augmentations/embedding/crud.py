from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores import FAISS, VectorStore
from tqdm import trange

from similarity_augmentations import utils

logger = utils.get_logger(name=__name__)


def batch_create_embeddings(
    embedder: Embeddings, dataset: List[str], batch_size: int
) -> np.ndarray:
    dataset_size = len(dataset)
    assert (
        batch_size <= dataset_size
    ), f"Batch size {batch_size} greater than dataset size {dataset_size}"
    embeddings = np.zeros((dataset_size, embedder.client[1].word_embedding_dimension))
    for i in trange(0, dataset_size, batch_size, desc="Embedding dataset..."):
        end = min(i + batch_size, dataset_size)
        embeddings[i:end, :] = np.array(embedder.embed_documents(dataset[i:end]))
    return embeddings


def create_or_load_faiss(
    savedir: Union[str, Path],
    embedder: Embeddings,
    dataset: Optional[List[str]] = None,
    index: Optional[str] = None,
    overwrite: bool = False,
    batch_size: int = -1,
) -> VectorStore:
    savedir = Path(savedir)
    if not index:
        index = savedir.stem  # when not given, assume index name is leaf folder name
    if savedir.is_dir() and not overwrite:
        logger.info(
            "Loading FAISS index '%s' from '%s' (overwrite: %s)",
            index,
            str(savedir),
            overwrite,
        )
        return FAISS.load_local(savedir, embedder, index)
    assert (
        dataset is not None
    ), f"Cannot load from {str(savedir)!r} or invalid dataset: {dataset}"
    logger.info("Creating FAISS db (overwrite: %s)", overwrite)
    if batch_size < 0:
        batch_size = len(dataset)
    embeddings = batch_create_embeddings(dataset, embedder, batch_size)
    db = FAISS.from_embeddings(list(zip(dataset, embeddings.tolist())), embedder)
    db.save_local(savedir, index)
    logger.info("Saved FAISS index '%s' to '%s'", index, str(savedir))
    return db
