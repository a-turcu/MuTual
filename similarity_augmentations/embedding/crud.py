from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores import FAISS, VectorStore
from tqdm.autonotebook import trange

from similarity_augmentations import consts, utils

logger = utils.get_logger(name=__name__)


def batch_create_embeddings(
    dataset: List[str], embedder: Embeddings, batch_size: int
) -> np.ndarray:
    dataset_size = len(dataset)
    if batch_size > dataset_size:
        logger.warning(
            "Batch size %d greater than dataset size %d, fallback to dataset size",
            batch_size,
            dataset_size,
        )
        batch_size = dataset_size
    embeddings = np.zeros((dataset_size, embedder.client[1].word_embedding_dimension))
    for i in trange(0, dataset_size, batch_size, desc="Embedding dataset..."):
        end = min(i + batch_size, dataset_size)
        embeddings[i:end, :] = np.array(embedder.embed_documents(dataset[i:end]))
    return embeddings


def create_or_load_faiss(
    savedir: Union[str, Path],
    index_name: str,
    embedder: Optional[Union[Embeddings, str]] = None,
    dataset: Optional[List[str]] = None,
    overwrite: bool = False,
    batch_size: int = -1,
) -> VectorStore:
    assert isinstance(index_name, str)
    if embedder is None:
        embedder = consts.DEFAULT_EMBEDDING_MODEL
    if isinstance(embedder, str):
        logger.info("Creating SBERT model: '%s'", embedder)
        embedder = HuggingFaceEmbeddings(model_name=embedder)
    savedir = Path(savedir)
    faiss_local_files = [f"{index_name}.faiss", f"{index_name}.pkl"]
    savedir_faiss_glob = [e.name for e in savedir.glob(f"{index_name}.*")]
    if (
        savedir.is_dir()
        and all(f in savedir_faiss_glob for f in faiss_local_files)
        and not overwrite
    ):
        logger.info("Loading FAISS index '%s' from '%s'", index_name, str(savedir))
        return FAISS.load_local(savedir, embedder, index_name)
    assert dataset is not None, f"Pass a dataset for embedding database creation."
    logger.info("Creating FAISS db (overwrite: %s)", overwrite)
    if batch_size < 0:
        batch_size = len(dataset)
    embeddings = batch_create_embeddings(dataset, embedder, batch_size)
    # use the index in the original texts list as metadata, enables to retrieve
    # id of document found with db.similarity_search() variants
    db = FAISS.from_embeddings(
        list(zip(dataset, embeddings.tolist())),
        embedder,
        metadatas=[{"idx": i} for i in range(len(embeddings))],
    )
    logger.info("Saving FAISS index '%s' to '%s'", index_name, str(savedir))
    db.save_local(savedir, index_name)
    return db
