from enum import Enum
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from typer import Argument, Option, Typer
from typing_extensions import Annotated

from similarity_augmentations import conf, consts, utils
from similarity_augmentations.embedding import faiss_utils

logger = utils.get_logger(name=__name__)


class MCDataset(str, Enum):
    mmlu = "mmlu"
    mutual = "mutual"
    mutual_plus = "mutual_plus"


class MCDatasetSplit(str, Enum):
    train = "train"
    validation = "validation"
    test = "test"


app = Typer(name="embedding")


@app.callback()
def embedding_help():
    """Create and retrieve text embeddings."""


CREATE_DB_CREATION_PANEL = "Embedding creation options"


@app.command()
def create_index(
    dataset: Annotated[
        MCDataset,
        Argument(help="Dataset retrieved from [b i]HuggingFace datasets[/b i]."),
    ],
    split: Annotated[
        MCDatasetSplit,
        Argument(help="Dataset split from the chosen [b i]HuggingFace dataset[/b i]."),
    ],
    index_name: Annotated[
        Optional[str],
        Option(
            help=f"[green]FAISS[/green] index name for embedded collection, defaults to [green i]'{conf.VECTORDB_DIR}/DATASET__SPLIT__EMBEDDING-MODEL[__NO-SPEAKER-TAGS]'[/green i]."
        ),
    ] = None,
    embedding_model: Annotated[
        str,
        Option(
            help="A pretrained [green]SentenceTransformer[/green] model from https://www.sbert.net/docs/pretrained_models.html."
        ),
    ] = consts.DEFAULT_EMBEDDING_MODEL,
    index_save_dir: Annotated[
        Path,
        Option(help="Folder where to store the [green]FAISS[/green] index."),
    ] = conf.VECTORDB_DIR,
    overwrite: Annotated[
        bool,
        Option(
            help="Whether to overwrite [green]FAISS[/green] database if it already exists (creates new embeddings).",
            rich_help_panel=CREATE_DB_CREATION_PANEL,
        ),
    ] = False,
    batch_size: Annotated[
        int,
        Option(
            help="Batch size for embedding creation - values > 1000 might not fit into GPU memory.",
            rich_help_panel=CREATE_DB_CREATION_PANEL,
        ),
    ] = 32,
    speaker_tags: Annotated[
        bool,
        Option(
            help="Whether to strip '[MF]:' tags from [b i]MuTual[/b i] dialogues.",
            rich_help_panel=CREATE_DB_CREATION_PANEL,
        ),
    ] = True,
):
    """
    Embed texts with a [b]SentenceTransformer[/b] model and store into a [b]FAISS[/b] index.
    """
    # load dataset split
    split_name = split.value
    logger.info("Loading: %s split: %s", dataset.value, split.value)
    if dataset.value.startswith("mutual"):
        hf_dataset = load_dataset(consts.MUTUAL_HF_PATH, name=dataset.value)
        if speaker_tags:
            logger.info("MuTual: remove speaker tags")
        split_articles = utils.preprocess_mutual(
            hf_dataset[split_name], speaker_tags=speaker_tags
        )["article"]
    else:
        # mmlu stores 'train' split under a different key than MuTual; it also
        # has a 'dev' split, but we don't use it
        if split_name == "train":
            split_name = "auxiliary_train"
        hf_dataset = load_dataset(consts.MMLU_HF_PATH, name="all")
        split_articles = hf_dataset[split_name]["question"]
    if index_name is None:
        index_name = f"{dataset.value}__{split_name}__{embedding_model}"
        index_name += "" if speaker_tags else "__no_speaker_tags"
    return faiss_utils.create_or_load_faiss_index(
        index_save_dir,
        index_name,
        SentenceTransformer(embedding_model),
        dataset=split_articles,
        overwrite=overwrite,
        batch_size=batch_size,
    )


@app.command()
def load_index(
    index_path: Annotated[
        Path, Argument(help="[green]FAISS[/green] index path of embedded collection.")
    ],
):
    """
    Load a [b]FAISS[/b] index from disk.
    """
    return faiss_utils.create_or_load_faiss_index(index_path.parent, index_path.name)
