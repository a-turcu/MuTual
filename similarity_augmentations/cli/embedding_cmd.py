from enum import Enum
from pathlib import Path

from datasets import load_dataset
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import VectorStore
from typer import Argument, Option, Typer
from typing_extensions import Annotated

from similarity_augmentations import conf, consts, utils
from similarity_augmentations.embedding import crud

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


CREATE_DB_CREATION_HELP_PANEL = "Embedding creation options"


@app.command()
def create_db(
    dataset: Annotated[
        MCDataset,
        Argument(help="Dataset retrieved from [b i]HuggingFace datasets[/b i]"),
    ],
    split: Annotated[
        MCDatasetSplit,
        Argument(help="Dataset split from the chosen [b i]HuggingFace dataset[/b i]"),
    ],
    index_name: Annotated[
        str,
        Option(
            help="[green]FAISS[/green] database index for embedded collection, (defaults to --db-save-dir value)"
        ),
    ] = None,
    embedding_model: Annotated[
        str,
        Option(
            help="A pretrained [green]SentenceTransformer[/green] model from https://www.sbert.net/docs/pretrained_models.html"
        ),
    ] = consts.DEFAULT_SBERT_MODEL,
    db_save_dir: Annotated[
        Path,
        Option(help="Folder where to store the [green]FAISS[/green] database"),
    ] = conf.VECTORDB_DEFAULT_DIR,
    overwrite: Annotated[
        bool,
        Option(
            help="Whether to overwrite [green]FAISS[/green] database if it already exists (creates new embeddings)",
            rich_help_panel=CREATE_DB_CREATION_HELP_PANEL,
            confirmation_prompt=True,
        ),
    ] = False,
    batch_size: Annotated[
        int,
        Option(
            help="Batch size for embedding creation, useful when embedding large amounts of text. A negative value prevents batching.",
            rich_help_panel=CREATE_DB_CREATION_HELP_PANEL,
        ),
    ] = -1,
    remove_speaker_tags: Annotated[
        bool,
        Option(
            help="Whether to strip [M:F] dialogue tags from [b i]MuTual[/b i] dialogues.",
            rich_help_panel=CREATE_DB_CREATION_HELP_PANEL,
        ),
    ] = False,
) -> VectorStore:
    """
    Embed texts with a [b]SentenceTransformer[/b] model and store into a [b]FAISS[/b] database.
    """
    # load dataset split
    split_name = split.value
    logger.info("Loading: %s split: %s", dataset.value, split.value)
    if dataset.value.startswith("mutual"):
        dataset = load_dataset("lighteval/mutual_harness", name=dataset.value)
        if remove_speaker_tags:
            logger.info("MuTual: remove speaker tags")
        split_articles = utils.preprocess_mutual(
            dataset[split_name], remove_speaker_tags=remove_speaker_tags
        )["article"]
    else:
        # mmlu stores 'train' split under a different key than MuTual; it also
        # has a 'dev' split, but we don't use it
        if split_name == "train":
            split_name = "auxiliary_train"
        dataset = load_dataset("cais/mmlu", name=dataset.value)
        split_articles = dataset[split_name]["question"]
    return crud.create_or_load_faiss(
        db_save_dir,
        HuggingFaceEmbeddings(model_name=embedding_model),
        dataset=split_articles,
        index=index_name,
        overwrite=overwrite,
        batch_size=batch_size,
    )


@app.command()
def load_db(
    db_load_dir: Annotated[
        Path,
        Option(help="Folder where to find the [green]FAISS[/green] database"),
    ],
    index_name: Annotated[
        str,
        Option(
            help="[green]FAISS[/green] database index for embedded collection (defaults to --db-load-dir value)"
        ),
    ] = None,
    embedding_model: Annotated[
        str,
        Option(
            help="The [green]SentenceTransformer[/green] model used to create the db"
        ),
    ] = consts.DEFAULT_SBERT_MODEL,
) -> VectorStore:
    """
    Load a [b]FAISS[/b] database from disk.
    """
    return crud.create_or_load_faiss(
        db_load_dir, HuggingFaceEmbeddings(model_name=embedding_model), index=index_name
    )
