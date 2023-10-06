from enum import Enum
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForMultipleChoice, AutoTokenizer, set_seed
from typer import Option, Typer
from typing_extensions import Annotated

from similarity_augmentations import conf, consts, utils
from similarity_augmentations.embedding import crud
from similarity_augmentations.finetuning import finetune

logger = utils.get_logger(name=__name__)


# NOTE copied from LangChain to add 'random'
class DataSelectionStrategy(str, Enum):
    euclidean_distance = "euclidean-distance"
    max_inner_product = "max-inner-product"
    dot_product = "dot-product"
    jaccard = "jaccard"
    cosine = "cosine"
    random = "random"


class MuTualSubset(str, Enum):
    mutual = "mutual"
    mutual_plus = "mutual_plus"


app = Typer(name="fine-tune")  # NOTE main entry point


@app.callback()
def finetune_help():
    """Fine-tune a multiclass classifier on [b i]MuTual[/b i]."""


# TODO split args in those for data augmentation and those for training
@app.command()
def fine_tune(
    model_name: Annotated[
        str,
        Option(
            help="A pretrained [i]encoder[/i] Transformer checkpoint compatible with architectures model listed at https://www.sbert.net/docs/pretrained_models.html."
        ),
    ] = consts.DEFAULT_MC_MODEL,
    scoring_strategy: Annotated[
        Optional[DataSelectionStrategy],
        Option(
            help="Distance metric for similarity scoring between [b i]MuTual[/b i] and [b i]MMLU[/b i]."
        ),
    ] = DataSelectionStrategy.random.value,
    percentage: Annotated[
        float,
        Option(
            "--percentage",
            "-p",
            help="Proportion of [b i]MMLU[/b i] datapoints added to [b i]MuTual[/b i]; by default, none is added (baseline).",
        ),
    ] = 0.0,
    speaker_tags: Annotated[
        bool,
        Option(help="Whether to strip '[MF]:' tags from [b i]MuTual[/b i] dialogues."),
    ] = True,
    seed: Annotated[int, Option(help="Reproducibility seed.")] = consts.SEED,
    mutual_version: Annotated[
        MuTualSubset, Option(help="The MuTual version to fine-tune on.")
    ] = MuTualSubset.mutual_plus.value,
    model_save_dir: Annotated[
        Optional[Path],
        Option(
            help=f"Trainer output folder, defaults to [green i]'{str(conf.FINETUNED_MODELS_DIR)}/model-name value'[/green i]."
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        Option(
            help="Whether to overwrite --model-save-dir if it already exists.",
        ),
    ] = False,
    faiss_db_dir: Annotated[
        Path,
        Option(
            help=f"Folder with FAISS files for [b i]MuTual[/b i] and [b i]MMLU[/b i] embedding indexes. [green b]Accessed[/green b] when [green i]percentage > 0 and scoring-strategy != {DataSelectionStrategy.random.value!r}[/green i]."
        ),
    ] = conf.VECTORDB_DIR,
    mutual_index: Annotated[
        Optional[str], Option(help="FAISS index name of MuTual embeddings.")
    ] = None,
    mmlu_index: Annotated[
        Optional[str], Option(help="FAISS index name of MMLU embeddings.")
    ] = None,
):
    """Run fine-tuning, possibly with data augmentation."""
    mutual = load_dataset(consts.MUTUAL_HF_PATH, name=mutual_version.value)
    logger.info("Preprocess MuTual, remove speaker tags: %s", speaker_tags)
    mutual = utils.preprocess_mutual(mutual, remove_speaker_tags=speaker_tags)
    mmlu = load_dataset(consts.MMLU_HF_PATH, name="all")

    # tokenize all datasets first so that they can be cached once
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mutual_tokenized = finetune.tokenize_dataset(mutual, tokenizer)
    mmlu_train_tokenized = finetune.tokenize_dataset(mmlu["auxiliary_train"], tokenizer)

    mmlu_add_ids = []
    if scoring_strategy == DataSelectionStrategy.random:
        # random augmentation, or baseline if percentage == 0
        ...
    elif percentage > 0.0:
        # need valid FAISS indices
        assert (
            mutual_index and mmlu_index
        ), f"Pass MuTual and MMLU FAISS index names when scoring_strategy != {DataSelectionStrategy.random.value:r} and percentage > 0."
        embedder = HuggingFaceEmbeddings(model_name=model_name)
        mutual_db = crud.create_or_load_faiss(faiss_db_dir, mutual_index, embedder)
        mmlu_db = crud.create_or_load_faiss(faiss_db_dir, mmlu_index, embedder)
        # now get the closest points from MMLU according to scoring_strategy
    logger.info(
        "Add %d MMLU datapoints (proportion: %.3f) found with strategy: '%s'",
        len(mmlu_add_ids),
        percentage,
        scoring_strategy.value,
    )
    train_split_tokenized = finetune.merge_mutual_mmlu(
        mutual_tokenized["train"], mmlu_train_tokenized, mmlu_merge_ids=mmlu_add_ids
    )

    set_seed(seed)
    trainer = finetune.build_trainer(
        train_split_tokenized,
        mutual_tokenized["validation"],
        model_save_dir,
        AutoModelForMultipleChoice(model_name),
        tokenizer,
        overwrite=overwrite,
    )
    trainer.train()
    # TODO save metrics
