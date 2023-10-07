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


DATA_AUGMENTATION_PANEL = "Data augmentation options"


@app.command()
def fine_tune(
    model_name: Annotated[
        str,
        Option(
            help="A pretrained [i]encoder[/i] Transformer checkpoint compatible with architectures model listed at https://www.sbert.net/docs/pretrained_models.html."
        ),
    ] = consts.DEFAULT_MC_MODEL,
    mutual_version: Annotated[
        MuTualSubset, Option(help="The MuTual version to fine-tune on.")
    ] = MuTualSubset.mutual_plus.value,
    speaker_tags: Annotated[
        bool,
        Option(help="Whether to strip '[MF]:' tags from [b i]MuTual[/b i] dialogues."),
    ] = True,
    seed: Annotated[int, Option(help="Reproducibility seed.")] = consts.SEED,
    model_save_dir: Annotated[
        Optional[Path],
        Option(
            help=f"Trainer output folder, defaults to [green i]'{str(conf.FINETUNED_MODELS_DIR)}/MODEL-NAME'[/green i]."
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        Option(
            help="Whether to overwrite --model-save-dir if it already exists.",
        ),
    ] = False,
    scoring_strategy: Annotated[
        Optional[DataSelectionStrategy],
        Option(
            help="Distance metric for similarity scoring between [b i]MuTual[/b i] and [b i]MMLU[/b i].",
            rich_help_panel=DATA_AUGMENTATION_PANEL,
        ),
    ] = DataSelectionStrategy.random.value,
    percentage: Annotated[
        float,
        Option(
            "--percentage",
            "-p",
            help="Proportion of [b i]MMLU[/b i] datapoints added to [b i]MuTual[/b i]; by default, none is added (baseline).",
            rich_help_panel=DATA_AUGMENTATION_PANEL,
        ),
    ] = 0.0,
    faiss_db_dir: Annotated[
        Path,
        Option(
            help=f"Folder with FAISS files for [b i]MuTual[/b i] and [b i]MMLU[/b i] embedding indexes. [green b]Accessed[/green b] when [green i]PERCENTAGE > 0 and SCORING-STRATEGY != {DataSelectionStrategy.random.value!r}[/green i].",
            rich_help_panel=DATA_AUGMENTATION_PANEL,
        ),
    ] = conf.VECTORDB_DIR,
    mutual_index: Annotated[
        Optional[str],
        Option(
            help="FAISS index name of MuTual embeddings.",
            rich_help_panel=DATA_AUGMENTATION_PANEL,
        ),
    ] = None,
    mmlu_index: Annotated[
        Optional[str],
        Option(
            help="FAISS index name of MMLU embeddings.",
            rich_help_panel=DATA_AUGMENTATION_PANEL,
        ),
    ] = None,
):
    """Run fine-tuning, possibly with data augmentation."""
    mutual = load_dataset(consts.MUTUAL_HF_PATH, name=mutual_version.value)
    logger.info("Preprocess MuTual")
    mutual = utils.preprocess_mutual(mutual, speaker_tags=speaker_tags)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    logger.info("Tokenizing all MuTual splits")
    mutual_tokenized = finetune.tokenize_dataset(mutual, tokenizer)

    train_split_tokenized = mutual_tokenized["train"]
    if percentage > 0.0:
        logger.info("DATA AUGMENTATION")
        mmlu_add_ids = []
        mmlu = load_dataset(consts.MMLU_HF_PATH, name="all")["auxiliary_train"]
        # make same feature names to use same tokenization function
        mmlu = finetune.unify_mutual_mmlu_structure(mmlu)
        logger.info("Tokenizing MMLU")
        mmlu_train_tokenized = finetune.tokenize_dataset(mmlu, tokenizer)
        if scoring_strategy == DataSelectionStrategy.random:
            # random augmentation, or baseline if percentage == 0
            ...
        else:  # need valid FAISS indices here
            assert (
                mutual_index and mmlu_index
            ), f"Pass MuTual and MMLU FAISS index names when scoring_strategy != {DataSelectionStrategy.random.value:r} and percentage > 0."
            embedder = HuggingFaceEmbeddings(model_name=model_name)
            mutual_db = crud.create_or_load_faiss(faiss_db_dir, mutual_index, embedder)
            mmlu_db = crud.create_or_load_faiss(faiss_db_dir, mmlu_index, embedder)
            # now get the closest points from MMLU according to scoring_strategy
            ...
        logger.info(
            "Add %d MMLU datapoints (proportion: %.3f) with strategy: '%s'",
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
        model_save_dir or conf.FINETUNED_MODELS_DIR / model_name,
        AutoModelForMultipleChoice.from_pretrained(model_name),
        tokenizer,
        overwrite=overwrite,
    )
    trainer.train()
    # more or less ~5 minutes per epoch
    # TODO properly save metrics and best checkpoints
    # TODO early stopping callback?
