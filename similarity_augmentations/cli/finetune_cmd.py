from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import load_dataset
from transformers import AutoModelForMultipleChoice, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from typer import Option, Typer
from typing_extensions import Annotated

from similarity_augmentations import conf, consts, utils
from similarity_augmentations.finetuning import augmentations, finetune
from similarity_augmentations.finetuning.augmentations import \
    DataSelectionStrategy

logger = utils.get_logger(name=__name__)


class MuTualSubset(str, Enum):
    mutual = "mutual"
    mutual_plus = "mutual_plus"


app = Typer(name="fine-tune")  # NOTE main entry point


@app.callback()
def finetune_help():
    """Fine-tune a multiclass classifier on [b i]MuTual[/b i]."""


DATA_AUGMENTATION_PANEL = "Data augmentation options"
TRAINING_PANEL = "Training options"

# TODO add tokenizaton cache dir, keeps recreating tokenized datasets when they
# exist already, slow for MMLU


@app.command()
def fine_tune(
    model_name: Annotated[
        str,
        Option(
            help="A pretrained [i]encoder[/i] Transformer checkpoint compatible with architectures model listed at https://www.sbert.net/docs/pretrained_models.html."
        ),
    ] = consts.DEFAULT_MC_MODEL,
    mutual_version: Annotated[
        MuTualSubset,
        Option(help="The MuTual version to fine-tune on."),
    ] = MuTualSubset.mutual_plus.value,
    speaker_tags: Annotated[
        bool,
        Option(help="Whether to strip '[MF]:' tags from [b i]MuTual[/b i] dialogues."),
    ] = True,
    epochs: Annotated[
        int, Option(help="Number mf training epochs.", rich_help_panel=TRAINING_PANEL)
    ] = 3,
    batch_size: Annotated[
        int,
        Option(help="Train and evaluation batch size.", rich_help_panel=TRAINING_PANEL),
    ] = 8,
    model_save_dir: Annotated[
        Optional[Path],
        Option(
            help=f"Trainer output folder, defaults to [green i]'{str(conf.FINETUNED_MODELS_DIR)}/MODEL-NAME'[/green i].",
            rich_help_panel=TRAINING_PANEL,
        ),
    ] = None,
    seed: Annotated[
        int, Option(help="Reproducibility seed.", rich_help_panel=TRAINING_PANEL)
    ] = consts.SEED,
    resume: Annotated[
        bool,
        Option(
            help="Resume from latest checkpoint found in --model-save-dir, or train from scratch and overwrite.",
            rich_help_panel=TRAINING_PANEL,
        ),
    ] = True,
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
            help="Proportion of [b i]MMLU[/b i] datapoints added to [b i]MuTual[/b i] [green i]relative to MuTual size[/green i]. By default, none is added.",
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
    mutual_train, mutual_eval = (
        mutual_tokenized["train"],
        mutual_tokenized["validation"],
    )

    train_split = mutual_train
    if percentage > 0:
        logger.info("***** DATA AUGMENTATION *****")
        mmlu_add_ids = []  # NOTE remove afterwards
        mmlu = load_dataset(consts.MMLU_HF_PATH, name="all")["auxiliary_train"]
        # make same feature names to use same tokenization function
        mmlu = finetune.unify_mutual_mmlu_structure(mmlu)
        logger.info("Tokenizing MMLU")
        mmlu_train = finetune.tokenize_dataset(mmlu, tokenizer)
        scaled_percentage = len(mutual_train) * percentage / len(mmlu_train)
        if scoring_strategy == DataSelectionStrategy.random:
            mmlu_add_ids = augmentations.random_augmentation(
                scaled_percentage, mmlu_train, np.random.default_rng(seed)
            )
        else:  # need valid FAISS indices here
            assert (
                mutual_index and mmlu_index
            ), f"Pass MuTual and MMLU FAISS index names when scoring_strategy != {DataSelectionStrategy.random.value:r} and percentage > 0."
            mmlu_add_ids = augmentations.embedding_similarity_augmentation(
                scaled_percentage,
                scoring_strategy,
                faiss_db_dir,
                mutual_index,
                mmlu_index,
                model_name,
            )
        logger.info(
            "Added %d MMLU datapoints (p: %.3f scaled: %.4f) with strategy: '%s'",
            len(mmlu_add_ids),
            percentage,
            scaled_percentage,
            scoring_strategy.value,
        )
        train_split = finetune.merge_mutual_mmlu(
            mutual_train, mmlu_train, mmlu_merge_ids=mmlu_add_ids
        )

    set_seed(seed)
    model_save_dir = model_save_dir or conf.FINETUNED_MODELS_DIR / model_name
    trainer = finetune.build_trainer(
        train_split,
        mutual_eval,
        epochs,
        batch_size,
        model_save_dir,
        AutoModelForMultipleChoice.from_pretrained(model_name),
        tokenizer,
    )

    checkpoint = None
    if resume:
        checkpoint = get_last_checkpoint(model_save_dir)
        if checkpoint is None:
            logger.info(
                "No checkpoint found in '%s', overwrite folder and train",
                model_save_dir,
            )
        else:
            logger.info("Resume training from checkpoint '%s'", checkpoint)
    trainer.train(resume_from_checkpoint=checkpoint)
    # more or less ~5 minutes per epoch
    # TODO properly save metrics and best checkpoints
    # TODO early stopping callback?
