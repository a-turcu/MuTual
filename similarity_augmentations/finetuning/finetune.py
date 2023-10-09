import math
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (AutoModelForMultipleChoice, PreTrainedTokenizerBase,
                          Trainer, TrainingArguments)
from transformers.tokenization_utils_base import PaddingStrategy
from transformers.trainer_utils import EvalPrediction

from similarity_augmentations import utils

logger = utils.get_logger(name=__name__)


# https://github.com/huggingface/transformers/blob/87499420bff74122a404d1bb0db2e57e6d1ecfe9/examples/pytorch/multiple-choice/run_swag.py#L175
@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that dynamically pads the inputs for multiple choices received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[str, List[List[int]]]]]):
        label_name = "label" if "label" in features[0] else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size, num_choices = len(features), len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)]
            for feature in features
        ]
        flattened_features = list(chain(*flattened_features))
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels under a key accepted by model()
        return batch | {"labels": torch.tensor(labels, dtype=torch.int64)}


def tokenize_dataset(
    dataset: Union[Dataset, DatasetDict], tokenizer: PreTrainedTokenizerBase
) -> Union[Dataset, DatasetDict]:
    """
    Tokenize an NLP multiple choice (MC) dataset.

    Parameters
    ----------
    `dataset`: MC dataset downloaded from HuggingFace; either a single split
        e.g. 'train' or the full `datasets.DatasetDict`. It must store the MC
        context as a string feature `article` and the choices as a string list
        feature `options`.

    `tokenizer`: A `transformers.Tokenizer`.

    Returns
    -------
    The tokenized dataset.
    """

    def tokenize_fn(
        datapoints: Dict[str, List[str]]
    ) -> Dict[str, List[List[List[int]]]]:
        n_options = len(datapoints["options"][0])
        # Repeat each context as many times along with each continuation
        full_options = [
            " ".join([article, opt])
            for article, options in zip(datapoints["article"], datapoints["options"])
            for opt in options
        ]
        # Tokenize
        tokenized_examples = tokenizer(full_options, truncation=True)
        # Un-flatten
        return {
            k: [v[i : i + n_options] for i in range(0, len(v), n_options)]
            for k, v in tokenized_examples.items()
        }

    return dataset.map(tokenize_fn, batched=True)


def evaluate_rankings(eval_predictions: EvalPrediction) -> Dict[str, float]:
    """Compute R@[1,2,3] and MRR for multiclass classification ranking problem."""
    # NOTE loss gets included automatically by `transformers.Trainer`
    predictions, labels = eval_predictions
    if len(labels.shape) < 2:
        # add batch dimension to labels if necessary
        labels = labels[..., None]
    ranked_predictions = np.argsort(-predictions)
    recalls = (ranked_predictions == labels).astype(np.float32)
    # retrieved documents are all relevant in this case (denominator)
    mean_recalls_at = recalls.mean(0)
    # compute recalls until max_rank - 1 (recall at last rank is 1.0)
    recalls_dict = {
        f"R@{k}": mean_recalls_at[:k].sum().item()
        for k in range(1, len(mean_recalls_at))
    }
    inverse_ranks = 1 / (np.argwhere(ranked_predictions == labels)[:, 1] + 1)
    return recalls_dict | {"MRR": inverse_ranks.mean().item()}


def finetune(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    epochs: int,
    batch_size: int,
    model_save_dir: Union[str, Path],
    model: AutoModelForMultipleChoice,
    tokenizer: PreTrainedTokenizerBase,
    compute_metrics_fn: Callable[[EvalPrediction], Dict] = evaluate_rankings,
    metric_for_best_model: str = "MRR",
    resume_checkpoint: Optional[str] = None,
) -> Trainer:
    # make sure that cyclic evaluation and logging is done even if the dataset
    # is small (at least one)
    n_train_step = int(math.ceil(len(train_dataset) / batch_size) * epochs)
    cycle_steps = min(TrainingArguments.logging_steps, n_train_step)
    steps_dict = {f"{p}_steps": cycle_steps for p in ["logging", "eval", "save"]}
    logger.info(
        "Total training steps: %d, log-eval-save steps: %d",
        n_train_step,
        cycle_steps,
    )
    train_args = TrainingArguments(
        model_save_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        evaluation_strategy="steps",
        overwrite_output_dir=bool(resume_checkpoint),
        **steps_dict,
    )
    # NOTE If the dataset is already tokenized tokenizer shouldn't be used,
    # still useful to insert into checkpoints and use when loading the model
    trainer = Trainer(
        model,
        train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        compute_metrics=compute_metrics_fn,
    )
    logger.info(
        "Created trainer with callbacks:\n%s", trainer.callback_handler.callback_list
    )

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    # evaluate one last time in case save_steps is not multiple of n_train_step,
    # in that case we might lose a best checkpoint
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    # save very last checkpoint
    # NOTE best model found is indicated in the 'trainer_state.json' file in
    # this last checkpoint dir
    trainer._save_checkpoint(trainer.model, None, metrics=eval_metrics)
    return trainer
