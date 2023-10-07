from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import (AutoModelForMultipleChoice, PreTrainedTokenizerBase,
                          Trainer, TrainingArguments)
from transformers.tokenization_utils_base import PaddingStrategy
from transformers.trainer_utils import EvalPrediction


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


def unify_mutual_mmlu_structure(mmlu: Dataset) -> Dataset:
    """Rename MMLU features to same ones of preprocessed MuTual."""
    if "subject" in mmlu.features:
        mmlu = mmlu.remove_columns("subject")  # unused column
    # normalize the feature names to MuTual's ones
    to_mutual_map = {"answer": "labels", "choices": "options", "question": "article"}
    return mmlu.rename_columns(to_mutual_map)


def merge_mutual_mmlu(
    mutual: Dataset, mmlu: Dataset, mmlu_merge_ids: Iterable[int] = -1
) -> Dataset:
    """
    Add specific datapoints from MMLU subset 'all' to the MuTual dataset.

    Arguments
    ---------
    `mutual`: A MuTual dataset split e.g. 'train'.

    `mmlu`: A MMLU dataset split e.g. 'auxiliary_train'.

    `mmlu_merge_ids`: The indices of dapoints in MMLU which should be added to
        MuTual. If negative, the two datasets are fully concatenated.

    Parameters
    ----------
    A `datasets.Dataset` with all datapoints from MuTual and selected datapoins
    from MMLU.
    """
    assert "labels" in mutual.features, "Call `utils.preprocess_mutual(mutual)` first"
    # TODO check that feature names are the same before merging, else call unify
    if isinstance(mmlu_merge_ids, int) and mmlu_merge_ids < 0:
        mmlu_merge_ids = range(len(mmlu))
    return concatenate_datasets([mutual, mmlu.select(mmlu_merge_ids)])


def build_trainer(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    model_save_dir: Union[str, Path],
    model: AutoModelForMultipleChoice,
    tokenizer: PreTrainedTokenizerBase,
    compute_metrics_fn: Callable[[EvalPrediction], Dict] = evaluate_rankings,
    batch_size: int = 32,
    overwrite: bool = False,
) -> Trainer:
    train_args = TrainingArguments(
        model_save_dir,
        overwrite_output_dir=overwrite,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        weight_decay=0.01,
        # push_to_hub=True,
    )
    # NOTE If the dataset is already tokenized tokenizer shouldn't be used,
    # still useful to insert into checkpoints and use when loading the model
    return Trainer(
        model,
        train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        compute_metrics=compute_metrics_fn,
    )
