import logging
import re
import sys
from typing import Dict, List, Optional, Union

from datasets import ClassLabel, Dataset, DatasetDict


def get_logger(
    log_level: Union[str, int] = "INFO", name: Optional[str] = None
) -> logging.Logger:
    logger = logging.getLogger(name or __name__)
    if isinstance(log_level, str):
        log_level = logging.getLevelName(log_level)
    logger.setLevel(log_level)
    handler = logging.StreamHandler(stream=sys.stdout)
    fmt_str = "[%(asctime)s: %(levelname)s] %(message)s (%(name)s:%(lineno)s)"
    handler.setFormatter(logging.Formatter(fmt=fmt_str))
    logger.addHandler(handler)
    return logger


def preprocess_mutual(
    mutual: Union[Dataset, DatasetDict], remove_speaker_tags: bool = False
) -> Union[Dataset, DatasetDict]:
    """
    Preprocess MuTual dataset.

    Parameters
    ----------
    `mutual`: MuTual dataset downloaded from HuggingFace; either a single
        split e.g. 'train' or the full `datasets.DatasetDict`.

    `remove_speaker_tags`: Whether to remove '[MF]:' tags from dialogues and continuations.

    Returns
    -------
    The MuTual dataset with

    - feature 'answers' renamed to 'labels'
    - feature 'labels' cast to numeric `datasets.ClassLabel`
    - dropped unused feature 'split'
    - speaker tags optionally removed
    """

    def tags_processor(datapoints: Dict[str, List]) -> Dict[str, List]:
        ret = [
            (match.sub("", dialogue), match.sub("", "\n".join(choices)).split("\n"))
            for dialogue, choices in zip(datapoints["article"], datapoints["options"])
        ]
        datapoints.update(dict(zip(["article", "options"], map(list, zip(*ret)))))
        return datapoints

    cast_fn = lambda split: split.cast_column(
        "labels", ClassLabel(num_classes=4, names=["A", "B", "C", "D"])
    )
    mutual = mutual.rename_column("answers", "labels")
    if isinstance(mutual, DatasetDict):
        features = mutual["train"].features
        for split in ["train", "validation"]:
            mutual[split] = cast_fn(mutual[split])
    else:
        features = mutual.features
        mutual = cast_fn(mutual)
    if "id" in features:
        mutual = mutual.remove_columns("id")
    if remove_speaker_tags:
        match = re.compile(r"\b([mfMF]) ?: ")
        mutual = mutual.map(tags_processor, batched=True)
    return mutual
