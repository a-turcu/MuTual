# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """

from __future__ import absolute_import, division, print_function

import glob
import json
import logging
import os
import re
from io import open
from pprint import pformat
from typing import List

import numpy as np
import tqdm
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class InputExample:
    """A single training/test example for multiple choice."""

    def __init__(self, example_id, contexts, endings, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.contexts = contexts
        self.endings = endings
        self.label = label

    def inplace_remove_speakers(self, replacement: str = ""):
        """Remove speaker labels including ':' and surrounding spaces."""
        match = re.compile(r"\b([mfMF]) ?: ")
        context = match.sub(replacement, self.contexts[0])
        self.contexts = [context] * 4
        self.endings = match.sub(replacement, "\n".join(self.endings)).split("\n")

    def __repr__(self) -> str:
        return (
            f"""<InputExample(context={self.contexts[0]}, """
            f"""endings={pformat(self.endings)}, label={self.label}, """
            f"""id={self.example_id})>"""
        )


class InputFeatures:
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {
                "input_ids": input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
            }
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class MuTualProcessor(DataProcessor):
    """Processor for the MuTual data set."""

    def get_train_examples(self, data_dir, percentage=0.04, train_mode=None, preload_similarities=False):
        
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        file = os.path.join(data_dir, "train")
        file = self._read_txt(file)
        examples = self._create_examples(file, "train")

        data_dir2 = "data/mmlu"
        file = os.path.join(data_dir2, "auxiliary_train")

        if train_mode == "random_mix":
            # randomly add % of MMLU to fine-tuning data
            logger.info("ADDITIONALLY LOOKING AT {} train".format(data_dir2))
            file = self._read_txt(file, percentage)
            examples.extend(self._create_examples(file, "train"))
        
        elif train_mode == "embeddings_mix":
            # add % of MMLU to fine-tuning data considering the euclidean distance between embeddings
            from embeddings import get_closest_embeddings, create_embeddings, get_precomputed_closest_embeddings

            create_embeddings(split="train", data_dir=data_dir, save_dir=f"{data_dir}/embeddings")
            create_embeddings(split="auxiliary_train", data_dir=data_dir2, save_dir=f"{data_dir2}/embeddings")

            if preload_similarities:
                best_ids_path = os.path.join(data_dir2, "embeddings", "best_ids.txt")
                best_emb_id = get_precomputed_closest_embeddings(best_ids_path)
            else:
                best_emb_id = get_closest_embeddings(f"{data_dir}/embeddings", f"{data_dir2}/embeddings", percentage)

            file = self._read_txt(file)
            file = [f for f in file if f["id_emb"] in best_emb_id]
            examples.extend(self._create_examples(file, "train"))
 
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        file = os.path.join(data_dir, "dev")
        file = self._read_txt(file)
        return self._create_examples(file, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        file = os.path.join(data_dir, "test")
        file = self._read_txt(file)
        return self._create_examples(file, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir, percentage=1.0):
        lines = []
        files = glob.glob(input_dir + "/*txt")

        files = np.random.choice(files, int(len(files) * percentage), replace=False)

        for file in tqdm.tqdm(files, desc="read files"):
            with open(file, "r", encoding="ISO-8859-1") as fin:
                data_raw = json.load(fin)
                data_raw["id_emb"] = data_raw["id"]
                data_raw["id"] = file
                lines.append(data_raw)
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for _, data_raw in enumerate(lines):
            id = "%s-%s" % (set_type, data_raw["id"])
            article = data_raw["article"]

            truth = str(ord(data_raw["answers"]) - ord("A"))
            options = data_raw["options"]

            examples.append(
                InputExample(
                    example_id=id,
                    contexts=[
                        article,
                        article,
                        article,
                        article,
                    ],  # this is not efficient but convenient
                    endings=[options[0], options[1], options[2], options[3]],
                    label=truth,
                )
            )
        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for ex_index, example in tqdm.tqdm(
        enumerate(examples), desc="convert examples to features"
    ):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(
            zip(example.contexts, example.endings)
        ):
            text_a = context
            text_b = ending

            inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                return_token_type_ids=True,
                truncation=True
            )
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention!You are poping response,"
                    "you need to try to use a bigger max seq length!"
                )
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = (
                    [0 if mask_padding_with_zero else 1] * padding_length
                ) + attention_mask
                token_type_ids = (
                    [pad_token_segment_id] * padding_length
                ) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length
                )
                token_type_ids = token_type_ids + (
                    [pad_token_segment_id] * padding_length
                )

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            choices_features.append((input_ids, attention_mask, token_type_ids))

        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (input_ids, attention_mask, token_type_ids) in enumerate(
                choices_features
            ):
                logger.info("choice: {}".format(choice_idx))
                logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
                logger.info(
                    "attention_mask: {}".format(" ".join(map(str, attention_mask)))
                )
                logger.info(
                    "token_type_ids: {}".format(" ".join(map(str, token_type_ids)))
                )
                logger.info("label: {}".format(label))

        features.append(
            InputFeatures(
                example_id=example.example_id,
                choices_features=choices_features,
                label=label,
            )
        )

    return features


processors = {
    "mutual": MuTualProcessor,
}

MULTIPLE_CHOICE_TASKS_NUM_LABELS = {
    "mutual",
    4,
}
