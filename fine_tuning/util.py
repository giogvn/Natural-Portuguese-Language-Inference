import datasets as ds

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from transformers import (
    XLMRobertaTokenizer,
    TrainingArguments,
    XLMRobertaForSequenceClassification,
)
from omegaconf import DictConfig
import pandas as pd
import evaluate

HYPERPARAMETER_TUNING_MAX_TRAIN_SAMPLES = 1000
HYPERPARAMETER_TUNING_MAX_EVAL_SAMPLES = 1000
HYPERPARAMETER_TUNING_MAX_PREDICT_SAMPLES = 1000


@dataclass
class HyperparameterTuningArguments:
    """
    Arguments pertaining to what hyperparameters we are going to tune

    """

    method: str = field(
        default="random",
        metadata={"help": ("The hyperparameter search strategy.")},
    )

    parameters: dict[str, dict] = field(
        default_factory=lambda: {},
        metadata={"help": ("The hyperparameters that will be tuned.")},
    )

    """epochs: dict[str, list[int] | float | int | str] = field(
        default={"values": [3]},
        metadata={
            "help": ("The number of complete passes through the training dataset.")
        },
    )

    batch_size: dict[str, list[int] | float | int | str] = field(
        default={"values": [32, 64]},
        metadata={
            "help": (
                "The number of training samples to work through before the model's internal parameters are updated"
            )
        },
    )

    learning_rate: dict[str, list[int] | float | int | str] = field(
        default={"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
        metadata={
            "help": (
                "The pace at which the gradient descent updates the network's parameters"
            )
        },
    )

    weight_decay: dict[str, list[int | float] | float | int | str] = field(
        default={"values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
        metadata={
            "help": (
                "The penalization parameter directly proportional do a model's complexity added to its loss function"
            )
        },
    )"""


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default="assin",
        metadata={"help": ("The name in HuggingFace of the dataset used for training")},
    )

    evaluation: dict[str, str] = field(
        default="accuracy",
        metadata={
            "help": (
                "The name of the metric in the HuggingFace Hub to be used in the model evaluation along with its subset's name "
            )
        },
    )

    rename_columns: Optional[dict[str, str]] = field(
        default=None,
        metadata={
            "help": ("A mapping of dataset columns to rename to their new names")
        },
    )

    subset: Optional[str] = field(
        default=None,
        metadata={"help": ("Defining the name of the dataset configuration")},
    )

    hyperparameter_tuning: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "Interpreted as a boolean, defines if the hyperparameter tuning should be performed"
            )
        },
    )

    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )

    subset: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Defining the name of the dataset subset. For example, it can be the pt-br version of it."
            )
        },
    )

    label_names: dict[str, dict] = field(
        default=None,
        metadata={
            "help": (
                "The name of the labels in each dataset split available (usually train, validation and predict) "
            )
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    language: Optional[str] = field(
        default=None,
        metadata={
            "help": "Evaluation language. Also train language if `train_language` is set to None."
        },
    )
    train_language: Optional[str] = field(
        default=None,
        metadata={
            "help": "Train language if it is different from the evaluation language."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={
            "help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )


class HuggingFaceLoader:
    def __init__(self, config: DictConfig):
        self.config = config

    def get_hyperparameter_tuning_args(self) -> dict:
        return dict(self.config.hyperparameter_tuning)

    def get_model_args(self) -> dict:
        return dict(self.config.model)

    def get_data_training_args(self) -> dict:
        return dict(self.config.data_training)

    def get_training_args(self) -> dict:
        return dict(self.config.training)

    def get_model_name(self) -> str:
        return self.config.model.model_name_or_path[0]

    def get_dataset_subset(self) -> str:
        return self.config.dataset.subset[0]

    def get_tokenizer_name(self) -> str:
        return self.config.model.tokenizer[0]

    def get_max_train_samples(self) -> int:
        if self.config.hyperparameter_tuning[0]:
            return HYPERPARAMETER_TUNING_MAX_TRAIN_SAMPLES
        else:
            return None

    def get_max_eval_samples(self) -> int:
        if self.config.hyperparameter_tuning[0]:
            return HYPERPARAMETER_TUNING_MAX_EVAL_SAMPLES
        else:
            return None

    def get_max_predict_samples(self) -> int:
        if self.config.hyperparameter_tuning[0]:
            return HYPERPARAMETER_TUNING_MAX_PREDICT_SAMPLES
        else:
            return None

    def load_train_dataset(
        self,
    ):
        split = "train"
        dataset_name = self.config.dataset.name[0]
        subset = self.config.dataset.subset[0]
        return ds.load_dataset(dataset_name, subset, split=split)

    def load_test_dataset(self):
        split = "test"
        dataset_name = self.config.dataset.name[0]
        subset = self.config.dataset.subset[0]
        return ds.load_dataset(dataset_name, subset, split=split)

    def get_model(self):
        model_name = self.config.model.name[0]
        model_type = self.config.model.type[0]
        num_labels = int(self.config.dataset.num_labels[0])

        if model_type == "XLMRobertaForSequenceClassification":
            return XLMRobertaForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )

    def get_tokenizer(self):
        tokenizer = self.config.model.tokenizer[0]
        if tokenizer == "xlmR_tokenizer":
            return self.xlmR_tokenizer

    def get_metric(self):
        metric_name = self.config.hyperparameter.metric[0]
        return evaluate.load(metric_name)

    def xlmR_tokenizer(self, input: dict) -> list:
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        return tokenizer(
            input["premise"], input["hypothesis"], padding="max_length", truncation=True
        )

    def get_checkpoints_and_preds_dir_name(self) -> str:
        return self.config.output.checkpoints_and_preds[0]


def get_dataset_splits_and_configs(
    dataset_name: str, all_configs: list
) -> dict[str:str]:
    all_splits = {}
    for config in all_configs:
        split = get_dataset_split_names(dataset_name, config_name=config)
        all_splits[config] = split

    return all_splits


def get_xlmR_pairs_to_tokenize(
    df: pd.DataFrame, cols: tuple[str, str, int]
) -> list[list[tuple[str, str], int]]:
    pairs = []
    for index, row in df.iterrows():
        hyp_prem = (row[cols[0]], row[cols[1]])
        ent = row[cols[2]]
        pairs.append([hyp_prem, ent])

    return pairs


def get_xlmR_tokens(df: pd.DataFrame, tokenizer) -> list:
    df.drop(["sentence_pair_id", "relatedness_score"], axis=1, inplace=True)

    to_tokenize = [(row["premise"], row["hypothesis"]) for _, row in df.iterrows()]

    encoded_inputs = tokenizer.batch_encode_plus(
        to_tokenize,
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=512,
        return_token_type_ids=True,
    )

    return encoded_inputs
