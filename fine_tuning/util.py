import datasets as ds


from transformers import EvalPrediction
from datasets import load_dataset, load_metric, Dataset
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from transformers import (
    XLMRobertaTokenizer,
    TrainingArguments,
    XLMRobertaForSequenceClassification,
    EvalPrediction,
    AutoModelForSequenceClassification,
    Trainer,
)
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import evaluate
import wandb
import os
import torch

HYPERPARAMETER_TUNING_MAX_TRAIN_SAMPLES = 1000
HYPERPARAMETER_TUNING_MAX_EVAL_SAMPLES = 1000
HYPERPARAMETER_TUNING_MAX_PREDICT_SAMPLES = 1000
WANDB_PROJECT = "gio_projs/"


@dataclass
class HyperparameterTuningArguments:
    """
    Arguments pertaining to what hyperparameters we are going to tune

    """

    best_model_path: Optional[str] = field(
        default="",
        metadata={"help": ("The path to a pretrained model")},
    )

    optimized_metric: Optional[str] = field(
        default="eval_accuracy",
        metadata={"help": ("The metric to be optimized")},
    )

    do_hyperparameter_tuning: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "Interpreted as a boolean, defines if the hyperparameter tuning should be performed"
            )
        },
    )

    load_optimized_parameters: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "Interpreted as a boolean, defines whether the optimized training arguments should be loaded"
            )
        },
    )

    optimized_metric: str = field(
        default="eval/accuracy",
        metadata={"help": ("The metric one wants to optimize")},
    )

    project_path: str = field(
        default="xlm_roberta_best",
        metadata={"help": ("The project path in Weights & Biases")},
    )

    sweep_id: str = field(
        default="",
        metadata={"help": ("The unique identifier for a sweep in Weights & Biases")},
    )

    training_args: dict = field(
        default_factory=lambda: {},
        metadata={"help": ("The training arguments that won't not be tuned.")},
    )

    sweep_config: dict = field(
        default_factory=lambda: {},
        metadata={
            "help": ("The training arguments that will be tuned and the search method.")
        },
    )

    max_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging or memory saving purposes or quicker training, truncate the number of hyperparameter tuning examples to this "
                "value if set."
            )
        },
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
class CrossTestsArguments:
    """
    Arguments pertaining to what hyperparameters we are going to tune

    """

    output_dir: Optional[str] = field(
        default="cross_test",
        metadata={"help": ("Path to where the cross tests results should be stored")},
    )
    do_cross_tests: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "Interpreted as a boolean, defines if the cross tests should be performed"
            )
        },
    )

    datasets: dict = field(
        default_factory=lambda: {},
        metadata={"help": ("The datasets used for cross-testing the models.")},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    new_label_map: Optional[dict] = field(
        default=None,
        metadata={"help": ("A mapping of dataset labels to their new names")},
    )

    label_map: Optional[dict] = field(
        default=None,
        metadata={"help": ("A mapping of dataset names to their new labels")},
    )

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

    train_split: Optional[str] = field(
        default="train",
        metadata={"help": ("The name of the train split")},
    )

    eval_split: Optional[str] = field(
        default="validation",
        metadata={"help": ("The name of the evaluation dataset split")},
    )

    test_split: Optional[str] = field(
        default="test",
        metadata={"help": ("The name of the test dataset split")},
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

    cross_tests: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "Interpreted as a boolean, defines if the the trained model prediction tests should be made using tests splits from datasets different from that used for training"
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

    modify_labels_and_preds: dict = field(
        default=None,
        metadata={
            "help": (
                "A dictionary where the key is the old label value and the value is the new label value"
            )
        },
    )

    test_dataset_split: Optional[dict[str, str]] = field(
        default="test",
        metadata={"help": ("The name of the test dataset")},
    )
    positive_filter: Optional[dict] = field(
        default=None,
        metadata={
            "help": (
                "A dictionary with a key indicating the column name and its value is list so that dataset rows are filtered by the column values"
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

    download_fine_tuned: bool = field(
        default=False,
        metadata={
            "help": "Will download a fine-tuned model from an Weights & Biases' Artifact object."
        },
    )
    fine_tuned_model_local_dir: Optional[str] = field(
        default="model_fine_tuned",
        metadata={"help": "The fine tuned model files path"},
    )

    fine_tuned_wandb_name_tag: Optional[str] = field(
        default="",
        metadata={"help": "The fine tuned model name tag in Weights & Biases"},
    )
    fine_tuned_checkpoint_path: Optional[str] = field(
        default="",
        metadata={"help": "The fine tuned model checkpoint path"},
    )

    load_fine_tuned: bool = field(
        default=False,
        metadata={"help": "Will load a fine-tuned model from a local directory."},
    )


class HuggingFaceLoader:
    def __init__(self, config: DictConfig):
        self.config = config

    def get_cross_tests_args(self) -> dict:
        return dict(self.config.cross_tests)

    def get_hyperparameter_tuning_args(self) -> dict:
        return dict(self.config.hyperparameter_tuning)

    def get_model_args(self) -> dict:
        return dict(self.config.model)

    def get_data_training_args(self) -> dict:
        return dict(self.config.data_training)

    def get_training_args(self) -> dict:
        return dict(self.config.training)


class WAndBLoader:
    def __init__(self, config):
        self.config = config

    def get_sweep_id(config) -> str:
        if config.sweep_id == "":
            return wandb.sweep(
                OmegaConf.to_container(config.sweep_config),
                project=os.path.basename(config.project_path),
            )
        return config.project_path + "/" + config.sweep_id

    def load_model(self, model_path: str):
        return AutoModelForSequenceClassification.from_pretrained(model_path)

    def get_best_model(config):
        if config.download_fine_tuned:
            run = wandb.init()
            artifact = run.use_artifact(config.fine_tuned_wandb_name_tag, type="model")
            artifact.download(config.fine_tuned_model_local_dir)

        return AutoModelForSequenceClassification.from_pretrained(
            config.fine_tuned_model_local_dir
        )


class PreprocessFunctions:
    def __init__(self, model_name: str, tokenizer, padding, data_args):
        self.model = model_name
        self.tokenizer = tokenizer
        self.padding = padding
        self.data_args = data_args

    def preprocess_function(self, examples):
        encoding = self.tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding=self.padding,
            max_length=self.data_args.max_seq_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"]
        attention_masks = encoding["attention_mask"]
        labels = torch.tensor(examples["label"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
        }

    def albertina_tokenize_function(self, examples):
        return self.tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            truncation=True,
        )

    def get_preprocess_function(self):
        if (
            self.model == "xlm-roberta-base"
            or self.model == "neuralmind/bert-large-portuguese-cased"
        ):
            return self.preprocess_function
        elif self.model == "PORTULAN/albertina-ptbr-base":
            return self.albertina_tokenize_function


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


class Predictor:
    def __init__(
        self,
        dest: str,
        origin: str,
        trainer: Trainer,
        model,
        tokenizer,
        train_class: dict,
        test_label: dict,
        premise_col: str = "premise",
        hypothesis_col: str = "hypothesis",
        by_class_metric: bool = False,
        model_name: str = "transformer_based",
        test_set_subset: str = "",
    ):
        self.dest = dest
        self.origin = origin
        self.trainer = trainer
        self.model = model
        self.tokenizer = tokenizer
        self.premise_col = premise_col
        self.hypothesis_col = hypothesis_col
        self.train_class = train_class
        self.test_label = test_label
        self.by_class_metric = by_class_metric
        self.model_name = model_name
        self.test_set_subset = test_set_subset

    def translate_and_predict(self, sentence_a: str, sentence_b: str) -> str:
        a_b = self.tokenizer(
            sentence_a,
            sentence_b,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        a_b = {key: value.to("cuda:0") for key, value in a_b.items()}
        self.model.to("cuda:0")
        self.model.eval()

        if self.origin.lower() == "assin2" and self.dest.lower() == "dlb/plue":
            with torch.no_grad():
                out_a_b = self.model(**a_b)
                pred_a_b = torch.argmax(out_a_b.logits, dim=1).item()
                if (
                    self.train_class[pred_a_b] == "ENTAILMENT"
                    or self.train_class[pred_a_b] == "PARAPHRASE"
                ):
                    return self.test_label["ENTAILMENT"]
                return pred_a_b

        if self.origin.lower() == "dlb/plue" and (self.dest.lower() == "assin"):
            b_a = self.tokenizer(
                sentence_b,
                sentence_a,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            b_a = {key: value.to("cuda:0") for key, value in b_a.items()}
            with torch.no_grad():
                out_a_b = self.model(**a_b)
                out_b_a = self.model(**b_a)

                pred_a_b = torch.argmax(out_a_b.logits, dim=1).item()
                pred_b_a = torch.argmax(out_b_a.logits, dim=1).item()
                if (
                    self.train_class[pred_a_b] == "contradiction"
                    or self.train_class[pred_a_b] == "neutral"
                ):
                    return self.test_label["NONE"]
                if (
                    self.train_class[pred_a_b]
                    == self.train_class[pred_b_a]
                    == "entailment"
                ):
                    return self.test_label["PARAPHRASE"]

                return self.test_label["ENTAILMENT"]

        if self.origin.lower() == "dlb/plue" and (self.dest.lower() == "assin2"):
            with torch.no_grad():
                out_a_b = self.model(**a_b)
                pred_a_b = torch.argmax(out_a_b.logits, dim=1).item()
                if (
                    self.train_class[pred_a_b] == "contradiction"
                    or self.train_class[pred_a_b] == "neutral"
                ):
                    return self.test_label["NONE"]

                return self.test_label["ENTAILMENT"]

        if self.origin.lower() == "assin" and self.dest.lower() == "assin2":
            with torch.no_grad():
                out_a_b = self.model(**a_b)
                pred_a_b = torch.argmax(out_a_b.logits, dim=1).item()

                if (
                    self.train_class[pred_a_b] == "ENTAILMENT"
                    or self.train_class[pred_a_b] == "PARAPHRASE"
                ):
                    return self.test_label["ENTAILMENT"]
                return pred_a_b

        if self.origin.lower() == "assin2" and self.dest.lower() == "assin":
            b_a = self.tokenizer(
                sentence_b,
                sentence_a,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            b_a = {key: value.to("cuda:0") for key, value in b_a.items()}

            with torch.no_grad():
                out_a_b = self.model(**a_b)
                out_b_a = self.model(**b_a)

                pred_a_b = torch.argmax(out_a_b.logits, dim=1).item()
                pred_b_a = torch.argmax(out_b_a.logits, dim=1).item()

                if self.train_class[pred_a_b] == "NONE":
                    return pred_a_b

                if (
                    self.train_class[pred_a_b]
                    == self.train_class[pred_b_a]
                    == "ENTAILMENT"
                ):
                    return self.test_label["PARAPHRASE"]

                return pred_a_b

    def predict(self, test_dataset: Dataset):
        preds = np.zeros(len(test_dataset), dtype=int)
        references = np.zeros(len(test_dataset), dtype=int)

        for i, entry in enumerate(test_dataset):
            sentence_a = entry[self.premise_col]
            sentence_b = entry[self.hypothesis_col]
            pred = self.translate_and_predict(sentence_a, sentence_b)
            preds[i] = pred
            references[i] = entry["label"]

        metrics = dict()

        accuracy_metric = load_metric("accuracy")
        precision_metric = load_metric("precision")
        recall_metric = load_metric("recall")
        f1_metric = load_metric("f1")

        if self.by_class_metric:
            labels = list(set(references))

            def get_metrics_by_class(metrics):
                return {label: score for label, score in zip(labels, metrics)}

            precisions = precision_metric.compute(
                predictions=preds, references=references, average=None
            )["precision"]
            recalls = recall_metric.compute(
                predictions=preds, references=references, average=None
            )["recall"]
            f1_scores = f1_metric.compute(
                predictions=preds, references=references, average=None
            )["f1"]

            precisions = get_metrics_by_class(precisions)
            recalls = get_metrics_by_class(recalls)
            f1_scores = get_metrics_by_class(f1_scores)
            for test_class, label in self.test_label.items():
                if label in precisions:
                    metric_name = "precision"
                    metrics[test_class + "_" + metric_name] = precisions[label]
                    metric_name = "recall"
                    metrics[test_class + "_" + metric_name] = recalls[label]
                    metric_name = "f1"
                    metrics[test_class + "_" + metric_name] = f1_scores[label]
        metrics["model_name"] = self.model_name
        metrics["train_dataset"] = self.dest
        metrics["test_dataset"] = self.origin
        metrics["test_subset"] = self.test_set_subset
        metrics.update(
            accuracy_metric.compute(predictions=preds, references=references)
        )
        metrics.update(
            precision_metric.compute(
                predictions=preds, references=references, average="weighted"
            )
        )
        metrics.update(
            recall_metric.compute(
                predictions=preds, references=references, average="weighted"
            )
        )
        metrics.update(
            f1_metric.compute(
                predictions=preds, references=references, average="weighted"
            )
        )
        return preds, references, metrics

    def custom_compute_metrics(self, p: EvalPrediction, modify_labels_and_preds: dict):
        metrics = dict()

        accuracy_metric = load_metric("accuracy")
        precision_metric = load_metric("precision")
        recall_metric = load_metric("recall")
        f1_metric = load_metric("f1")

        labels = p.label_ids
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(logits, axis=-1)

        metrics.update(accuracy_metric.compute(predictions=preds, references=labels))
        metrics.update(
            precision_metric.compute(
                predictions=preds, references=labels, average="weighted"
            )
        )
        metrics.update(
            recall_metric.compute(
                predictions=preds, references=labels, average="weighted"
            )
        )
        metrics.update(
            f1_metric.compute(predictions=preds, references=labels, average="weighted")
        )

        return metrics

    def translate(self, a, b) -> str:
        pass

    translation = {
        "ENTAILMENT": "ENTAILMENT",
        "PARAPHRASE": "ENTAILMENT AND ENTAILMENT",
        "NONE": "NONE",
    }
