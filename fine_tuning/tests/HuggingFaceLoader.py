import pytest
from hydra import compose, initialize
from ..util import (
    HuggingFaceLoader,
    ModelArguments,
    DataTrainingArguments,
    HyperparameterTuningArguments,
)
from pathlib import Path
from dataclasses import dataclass
from omegaconf import OmegaConf


@dataclass
class TrainingArgs:
    output_dir: str
    overwrite_output_dir: int
    do_train: int
    do_eval: int
    do_predict: int
    evaluation_strategy: str


@pytest.fixture
def loader():
    initialize(config_path="config", version_base=None)
    config = compose(config_name="main")
    return HuggingFaceLoader(config)


@pytest.fixture
def generic_model_args():
    return {
        "model_name_or_path": "model_name_or_path",
        "tokenizer_name": "tokenizer_name",
        "language": "language",
        "train_language": "train_language",
        "config_name": "config_name",
        "tokenizer_name": "tokenizer_name",
        "cache_dir": "cache_dir",
        "do_lower_case": False,
        "use_fast_tokenizer": True,
        "model_revision": "model_revision",
        "use_auth_token": False,
        "ignore_mismatched_sizes": False,
    }


@pytest.fixture
def generic_data_training_args():
    return {
        "dataset_name": "dataset_name",
        "subset": "subset",
        "evaluation": {"path": "path", "config_name": "config_name"},
        "hyperparameter_tuning": 1,
        "rename_columns": {"old_name": "old_name"},
        "label_names": {
            "train_dataset": {0: "NONE", 1: "ENTAILMENT", 2: "PARAPHRASE"},
            "eval_dataset": {0: "NONE", 1: "ENTAILMENT", 2: "PARAPHRASE"},
            "predict_dataset": {0: "NONE", 1: "ENTAILMENT", 2: "PARAPHRASE"},
        },
        "max_eval_samples": None,
        "max_predict_samples": None,
        "max_seq_length": 128,
        "max_train_samples": None,
        "overwrite_cache": False,
        "pad_to_max_length": True,
    }


@pytest.fixture
def generic_hyperparameter_tuning_args():
    return {
        "training_args": {
            "output_dir": "output_dir",
            "report_to": "report_to",
            "save_strategy": "save_strategy",
            "evaluation_strategy": "evaluation_strategy",
            "logging_strategy": "logging_strategy",
            "load_best_model_at_end": 1,
            "remove_unused_columns": 0,
            "fp16": 1,
        },
        "sweep_config": {
            "method": "method",
            "parameters": {
                "epochs": {"values": [1]},
                "batch_size": {"values": [8, 16, 32, 64]},
                "learning_rate": {"distribution": "distribution", "min": 0, "max": 1},
                "weight_decay": {"values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
            },
        },
    }


@pytest.fixture
def generic_training_args():
    return {
        "output_dir": "output_dir",
        "overwrite_output_dir": 1,
        "do_train": 1,
        "do_eval": 1,
        "do_predict": 1,
        "evaluation_strategy": "evaluation_strategy",
    }


@pytest.fixture
def generic_model_obj(generic_model_args):
    return ModelArguments(**generic_model_args)


@pytest.fixture
def generic_data_training_obj(generic_data_training_args):
    return DataTrainingArguments(**generic_data_training_args)


@pytest.fixture
def generic_training_obj(generic_training_args):
    return TrainingArgs(**generic_training_args)


@pytest.fixture
def generic_hyperparameter_tuning_obj(generic_hyperparameter_tuning_args):
    return HyperparameterTuningArguments(**generic_hyperparameter_tuning_args)


def write_yaml_file(
    path: Path,
    data: ModelArguments
    | DataTrainingArguments
    | TrainingArgs
    | HyperparameterTuningArguments,
) -> None:
    with open(path, "w") as f:
        OmegaConf.save(data, f)


def test_loader_should_return_the_right_args(
    loader,
    generic_model_obj,
    generic_data_training_obj,
    generic_training_obj,
    generic_hyperparameter_tuning_obj,
    generic_model_args,
    generic_data_training_args,
    generic_training_args,
    generic_hyperparameter_tuning_args,
):
    generic_configs_path = Path("config")
    file_name = "generic.yaml"
    write_yaml_file(generic_configs_path / "model" / file_name, generic_model_obj)
    write_yaml_file(
        generic_configs_path / "data_training" / file_name, generic_data_training_obj
    )
    write_yaml_file(
        generic_configs_path / "hyperparameter_tuning" / file_name,
        generic_hyperparameter_tuning_obj,
    )
    write_yaml_file(generic_configs_path / "training" / file_name, generic_training_obj)

    assert loader.get_model_args() == generic_model_args
    assert loader.get_data_training_args() == generic_data_training_args
    assert loader.get_training_args() == generic_training_args
    assert loader.get_hyperparameter_tuning_args() == generic_hyperparameter_tuning_args
