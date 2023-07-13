import sys

sys.path.append("../..")

import pytest, yaml
from hydra import compose, initialize
from util import HuggingFaceLoader, ModelArguments, DataTrainingArguments
from pathlib import Path
from dataclasses import dataclass
from omegaconf import OmegaConf
from typing import List

# TODO: update tests to deal with changes in the configuration files structure
# TODO: add test to the util's HyperparameterTuningArguments loader


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
    initialize(config_path="../config", version_base=None)
    config = compose(config_name="main")
    return HuggingFaceLoader(config)


@pytest.fixture
def generic_model_args():
    return {
        "model_name_or_path": "model_name",
        "tokenizer_name": "tokenizer_name",
        "config_name": "config_name",
    }


@pytest.fixture
def generic_data_training_args():
    return {
        "dataset_name": "dataset_name",
        "subset": "subset_name",
        "hyperparameter_tuning": 0,
        "max_train_samples": 1,
        "max_predict_samples": 1,
        "label_names": {
            "train_dataset": ["A", "B"],
            "eval_dataset": ["A", "B"],
            "predict_dataset": ["A", "B"],
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


def write_yaml_file(
    path: Path, data: ModelArguments | DataTrainingArguments | TrainingArgs
) -> None:
    print(data)
    with open(path, "w") as f:
        OmegaConf.save(data, f)


def test_loader_should_return_the_right_args(
    loader,
    generic_model_obj,
    generic_data_training_obj,
    generic_training_obj,
    generic_model_args,
    generic_data_training_args,
    generic_training_args,
):
    generic_configs_path = Path("../config")
    file_name = "generic.yaml"
    write_yaml_file(generic_configs_path / "model" / file_name, generic_model_obj)
    write_yaml_file(
        generic_configs_path / "data_training" / file_name, generic_data_training_obj
    )
    write_yaml_file(generic_configs_path / "training" / file_name, generic_training_obj)

    assert loader.get_model_args() == generic_model_args
    assert loader.get_data_training_args() == generic_data_training_args
    assert loader.get_training_args() == generic_training_args
