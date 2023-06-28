import sys

sys.path.append("../..")

import pytest, yaml
from hydra import compose, initialize
from util import HuggingFaceLoader
from pathlib import Path
from dataclasses import dataclass
from omegaconf import OmegaConf
from typing import List


@dataclass
class ModelArgs:
    model_name_or_path: str
    config: str
    tokenizer: str


@pytest.fixture
def loader():
    initialize(config_path="../config", version_base=None)
    config = compose(config_name="main")
    return HuggingFaceLoader(config)


@pytest.fixture
def generic_model_args():
    return {
        "model_name_or_path": "model_name",
        "tokenizer": "tokenizer_name",
        "config": "config_name",
    }


@pytest.fixture
def generic_model_obj(generic_model_args):
    return ModelArgs(**generic_model_args)


def write_yaml_file(file_path: Path, data: ModelArgs):
    with open(file_path, "w") as f:
        OmegaConf.save(data, f)


def test_loader_should_return_the_right_model_args(
    loader, generic_model_obj, generic_model_args
):
    write_yaml_file(Path("../config/model/generic.yaml"), generic_model_obj)
    assert loader.get_model_args() == generic_model_args
