import datasets as ds

import numpy as np
import hydra
from transformers import (
    XLMRobertaTokenizer,
    TrainingArguments,
    XLMRobertaForSequenceClassification,
)
from omegaconf import DictConfig
import pandas as pd
import evaluate


class HuggingFaceLoader:
    def load_train_dataset(
        self,
        config: DictConfig,
    ):
        split = "train"
        dataset_name = config.dataset.name[0]
        subset = config.dataset.subset[0]
        return ds.load_dataset(dataset_name, subset, split=split)

    def load_test_dataset(
        self,
        config: DictConfig,
    ):
        split = "test"
        dataset_name = config.dataset.name[0]
        subset = config.dataset.subset[0]
        return ds.load_dataset(dataset_name, subset, split=split)

    def get_model(self, config: DictConfig):
        model_name = config.model.name[0]
        model_type = config.model.type[0]
        num_labels = int(config.dataset.num_labels[0])

        if model_type == "XLMRobertaForSequenceClassification":
            return XLMRobertaForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )

    def get_tokenizer(self, config: DictConfig):
        tokenizer = config.model.tokenizer[0]
        if tokenizer == "xlmR_tokenizer":
            return self.xlmR_tokenizer

    def get_metric(self, config: DictConfig):
        metric_name = config.hyperparameter.metric[0]
        return evaluate.load(metric_name)

    def get_training_args(self, config: DictConfig):
        output_dir = config.output.output_dir[0]
        eval_strat = config.hyperparameter.evaluation_strategy[0]
        return TrainingArguments(output_dir=output_dir, evaluation_strategy=eval_strat)

    def xlmR_tokenizer(self, input: dict) -> list:
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        return tokenizer(
            input["premise"], input["hypothesis"], padding="max_length", truncation=True
        )


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
