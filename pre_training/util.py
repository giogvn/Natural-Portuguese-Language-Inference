from datasets import (
    load_dataset_builder,
    get_dataset_split_names,
    get_dataset_config_names,
)

import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import pandas as pd
import evaluate


def compute_metrics(pred: tuple, metric_name: str = "accuracy"):
    metric = evaluate.load(metric_name)
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


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


def xlmR_tokenizer(input: dict) -> list:
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    return tokenizer(
        input["premise"], input["hypothesis"], padding="max_length", truncation=True
    )


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
