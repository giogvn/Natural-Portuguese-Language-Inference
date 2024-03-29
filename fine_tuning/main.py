#!/usr/bin/env python
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
""" Finetuning multi-lingual models on XNLI (e.g. Bert, DistilBERT, XLM).
    Adapted from `examples/text-classification/run_glue.py`"""

import logging
import os
import random
import sys
import datasets
import torch
import numpy as np
from datasets import load_dataset, load_metric, concatenate_datasets, Dataset
import transformers
import wandb
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from util import (
    ModelArguments,
    DataTrainingArguments,
    HyperparameterTuningArguments,
    CrossTestsArguments,
    HuggingFaceLoader,
    WAndBLoader,
    PreprocessFunctions,
)
from hydra import compose, initialize
from hyperparameter_search import HyperparameterTuner, CrossPredictor
from collections import Counter

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.31.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

logger = logging.getLogger(__name__)


def main(m_args: dict, d_args: dict, t_args: dict, h_args: dict, c_args: dict):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    model_args = ModelArguments(**m_args)
    data_args = DataTrainingArguments(**d_args)
    training_args = TrainingArguments(**t_args)
    hyperparameter_tuning_args = HyperparameterTuningArguments(**h_args)
    cross_tests_args = CrossTestsArguments(**c_args)

    # parser = HfArgumentParser(
    #    (ModelArguments, DataTrainingArguments, TrainingArguments)
    # )

    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_xnli", model_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Downloading and loading xnli dataset from the hub.

    def get_train_dataset(data_args):
        if data_args.subset is None:
            train_dataset = load_dataset(
                data_args.dataset_name,
                split=data_args.train_split,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            train_dataset = load_dataset(
                data_args.dataset_name,
                name=data_args.subset,
                split=data_args.train_split,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        if data_args.rename_columns is not None:
            if training_args.do_train:
                train_dataset = train_dataset.rename_columns(data_args.rename_columns)
        return train_dataset

    train_dataset = get_train_dataset(data_args)
    label_list = data_args.label_names["train_dataset"]

    eval_dataset = load_dataset(
        data_args.dataset_name,
        name=data_args.subset,
        split=data_args.eval_split,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    label_list = data_args.label_names["eval_dataset"]

    predict_dataset = load_dataset(
        data_args.dataset_name,
        name=data_args.subset,
        split=data_args.test_split,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    label_list = data_args.label_names["predict_dataset"]

    if data_args.rename_columns is not None:
        if training_args.do_eval:
            eval_dataset = eval_dataset.rename_columns(data_args.rename_columns)
        if training_args.do_predict:
            predict_dataset = predict_dataset.rename_columns(data_args.rename_columns)

    # Labels
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label={str(i): label for i, label in enumerate(label_list)},
        label2id={label: i for i, label in enumerate(label_list)},
        finetuning_task="xnli",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    def get_model():
        return AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )

    model = get_model()

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    preprocess_function = PreprocessFunctions(
        model_args.model_name_or_path, tokenizer, padding, data_args
    )
    preprocess_function = preprocess_function.get_preprocess_function()

    def select_balanced_subset(dataset, n, rows_to_ignore: list = []):
        if len(rows_to_ignore):
            dataset = dataset.filter(
                lambda example: example["idx"] not in rows_to_ignore
            )
        labels_count = Counter(dataset["label"])
        max_samples = list(labels_count.values())
        max_samples.append(n // len(labels_count))
        max_samples = min(max_samples)
        selected_rows = []
        selected_indexes = []
        for label in labels_count.keys():
            label_data = dataset.filter(lambda row: row["label"] == label)
            label_data = label_data.shuffle()[:max_samples]
            selected_indexes.extend(label_data["idx"])
            selected_rows.append(label_data)

        selected_rows = [Dataset.from_dict(row) for row in selected_rows]

        return concatenate_datasets(selected_rows), selected_indexes

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            max_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset, _ = select_balanced_subset(train_dataset, max_samples)
        if hyperparameter_tuning_args.max_samples is not None:
            max_samples = min(
                len(train_dataset), hyperparameter_tuning_args.max_samples
            )
            (
                hyperparameter_tuning_dataset,
                _,
            ) = select_balanced_subset(train_dataset, max_samples)
        else:
            hyperparameter_tuning_dataset = train_dataset

        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
                remove_columns=train_dataset.column_names,
            )
        if hyperparameter_tuning_dataset != train_dataset:
            with training_args.main_process_first(
                desc="train dataset map pre-processing"
            ):
                hyperparameter_tuning_dataset = hyperparameter_tuning_dataset.map(
                    preprocess_function,
                    batched=True,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on hyperparameter tuning dataset dataset",
                    remove_columns=hyperparameter_tuning_dataset.column_names,
                )
        else:
            hyperparameter_tuning_dataset = train_dataset
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            if data_args.max_eval_samples is not None:
                eval_dataset, eval_rows = select_balanced_subset(
                    eval_dataset, max_eval_samples
                )
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
                remove_columns=eval_dataset.column_names,
            )

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))

        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
                remove_columns=predict_dataset.column_names,
            )

    # Get the metric function

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.

    def compute_metrics(p: EvalPrediction):
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

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    # Hyperparameter Tuning
    if hyperparameter_tuning_args.do_hyperparameter_tuning:
        sweep_id = WAndBLoader.get_sweep_id(hyperparameter_tuning_args)
        hyperparameter_tuner = HyperparameterTuner(
            hyperparameter_tuning_args,
            training_args,
            hyperparameter_tuning_dataset,
            eval_dataset,
            compute_metrics,
            data_collator,
            get_model,
        )
        wandb.agent(sweep_id, hyperparameter_tuner.train)

    if model_args.load_fine_tuned:
        model = WAndBLoader.get_best_model(model_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Cross Tests
    if cross_tests_args.do_cross_tests:
        cross_predictor = CrossPredictor(
            logger,
            cross_tests_args.output_dir,
            cross_tests_args.datasets,
            trainer,
            tokenizer,
            model_args,
            training_args,
            data_args.dataset_name,
            data_args.overwrite_cache,
            preprocess_function,
            compute_metrics,
            train_class=label_list,
        )
        cross_predictor.do_cross_tests()

    """
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(
            predict_dataset, metric_key_prefix="predict"
        )

        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    if not type(list(label_list.keys())[0]) == type(item):
                        item = type(list(label_list.keys())[0])(item)
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")"""


if __name__ == "__main__":
    initialize(config_path="config/assin2_config", version_base=None)
    config = compose(config_name="main")
    loader = HuggingFaceLoader(config)
    m_args = loader.get_model_args()
    d_args = loader.get_data_training_args()
    t_args = loader.get_training_args()
    h_args = loader.get_hyperparameter_tuning_args()
    c_args = loader.get_cross_tests_args()
    main(m_args, d_args, t_args, h_args, c_args)
