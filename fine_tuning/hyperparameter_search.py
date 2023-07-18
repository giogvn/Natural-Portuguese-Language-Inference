from util import HyperparameterTuningArguments, DataTrainingArguments, ModelArguments
import wandb, os
from transformers import TrainingArguments, Trainer, EvalPrediction
from datasets import load_dataset, load_metric
import numpy as np


class CrossPredictor:
    def __init__(
        self,
        args,
        trainer: Trainer,
        tokenizer,
        data_args: DataTrainingArguments,
        model_args: ModelArguments,
        train_args: TrainingArguments,
        preprocess_func: callable,
        metrics_computer: callable,
    ):
        self.preprocess_func = preprocess_func
        self.metrics_computer = metrics_computer
        self.train_args = train_args
        self.model_args = model_args
        self.data_args = data_args
        self.args = args
        self.config = args.config
        self.trainer = trainer
        self.tokenizer = tokenizer

    def predict(self, config=None):
        with wandb.init(config=config):
            self.config = wandb.config

            predict_dataset = load_dataset(
                config.dataset_name,
                name=config.subset,
                split="test"
                if (self.data_args.dataset_name == self.config.dataset_name)
                else None,
                cache_dir=self.model_args.cache_dir,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )

            if self.args.data_args.rename_colums:
                predict_dataset = predict_dataset.rename_columns(
                    self.args.data_args.rename_columns
                )
            with self.train_args.main_process_first(
                desc="prediction dataset map pre-processing"
            ):
                predict_dataset = predict_dataset.map(
                    self.preprocess_func,
                    batched=True,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                    remove_columns=predict_dataset.column_names,
                )
                predictions, labels, metrics = self.trainer.predict(
                    predict_dataset, metric_key_prefix="predict"
                )

                predictions = np.argmax(predictions, axis=1)
                return self.metrics_computer(
                    EvalPrediction(predictions=predictions, label_ids=labels)
                )


class HyperparameterTuner:
    def __init__(
        self,
        args: HyperparameterTuningArguments,
        train_dataset,
        eval_dataset,
        metrics_computer: callable,
        collate_func: callable,
        model_getter: callable,
    ):
        self.config = args.sweep_config
        self.training_args = args.training_args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.metrics_computer = metrics_computer
        self.model_getter = model_getter
        self.collate_func = collate_func

    def train(self, config=None):
        with wandb.init(config=config):
            self.config = wandb.config

            training_args = TrainingArguments(
                output_dir=self.training_args.output_dir,
                report_to=self.training_args.report_to,  # Turn on Weights & Biases logging
                num_train_epochs=self.config.epochs,
                # learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=16,
                save_strategy=self.training_args.save_strategy,
                evaluation_strategy=self.training_args.evaluation_strategy,
                logging_strategy=self.training_args.logging_strategy,
                load_best_model_at_end=self.training_args.load_best_model_at_end,
                remove_unused_columns=self.training_args.remove_unused_columns,
                fp16=self.training_args.fp16,
            )

            trainer = Trainer(
                model_init=self.model_getter,
                args=training_args,
                data_collator=self.collate_func,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                compute_metrics=self.metrics_computer,
            )

            trainer.train()
