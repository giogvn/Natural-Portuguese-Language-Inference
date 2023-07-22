from util import HyperparameterTuningArguments, DataTrainingArguments, ModelArguments
import wandb, os
from transformers import TrainingArguments, Trainer, EvalPrediction
from datasets import load_dataset, load_metric, Dataset
import numpy as np
import os


class CrossPredictor:
    def __init__(
        self,
        logger,
        output_dir: str,
        args,
        trainer: Trainer,
        tokenizer,
        model_args: ModelArguments,
        training_args: TrainingArguments,
        train_dataset_name: str,
        overwrite_cache: bool,
        preprocess_func: callable,
        metrics_computer: callable,
    ):
        self.train_dataset_name = train_dataset_name
        self.output_dir = output_dir
        self.logger = logger
        self.preprocess_func = preprocess_func
        self.metrics_computer = metrics_computer
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.model_args = model_args
        self.data_args = args.data_args
        self.test_datasets = args.test_datasets
        self.train_dataset_name = train_dataset_name

    def do_cross_tests(self):
        for dataset, args in self.test_datasets.items():
            subsets = args["subsets"]
            data_args = args["data_args"]
            if len(subsets):
                for subset in subsets:
                    dataset = load_dataset(
                        dataset,
                        name=subset,
                        split="test" if (self.train_dataset_name == dataset) else None,
                        cache_dir=self.model_args.cache_dir,
                        use_auth_token=True if self.model_args.use_auth_token else None,
                    )
                    self._predict(dataset, data_args, subset=subset)
            else:
                self._predict(dataset, data_args)

    def _predict(
        self,
        predict_dataset: Dataset,
        data_args: DataTrainingArguments,
        subset: str = "",
    ):
        self.logger.info("*** Predict ***")
        if data_args.rename_colums:
            predict_dataset = predict_dataset.rename_columns(
                self.data_args.rename_columns
            )
        with self.training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            predict_dataset = predict_dataset.map(
                self.preprocess_func,
                batched=True,
                load_from_cache_file=not self.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
                remove_columns=predict_dataset.column_names,
            )
        predictions, labels, metrics = self.trainer.predict(
            predict_dataset, metric_key_prefix="predict"
        )

        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        self.trainer.log_metrics("predict", metrics)
        self.trainer.save_metrics("predict", metrics)

        predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(self.output_dir, subset + "_predictions.txt")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        label_list = data_args.label_names["predict_dataset"]
        if self.trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    if not type(list(label_list.keys())[0]) == type(item):
                        item = type(list(label_list.keys())[0])(item)
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")


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
        with wandb.init(config=config, resume=True):
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
