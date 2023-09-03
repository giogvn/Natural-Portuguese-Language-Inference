from util import (
    HyperparameterTuningArguments,
    DataTrainingArguments,
    ModelArguments,
    Predictor,
)
import wandb, os, csv
from transformers import TrainingArguments, Trainer, EvalPrediction
from datasets import load_dataset, load_metric, Dataset
import numpy as np
import os, shutil, json
from functools import partial


class CrossPredictor:
    def __init__(
        self,
        logger,
        output_base_dir: str,
        test_datasets: dict,
        trainer: Trainer,
        tokenizer,
        model_args: ModelArguments,
        training_args: TrainingArguments,
        train_dataset_name: str,
        overwrite_cache: bool,
        preprocess_func: callable,
        metrics_computer: callable,
        train_class: dict = None,
    ):
        self.train_dataset_name = train_dataset_name
        self.output_base_dir = output_base_dir
        self.logger = logger
        self.preprocess_func = preprocess_func
        self.metrics_computer = metrics_computer
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.model_args = model_args
        self.test_datasets = test_datasets
        self.train_dataset_name = train_dataset_name
        self.overwrite_cache = overwrite_cache
        self.train_class = train_class

    def do_cross_tests(self):
        test_datasets = {
            t_ds: args
            for t_ds, args in self.test_datasets.items()
            if args["do_cross_tests"]
        }
        for dataset, args in test_datasets.items():
            data_args = DataTrainingArguments(**args["data_args"])
            do_label_translation = args["do_label_translation"]
            by_class_metric = args["by_class_metric"]
            if "subsets" in args:
                subsets = args["subsets"]
                for subset in subsets:
                    test_dataset = load_dataset(
                        dataset,
                        name=subset,
                        split=data_args.test_dataset_split,
                        cache_dir=self.model_args.cache_dir,
                        use_auth_token=True if self.model_args.use_auth_token else None,
                    )
                    if data_args.positive_filter != None:
                        test_dataset = self._filter_rows(
                            test_dataset, data_args.positive_filter
                        )

                    self._predict(
                        dataset,
                        test_dataset,
                        data_args,
                        do_label_translation,
                        subset=subset,
                        by_class_metric=by_class_metric,
                    )
            else:
                test_dataset = load_dataset(
                    dataset,
                    split=data_args.test_dataset_split,
                    cache_dir=self.model_args.cache_dir,
                    use_auth_token=True if self.model_args.use_auth_token else None,
                )
                if data_args.positive_filter != None:
                    test_dataset = self._filter_rows(
                        test_dataset, data_args.positive_filter
                    )
                self._predict(
                    dataset,
                    test_dataset,
                    data_args,
                    do_label_translation,
                    by_class_metric=by_class_metric,
                )

    def _filter_rows(self, dataset: Dataset, conditions: dict, positive: bool = True):
        for column, values in conditions.items():
            filters = [
                (values[i], values[i + 1])
                for i in range(0, len(values), 2)
                if i + 1 < len(values)
            ]
            if filters[-1][1] != values[-1]:
                filters.append((values[-1], None))
            for f in filters:
                val1, val2 = f[0], f[1]
                if positive and val2 != None:
                    dataset = dataset.filter(
                        lambda row: row[column] == val1 or row[column] == val2
                    )
                elif val2 == None:
                    dataset = dataset.filter(lambda row: row[column] == val1)
        return dataset

    def _predict(
        self,
        predict_dataset_name: str,
        predict_dataset: Dataset,
        data_args: DataTrainingArguments,
        do_label_translation: bool = False,
        subset: str = "",
        by_class_metric: bool = False,
    ):
        self.logger.info("*** Predict ***")
        if data_args.rename_columns != None:
            predict_dataset = predict_dataset.rename_columns(data_args.rename_columns)

        output_dir = os.path.join(
            self.output_base_dir, data_args.dataset_name + "_" + subset
        )
        if not os.path.exists(self.output_base_dir):
            os.makedirs(self.output_base_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        t_args = self.training_args.to_dict()
        t_args["output_dir"] = output_dir
        class_label = {
            v: k for k, v in data_args.label_names["predict_dataset"].items()
        }
        training_args = TrainingArguments(**t_args)
        trainer = Trainer(
            model=self.trainer.model,
            args=training_args,
            train_dataset=self.trainer.train_dataset
            if training_args.do_train
            else None,
            eval_dataset=self.trainer.eval_dataset if training_args.do_eval else None,
            compute_metrics=self.trainer.compute_metrics
            if not True
            else partial(
                self.custom_compute_metrics,
                modify_labels_and_preds=data_args.modify_labels_and_preds,
                by_class_metric=by_class_metric,
                test_label=class_label,
                subset=subset,
                model_name=self.model_args.model_name_or_path,
                train_dataset=self.train_dataset_name,
                test_dataset=predict_dataset_name,
            ),
            tokenizer=self.tokenizer,
            data_collator=self.trainer.data_collator,
        )

        # HERE IS WHERE THE TRANSLATION COMES IN

        if do_label_translation:
            predictor = Predictor(
                predict_dataset_name,
                self.train_dataset_name,
                trainer,
                self.trainer.model,
                self.tokenizer,
                self.train_class,
                {v: k for k, v in data_args.label_names["predict_dataset"].items()},
                by_class_metric=by_class_metric,
                model_name=self.model_args.model_name_or_path,
                test_set_subset=subset,
            )

            predictions, labels, metrics = predictor.predict(predict_dataset)

        # HERE IS WHERE THE TRANSLATION COMES OUT

        else:
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
            predictions, labels, metrics = trainer.predict(
                predict_dataset, metric_key_prefix="predict"
            )
            predictions = np.argmax(predictions, axis=1)

        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        output_predict_file = os.path.join(output_dir, "predictions.csv")

        label_list = data_args.label_names["predict_dataset"]
        model_name = self.model_args.model_name_or_path
        test_dataset = predict_dataset_name
        train_dataset = self.train_dataset_name
        columns = [
            "model_name",
            "test_dataset",
            "train_dataset",
            "index",
            "prediction",
            "label",
        ]
        data = {k: [] for k in columns}
        data["model_name"] = [model_name] * len(predictions)
        data["test_dataset"] = [test_dataset] * len(predictions)
        data["train_dataset"] = [train_dataset] * len(predictions)
        data["index"] = list(range(len(predictions)))
        if trainer.is_world_process_zero():
            for index, item in enumerate(predictions):
                if not type(list(label_list.keys())[0]) == type(item):
                    item = type(list(label_list.keys())[0])(item)
                item = label_list[item]
                label = label_list[int(labels[index])]
                data["prediction"].append(item)
                data["label"].append(label)

            rows = zip(*data.values())
            with open(output_predict_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(columns)
                writer.writerows(rows)

    def custom_compute_metrics(
        self,
        p: EvalPrediction,
        modify_labels_and_preds: dict = {},
        by_class_metric: bool = False,
        test_label: dict = {},
        model_name: str = "",
        train_dataset: str = "",
        test_dataset: str = "",
        subset: str = "",
    ):
        metrics = dict()

        accuracy_metric = load_metric("accuracy")
        precision_metric = load_metric("precision")
        recall_metric = load_metric("recall")
        f1_metric = load_metric("f1")

        labels = p.label_ids
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(logits, axis=-1)

        if modify_labels_and_preds:
            if "preds" in modify_labels_and_preds:
                for old_val, new_val in modify_labels_and_preds.preds.items():
                    preds[preds == old_val] = new_val
            if "labels" in modify_labels_and_preds:
                for old_val, new_val in modify_labels_and_preds.labels.items():
                    labels[labels == old_val] = new_val

        if by_class_metric:
            for test_class, label in test_label.items():
                indices = [
                    i for i, true_label in enumerate(labels) if true_label == label
                ]
                subset_true = [labels[i] for i in indices]
                subset_pred = [preds[i] for i in indices]

                args = {
                    "predictions": subset_pred,
                    "references": subset_true,
                    "average": "macro",
                }

                metric_name = "accuracy"
                metrics[test_class + "_" + metric_name] = accuracy_metric.compute(
                    predictions=subset_pred, references=subset_true
                )[metric_name]

                metric_name = "recall"
                metrics[test_class + "_" + metric_name] = recall_metric.compute(**args)[
                    metric_name
                ]

                metric_name = "precision"
                metrics[test_class + "_" + metric_name] = precision_metric.compute(
                    **args
                )[metric_name]

                metric_name = "f1"
                metrics[test_class + "_" + metric_name] = f1_metric.compute(**args)[
                    metric_name
                ]

        metrics["model_name"] = model_name
        metrics["train_dataset"] = train_dataset
        metrics["test_dataset"] = test_dataset
        metrics["test_subset"] = subset

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


class HyperparameterTuner:
    def __init__(
        self,
        args: HyperparameterTuningArguments,
        training_args: TrainingArguments,
        train_dataset,
        eval_dataset,
        metrics_computer: callable,
        collate_func: callable,
        model_getter: callable,
    ):
        self.project_path = args.project_path
        self.config = args.sweep_config
        self.optm_metric = self.config.metric.name
        self.goal = self.config.metric.goal
        self.training_args = training_args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.metrics_computer = metrics_computer
        self.model_getter = model_getter
        self.collate_func = collate_func
        self.best_run = None
        self.curr_eval_accuracy = float("-inf")
        self.curr_eval_loss = float("inf")

    def mount_training_args(self):
        out = self.training_args.to_dict().copy()

        for key, value in self.config.items():
            out[key] = value

        return TrainingArguments(**out)

    def train(self, config=None):
        run = wandb.init(config=config, resume=True)
        with run:
            self.config = wandb.config

            training_args = self.mount_training_args()

            """training_args = TrainingArguments(
                output_dir=self.training_args.output_dir,
                report_to=self.training_args.report_to,
                num_train_epochs=self.config.epochs,
                # weight_decay=self.config.weight_decay,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=16,
                evaluation_strategy=self.training_args.evaluation_strategy,
                logging_strategy=self.training_args.logging_strategy,
                metric_for_best_model=self.training_args.metric_for_best_model,
                greater_is_better=self.training_args.greater_is_better,
                remove_unused_columns=self.training_args.remove_unused_columns,
                fp16=self.training_args.fp16,
                eval_steps=self.training_args.eval_steps,
                save_total_limit=self.training_args.save_total_limit,
                save_strategy=self.training_args.save_strategy,
                save_steps=self.training_args.save_steps,
                learning_rate=self.config.learning_rate,
            )"""

            trainer = Trainer(
                model_init=self.model_getter,
                args=training_args,
                data_collator=self.collate_func,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                compute_metrics=self.metrics_computer,
            )

            trainer.train()
            trainer._load_best_model()
            eval_results = trainer.evaluate()
            run_eval_accuracy = eval_results["eval_accuracy"]
            run_eval_loss = eval_results["eval_loss"]
            print(f"Best Model Yet Eval accuracy: {run_eval_accuracy}")
            if run_eval_accuracy > self.curr_eval_accuracy or (
                run_eval_accuracy == self.curr_eval_accuracy
                and run_eval_loss < self.curr_eval_loss
            ):
                if trainer.is_world_process_zero():
                    artifact = wandb.Artifact("model", type="model")
                    artifact.add_dir(self.training_args.output_dir)
                    run.log_artifact(artifact)
                    self.curr_eval_accuracy = run_eval_accuracy
                    self.curr_eval_loss = run_eval_loss
