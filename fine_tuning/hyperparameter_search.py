from util import HyperparameterTuningArguments, DataTrainingArguments, ModelArguments
import wandb, os
from transformers import TrainingArguments, Trainer, EvalPrediction
from datasets import load_dataset, load_metric, Dataset
import numpy as np
import os, shutil, json


class CrossPredictor:
    def __init__(
        self,
        logger,
        output_base_dir: str,
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
        self.output_base_dir = output_base_dir
        self.logger = logger
        self.preprocess_func = preprocess_func
        self.metrics_computer = metrics_computer
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.model_args = model_args
        self.test_datasets = args.datasets
        self.train_dataset_name = train_dataset_name
        self.overwrite_cache = overwrite_cache

    def do_cross_tests(self):
        for dataset, args in self.test_datasets.items():
            data_args = DataTrainingArguments(**args["data_args"])
            if "subsets" in args:
                subsets = args["subsets"]
                for subset in subsets:
                    test_dataset = load_dataset(
                        dataset,
                        name=subset,
                        split="test",
                        cache_dir=self.model_args.cache_dir,
                        use_auth_token=True if self.model_args.use_auth_token else None,
                    )
                    self._predict(test_dataset, data_args, subset=subset)
            else:
                test_dataset = load_dataset(
                    dataset,
                    split="test",
                    cache_dir=self.model_args.cache_dir,
                    use_auth_token=True if self.model_args.use_auth_token else None,
                )
                self._predict(test_dataset, data_args)

    def _predict(
        self,
        predict_dataset: Dataset,
        data_args: DataTrainingArguments,
        subset: str = "",
    ):
        self.logger.info("*** Predict ***")
        if hasattr(data_args, "rename_columns"):
            predict_dataset = predict_dataset.rename_columns(data_args.rename_columns)
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

        output_dir = os.path.join(
            self.output_base_dir, data_args.dataset_name + "_" + subset
        )
        if not os.path.exists(self.output_base_dir):
            os.makedirs(self.output_base_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        t_args = self.training_args.to_dict()
        t_args["output_dir"] = output_dir
        training_args = TrainingArguments(**t_args)
        trainer = Trainer(
            model=self.trainer.model,
            args=training_args,
            train_dataset=self.trainer.train_dataset
            if training_args.do_train
            else None,
            eval_dataset=self.trainer.eval_dataset if training_args.do_eval else None,
            compute_metrics=self.trainer.compute_metrics,
            tokenizer=self.tokenizer,
            data_collator=self.trainer.data_collator,
        )
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
        output_predict_file = os.path.join(output_dir, subset + "_predictions.txt")

        label_list = data_args.label_names["predict_dataset"]
        if trainer.is_world_process_zero():
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
        self.curr_eval_accuracy = float("-inf")

    def train(self, config=None):
        run = wandb.init(config=config, resume=True)
        with run:
            self.config = wandb.config

            training_args = TrainingArguments(
                output_dir=self.training_args.output_dir,
                report_to=self.training_args.report_to,
                num_train_epochs=self.config.epochs,
                weight_decay=self.config.weight_decay,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=16,
                save_strategy=self.training_args.save_strategy,
                evaluation_strategy=self.training_args.evaluation_strategy,
                logging_strategy=self.training_args.logging_strategy,
                load_best_model_at_end=self.training_args.load_best_model_at_end,
                metric_for_best_model=self.optm_metric,
                greater_is_better=self.goal == "maximize",
                remove_unused_columns=self.training_args.remove_unused_columns,
                fp16=self.training_args.fp16,
                eval_steps=self.training_args.eval_steps,
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
            eval_results = trainer.evaluate()
            run_eval_accuracy = eval_results["eval_accuracy"]
            if run_eval_accuracy > self.curr_eval_accuracy:
                if trainer.is_world_process_zero():
                    run_id = run.id
                    save_dir = f"./best_model_{run_id}"
                    trainer.model.save_pretrained(save_dir)
                    artifact = wandb.Artifact("model", type="model")
                    artifact.add_dir(save_dir)
                    run.log_artifact(artifact)

                    file_path = save_dir + "/eval_results.json"
                    with open(file_path, "w") as outfile:
                        json.dump(eval_results, outfile)

                    artifact.add_file(file_path)
                    shutil.rmtree(save_dir)
                    self.curr_eval_accuracy = run_eval_accuracy
