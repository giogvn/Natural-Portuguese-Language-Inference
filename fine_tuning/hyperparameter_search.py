from util import HyperparameterTuningArguments
import wandb
from transformers import TrainingArguments, Trainer

class HyperparameterTuner:
    def __init__(self, args: HyperparameterTuningArguments, 
                 train_dataset, eval_dataset, metrics_computer: callable, 
                 collate_func: callable, model_getter: callable):
        self.config = args.sweep_config
        self.training_args = args.training_args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.metrics_computer = metrics_computer
        self.model_getter = model_getter
        self.collate_func = collate_func
    
    def train(self, config = None):
        with wandb.init(config=config):
            self.config = wandb.config

            training_args = TrainingArguments(
                output_dir=self.training_args.output_dir
                report_to=self.training_args.report_to,  # Turn on Weights & Biases logging
                num_train_epochs=self.config.epochs,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=16,
                save_strategy=self.training_args.save_strategy,
                evaluation_strategy=self.training_args.evaluation_strategy,
                logging_strategy=self.training_args.logging_strategy,
                load_best_model_at_end=self.training_args.load_best_model_at_end,
                remove_unused_columns=self.training_args.remove_unused_columns,
                fp16=self.training_args.fp16
        )

            trainer = Trainer(
                model_init=self.model_getter,
                args=training_args,
                data_collator=self.collate_func,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                compute_metrics=self.metrics_computer
            )

            trainer.train()

