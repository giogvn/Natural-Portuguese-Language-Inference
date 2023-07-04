"""
This example is uses the official
huggingface transformers `hyperparameter_search` API.
"""
import os

from ray import tune
from ray.air.config import CheckpointConfig
from ray.tune import CLIReporter
from ray.tune.examples.pbt_transformers.utils import (
    download_data,
    build_compute_metrics_fn,
)
from ray.tune.schedulers import PopulationBasedTraining
from transformers import (
    glue_tasks_num_labels,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    GlueDataset,
    GlueDataTrainingArguments,
    TrainingArguments,
)

"""Uses the Population Based Training Algorithm (https://www.deepmind.com/blog/population-based-training-of-neural-networks) 
to search for the best hyperparameters of a model"""


class PBTHyperparameterOptimizer:

    def __init__(trainer: Trainer, gpus_per_trial: int = 1, smoke_test: bool = False)
        self.trainer = trainer
        self.smoke_test = smoke_test


        tune_config = {
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 32,
            "num_train_epochs": tune.choice([2, 3, 4, 5]),
            "max_steps": 1 if smoke_test else -1,  # Used for smoke test.
        }

        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="eval_acc",
            mode="max",
            perturbation_interval=1,
            hyperparam_mutations={
                "weight_decay": tune.uniform(0.0, 0.3),
                "learning_rate": tune.uniform(1e-5, 5e-5),
                "per_device_train_batch_size": [16, 32, 64],
            },
        )

        reporter = CLIReporter(
            parameter_columns={
                "weight_decay": "w_decay",
                "learning_rate": "lr",
                "per_device_train_batch_size": "train_bs/gpu",
                "num_train_epochs": "num_epochs",
            },
            metric_columns=["eval_acc", "eval_loss", "epoch", "training_iteration"],
        )

        trainer.hyperparameter_search(
            hp_space=lambda _: tune_config,
            backend="ray",
            n_trials=num_samples,
            resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
            scheduler=scheduler,
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="training_iteration",
            ),
            stop={"training_iteration": 1} if smoke_test else None,
            progress_reporter=reporter,
            local_dir="~/ray_results/",
            name="tune_transformer_pbt",
            log_to_file=True,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    args, _ = parser.parse_known_args()

    if args.smoke_test:
        tune_transformer(num_samples=1, gpus_per_trial=0, smoke_test=True)
    else:
        # You can change the number of GPUs here:
        tune_transformer(num_samples=8, gpus_per_trial=1)
