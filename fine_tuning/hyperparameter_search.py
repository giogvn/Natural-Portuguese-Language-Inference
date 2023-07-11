"""
This example is uses the official
huggingface transformers `hyperparameter_search` API.
"""
import os

from ray import tune
from ray.air.config import CheckpointConfig
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from transformers import Trainer

"""Uses the Population Based Training Algorithm (https://www.deepmind.com/blog/population-based-training-of-neural-networks) 
to search for the best hyperparameters of a model"""


def hyperparameter_opt(
    trainer: Trainer,
    gpus_per_trial: int = 1,
    smoke_test: bool = False,
    num_samples: int = 8,
):
    tune_config = {
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 32,
        "num_train_epochs": tune.choice([2, 3, 4, 5]),
        "max_steps": 1 if smoke_test else -1,  # Used for smoke test.
    }

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="eval_acc",
        mode="max",
        max_t=100,
        grace_period=10,
        reduction_factor=3,
        brackets=1,
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
        stop={"training_iteration": 1} if smoke_test else None,
        progress_reporter=reporter,
        local_dir="~/ray_results/",
        name="tune_transformer_pbt",
        log_to_file=True,
    )
