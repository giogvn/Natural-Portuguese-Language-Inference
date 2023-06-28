from transformers import Trainer
from hydra import compose, initialize
import numpy as np
from util import HuggingFaceLoader
import torch

torch.cuda.empty_cache()
loader = HuggingFaceLoader()

initialize(config_path="config", version_base=None)
config = compose(config_name="main")

train_dataset = loader.load_train_dataset(config)
tokenizer = loader.get_tokenizer(config)
test_dataset = loader.load_test_dataset(config)
metric = loader.get_metric(config)
training_args = loader.get_training_args(config)
model = loader.get_model(config)

tokenized_train = train_dataset.map(tokenizer, batched=True)
tokenized_test = test_dataset.map(tokenizer, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

trainer.train()
