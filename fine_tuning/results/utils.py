import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import hashlib
import fnmatch
import os
import json

RESULTS_BASE_PATH = "../translations"
COLOR_CODES = {"teal": "#00B7B7", "coral": "#FF7F50", "gold": "#FFD700"}
COLORS = {
    "ASSIN": COLOR_CODES["teal"],
    "ASSIN-2": COLOR_CODES["coral"],
    "PLUE/MNLI": COLOR_CODES["gold"],
}

MODELS_MAP = {
    "xlm-roberta-base": "XLM-RoBERTa (Base)",
    "neuralmind/bert-large-portuguese-cased": "BERTimbau (Large)",
}

DATASETS_MAPS = {"dlb/plue": "PLUE/MNLI", "assin": "ASSIN", "assin2": "ASSIN-2"}
RESULTS_DF_COLS = [
    "model_name",
    "test_dataset",
    "train_dataset",
    "accuracy",
    "f1",
    "precision",
    "recall",
    "NONE_accuracy",
    "NONE_f1",
    "NONE_precision",
    "NONE_recall",
    "ENTAILMENT_accuracy",
    "ENTAILMENT_f1",
    "ENTAILMENT_precision",
    "ENTAILMENT_recall",
    "PARAPHRASE_accuracy",
    "PARAPHRASE_f1",
    "PARAPHRASE_precision",
    "PARAPHRASE_recall",
    "neutral_accuracy",
    "neutral_f1",
    "neutral_precision",
    "neutral_recall",
    "contradiction_accuracy",
    "contradiction_f1",
    "contradiction_precision",
    "contradiction_recall",
    "entailment_accuracy",
    "entailment_f1",
    "entailment_precision",
    "entailment_recall",
]
PRECISION = ".2f"
LABEL_SIZE = 15
VALUE_SIZE = 12
TITLE_SIZE = 20


def find_files(base: str, pattern: str):
    """Returns a list of paths to files matching the given pattern"""

    result = []
    for root, dirs, files in os.walk(base):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def mount_metrics_dataframe(
    base_path: str = RESULTS_BASE_PATH, file_name: str = "predict_results.json"
) -> pd.DataFrame:
    results_files = find_files(base_path, file_name)
    for index, f in enumerate(results_files):
        with open(f, "r") as file:
            results_files[index] = json.load(file)

    results_dfs = [pd.DataFrame([f]) for f in results_files]

    out = pd.DataFrame()
    out = pd.concat(results_dfs, axis=0, ignore_index=True, sort=False)
    return out


def change_font_color_and_weight(colors: dict, ticklabels: list, bold: bool):
    for label in ticklabels:
        color = [
            pattern
            for pattern in colors.keys()
            if pattern.lower() in (label.get_text()).lower()
        ]
        color = colors[max(color, key=len)] if len(color) else "black"
        label.set_color(color)
        if bold:
            label.set_weight("bold")


def plot_metrics_results(
    metrics_df: pd.DataFrame,
    model_name: str,
    metric: str,
    precision: str = PRECISION,
    colors: dict = COLORS,
    bold: bool = True,
    label_size: int = LABEL_SIZE,
    value_size: int = VALUE_SIZE,
    title_size: int = TITLE_SIZE,
):
    plot = metrics_df[metrics_df["model_name"] == model_name].pivot(
        index="test_dataset", columns="train_dataset", values=metric
    )
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        plot,
        annot=True,
        cmap="Blues",
        fmt=precision,
        vmin=0.5,
        vmax=1,
        annot_kws={"size": value_size},
    )
    plt.title(f"{metric[0].upper() + metric[1:]} for {model_name}", fontsize=title_size)
    change_font_color_and_weight(colors, ax.get_xticklabels(), bold)
    change_font_color_and_weight(colors, ax.get_yticklabels(), bold)
    ax.yaxis.set_tick_params(labelsize=label_size)
    ax.xaxis.set_tick_params(labelsize=label_size)
    plt.show()


def plot_models_benchmark_results(
    metrics_df: pd.DataFrame,
    metric: str,
    precision: str = PRECISION,
    colors: dict = COLORS,
    bold: bool = True,
    label_size: int = LABEL_SIZE,
    value_size: int = VALUE_SIZE,
    title_size: int = TITLE_SIZE,
):
    plot = metrics_df.pivot(index="test_dataset", columns="model", values=metric)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        plot,
        annot=True,
        cmap="Blues",
        fmt=precision,
        vmin=0.5,
        vmax=1,
        annot_kws={"size": value_size},
    )
    plt.title(f"{metric.upper()}", fontsize=title_size)
    plt.xticks(rotation=90)

    change_font_color_and_weight(colors, ax.get_xticklabels(), bold)
    change_font_color_and_weight(colors, ax.get_yticklabels(), bold)
    ax.yaxis.set_tick_params(labelsize=label_size)
    ax.xaxis.set_tick_params(labelsize=label_size)

    plt.show()
