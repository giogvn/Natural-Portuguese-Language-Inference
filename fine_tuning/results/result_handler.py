import matplotlib.pyplot as plt
import pandas as pd
import json, os, csv
from pathlib import Path
import numpy as np
import seaborn as sns
import hashlib


def get_file_sha256(content):
    """Return SHA256 hash of file content."""
    return hashlib.sha256(content.encode()).hexdigest()


def concatenate_json_files(root_directory: str, filename: str, output_file):
    result = {}

    def handle_non_standard_vals(val):
        """Convert non-standard JSON values."""
        if val == "NaN":
            return float("nan")
        return val

    # Recursively search for files with the specified name
    for dirpath, _, filenames in os.walk(root_directory):
        for fname in filenames:
            if fname == filename:
                file_path = os.path.join(dirpath, fname)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    parsed_json = json.loads(
                        content, parse_constant=handle_non_standard_vals
                    )
                    file_hash = get_file_sha256(content)
                    result[file_hash] = parsed_json

    # Write the consolidated content to another JSON file
    with open(output_file, "w", encoding="utf-8") as out_file:
        json.dump(result, out_file, indent=4)


def concatenate_csv_files(root_directory: str, filename: str, output_file):
    first_file = True

    with open(output_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)

        # Recursively search for files with the specified name
        for dirpath, _, filenames in os.walk(root_directory):
            for fname in filenames:
                if fname == filename:
                    file_path = os.path.join(dirpath, fname)
                    with open(file_path, "r", newline="") as infile:
                        reader = csv.reader(infile)

                        # Write header only from the first file
                        for row_idx, row in enumerate(reader):
                            if first_file or row_idx > 0:
                                writer.writerow(row)

                        # After the first file is processed, turn off the first_file flag
                        first_file = False


def plot_metrics_by_attribute(
    pipelines_dfs,
    attr_col,
    cols=["accuracy", "f1", "recall", "precision"],
):
    img_types_dfs = {
        pipeline: pipelines_dfs[pipeline][attr_col] for pipeline in pipelines_dfs.keys()
    }

    all_img_types = np.unique(
        np.concatenate([df.index for df in img_types_dfs.values()])
    )

    bar_width = 0.1
    n = len(img_types_dfs)

    for j, img in enumerate(all_img_types):
        fig, ax = plt.subplots()

        for i, (pipeline, img_type_df) in enumerate(img_types_dfs.items()):
            acc = img_type_df.loc[img, "accuracy"] if img in img_type_df.index else 0
            rec = img_type_df.loc[img, "recall"] if img in img_type_df.index else 0
            prec = img_type_df.loc[img, "precision"] if img in img_type_df.index else 0
            f1 = img_type_df.loc[img, "f1"] if img in img_type_df.index else 0

            acc_bar = ax.bar(i - bar_width / 4, acc, bar_width, color="b")
            rec_bar = ax.bar(i + bar_width / 4, rec, bar_width, color="r")
            prec_bar = ax.bar(i + bar_width / 4, prec, bar_width, color="g")
            f1_bar = ax.bar(i + bar_width, f1, bar_width, color="y")

            ax.bar_label(acc_bar, labels=[f"{acc:.5f}"], padding=3)
            ax.bar_label(rec_bar, labels=[f"{rec:.5f}"], padding=3)
            ax.bar_label(prec_bar, labels=[f"{prec:.5f}"], padding=3)
            ax.bar_label(f1_bar, labels=[f"{f1:.5f}"], padding=3)

        ax.set_xlabel("Model Name")
        ax.set_ylabel("Values")
        ax.set_title(f"Metrics for {img}")
        ax.set_xticks(range(n))
        ax.set_xticklabels(img_types_dfs.keys(), rotation=45)
        plt.legend(cols, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()
