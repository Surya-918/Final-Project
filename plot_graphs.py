import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Union

import matplotlib.pyplot as plt


def save_training_metrics(log_history, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "training_metrics.csv"
    graph_path = output_dir / "training_metrics.png"

    rows = []
    for entry in log_history:
        eval_loss = entry.get("eval_loss")
        rows.append(
            {
                "step": entry.get("step"),
                "epoch": entry.get("epoch"),
                "train_loss": entry.get("loss"),
                "eval_loss": eval_loss,
                "eval_perplexity": math.exp(eval_loss) if eval_loss is not None and eval_loss < 700 else None,
                "eval_next_token_accuracy": entry.get("eval_next_token_accuracy"),
            }
        )

    if not rows:
        raise ValueError("Trainer did not record any metrics; no graph can be generated.")

    with csv_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    train_rows = [row for row in rows if row["train_loss"] is not None]
    eval_rows = [row for row in rows if row["eval_loss"] is not None]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    if train_rows:
        axes[0].plot(
            [row["step"] for row in train_rows],
            [row["train_loss"] for row in train_rows],
            color="#3776ab",
            linewidth=2,
            label="Training loss",
        )
    if eval_rows:
        axes[0].plot(
            [row["step"] for row in eval_rows],
            [row["eval_loss"] for row in eval_rows],
            marker="o",
            color="#c87525",
            linewidth=2,
            label="Validation loss",
        )
    axes[0].set_title("Language-model loss")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Loss (lower is better)")
    axes[0].legend(frameon=False)

    perplexity_rows = [row for row in eval_rows if row["eval_perplexity"] is not None]
    if perplexity_rows:
        axes[1].plot(
            [row["epoch"] for row in perplexity_rows],
            [row["eval_perplexity"] for row in perplexity_rows],
            marker="o",
            color="#25834f",
            linewidth=2,
        )
    axes[1].set_title("Validation perplexity")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Perplexity (lower is better)")

    accuracy_rows = [
        row for row in eval_rows if row["eval_next_token_accuracy"] is not None
    ]
    if accuracy_rows:
        axes[2].plot(
            [row["epoch"] for row in accuracy_rows],
            [100 * row["eval_next_token_accuracy"] for row in accuracy_rows],
            marker="o",
            color="#8b5a2b",
            linewidth=2,
        )
    axes[2].set_title("Next-token accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Correct next tokens (%)")

    for axis in axes:
        axis.grid(True, color="#d8d8d8", alpha=0.55)
    figure.suptitle("Metrics recorded during this training run")
    figure.tight_layout()
    figure.savefig(graph_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return csv_path, graph_path


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate the graphs from a real GPT-2 training metrics CSV."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "trained model",
        help="Training output folder containing training_metrics.csv.",
    )
    args = parser.parse_args()

    csv_path = args.output_dir / "training_metrics.csv"
    if not csv_path.is_file():
        parser.error(f"No training metrics found at {csv_path}; run Pretrained_GPT-2.py first.")

    log_history = []
    with csv_path.open(newline="", encoding="utf-8") as metrics_file:
        for row in csv.DictReader(metrics_file):
            entry: Dict[str, Union[int, float]] = {"step": int(row["step"])}
            if row["epoch"]:
                entry["epoch"] = float(row["epoch"])
            if row["train_loss"]:
                entry["loss"] = float(row["train_loss"])
            if row["eval_loss"]:
                entry["eval_loss"] = float(row["eval_loss"])
            if row["eval_next_token_accuracy"]:
                entry["eval_next_token_accuracy"] = float(row["eval_next_token_accuracy"])
            log_history.append(entry)

    metrics_csv, metrics_graph = save_training_metrics(log_history, args.output_dir)
    print(f"Training metrics: {metrics_csv}")
    print(f"Training graph: {metrics_graph}")


if __name__ == "__main__":
    main()