"""Run nested training-size experiments and save metrics, model, and figures."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split

from src.data import load_imdb
from src.model import build_model


def run(output: Path, sizes: list[int] | None = None) -> pd.DataFrame:
    train, test = load_imdb()
    sizes = sorted(set(sizes or [1_000, 5_000, 25_000]))
    if sizes[0] < 2 or sizes[-1] > len(train):
        raise ValueError(f"Training sizes must be between 2 and {len(train)}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    results = []
    full_model = None
    for size in sizes:
        if size == len(train):
            subset = train
        else:
            subset, _ = train_test_split(
                train, train_size=size, random_state=42, stratify=train["label"]
            )
        model = build_model()
        start = time.perf_counter()
        model.fit(subset["text"], subset["label"])
        fit_seconds = time.perf_counter() - start
        predicted = model.predict(test["text"])
        results.append({
            "training_reviews": size,
            "test_reviews": len(test),
            "accuracy": accuracy_score(test["label"], predicted),
            "precision": precision_score(test["label"], predicted, zero_division=0),
            "recall": recall_score(test["label"], predicted, zero_division=0),
            "f1": f1_score(test["label"], predicted, zero_division=0),
            "fit_seconds": fit_seconds,
        })
        print(f"{size:>6} train reviews | accuracy={results[-1]['accuracy']:.4f} "
              f"F1={results[-1]['f1']:.4f} fit={fit_seconds:.1f}s")
        if size == max(sizes):
            full_model = model
            report = classification_report(test["label"], predicted,
                                           target_names=["negative", "positive"], digits=4)
            (output / "classification_report.txt").write_text(report, encoding="utf-8")
            matrix = confusion_matrix(test["label"], predicted)
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            image = ax.imshow(matrix, cmap="Blues")
            ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"],
                   yticklabels=["Negative", "Positive"], xlabel="Predicted label",
                   ylabel="True label", title="IMDb test-set confusion matrix")
            for (row, col), value in __import__("numpy").ndenumerate(matrix):
                ax.text(col, row, f"{value:,}", ha="center", va="center", color="black")
            fig.colorbar(image, ax=ax, label="Reviews")
            fig.tight_layout()
            fig.savefig(output / "figures" / "confusion_matrix.png", dpi=180)
            plt.close(fig)
    frame = pd.DataFrame(results)
    frame.to_csv(output / "metrics.csv", index=False, float_format="%.6f")
    (output / "metrics.json").write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")
    dump(full_model, output / "sentiment_model.joblib")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--sizes", nargs="+", type=int, default=[1_000, 5_000, 25_000])
    args = parser.parse_args()
    run(args.output, args.sizes)


if __name__ == "__main__":
    main()
