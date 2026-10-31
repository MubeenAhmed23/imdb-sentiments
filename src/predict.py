"""Predict sentiment for user-provided movie reviews."""
import argparse
from pathlib import Path

from joblib import load


def predict(text: str, model_path: Path = Path("results/sentiment_model.joblib")) -> str:
    if not text.strip():
        raise ValueError("Review text cannot be empty")
    model = load(model_path)
    label = int(model.predict([text])[0])
    return "positive" if label == 1 else "negative"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", help="Movie review text to classify")
    parser.add_argument("--model", type=Path, default=Path("results/sentiment_model.joblib"))
    args = parser.parse_args()
    print(predict(args.review, args.model))


if __name__ == "__main__":
    main()
