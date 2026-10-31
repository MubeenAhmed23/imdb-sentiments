"""Reproducible classical NLP pipeline for IMDb sentiment classification."""
import re

import numpy as np
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Keep negation words: removing "not" can reverse review sentiment.
STEMMER = SnowballStemmer("english")
STOP_WORDS = ENGLISH_STOP_WORDS - {"no", "nor", "not", "never", "neither", "none"}


def tokenize_and_stem(text: str) -> list[str]:
    """Lowercase, tokenize, and stem text while preserving negation."""
    return [STEMMER.stem(token) for token in re.findall(r"(?u)\b\w\w+\b", text.lower())
            if token not in STOP_WORDS]


def build_model() -> Pipeline:
    """TF-IDF word/bigram features plus a linear SVM, fit without test leakage."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
            min_df=2,
            max_features=200_000,
            sublinear_tf=True,
            dtype=np.float32,
        )),
        ("classifier", LinearSVC(C=1.0, random_state=42)),
    ])
