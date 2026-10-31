"""Download and load the Stanford IMDb Large Movie Review dataset."""
from __future__ import annotations

import hashlib
import re
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from pathlib import Path

import pandas as pd

URLS = [
    "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    "https://dataset.bj.bcebos.com/imdb%2FaclImdb_v1.tar.gz",
]
MD5 = "7c2ac02c03563afcf9b574c7e56c153a"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_complete(root: Path) -> bool:
    return all((root / split / label).is_dir()
               and len(list((root / split / label).glob("*.txt"))) == 12_500
               for split in ("train", "test") for label in ("pos", "neg"))


def download_imdb(data_dir: Path = DATA_DIR) -> Path:
    """Return a complete extracted dataset, downloading it if necessary."""
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted = data_dir / "aclImdb"
    if _is_complete(extracted):
        return extracted
    archive = data_dir / "aclImdb_v1.tar.gz"
    if not archive.exists() or _md5(archive) != MD5:
        archive.unlink(missing_ok=True)
        for url in URLS:
            print(f"Downloading IMDb dataset from {url}")
            try:
                urllib.request.urlretrieve(url, archive)
                if _md5(archive) != MD5:
                    archive.unlink(missing_ok=True)
                    raise RuntimeError("Downloaded archive failed its MD5 checksum.")
                break
            except (OSError, RuntimeError) as exc:
                archive.unlink(missing_ok=True)
                print(f"Download source failed: {exc}")
        else:
            raise RuntimeError("Could not download the dataset. Place the verified aclImdb_v1.tar.gz in "
                               f"{data_dir} and retry.")
    temp = data_dir / "_extracting"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir()
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(temp, filter="data")
    if not _is_complete(temp / "aclImdb"):
        raise RuntimeError("Extracted IMDb dataset is incomplete; archive may be damaged.")
    if extracted.exists():
        shutil.rmtree(extracted)
    (temp / "aclImdb").rename(extracted)
    temp.rmdir()
    return extracted


def _read_movie_ids(path: Path) -> list[str | None]:
    with path.open(encoding="utf-8") as stream:
        return [match.group(0) if (match := re.search(r"tt\d+", line)) else None
                for line in stream]


def load_split(split: str, root: Path) -> pd.DataFrame:
    """Load one official split, preserving IDs, raw text, ratings, and labels."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    rows = []
    for label_name, label in (("neg", 0), ("pos", 1)):
        folder = root / split / label_name
        ids_path = root / split / f"urls_{label_name}.txt"
        if not folder.is_dir() or not ids_path.is_file():
            raise FileNotFoundError(f"Missing {split}/{label_name} reviews or URL mapping in {root}")
        movie_ids = _read_movie_ids(ids_path)
        paths = sorted(folder.glob("*.txt"), key=lambda p: p.name)
        def read_review(path: Path) -> dict:
            review_id, rating = path.stem.rsplit("_", 1)
            index = int(review_id)
            return {
                "review_id": f"{split}_{label_name}_{review_id}",
                "text": path.read_text(encoding="utf-8", errors="replace"),
                "label": label,
                "rating": int(rating),
                "movie_id": movie_ids[index] if index < len(movie_ids) else None,
            }
        # The dataset has 50,000 small files. Bounded parallel reads reduce
        # wall time on Windows and network-backed folders while preserving order.
        with ThreadPoolExecutor(max_workers=16) as pool:
            rows.extend(pool.map(read_review, paths))
    frame = pd.DataFrame(rows)
    if frame.empty or frame["text"].isna().any():
        raise ValueError(f"No valid reviews found in {split} split at {root}")
    return frame


def load_imdb() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load official 25k-review train and test splits; never silently return partial data."""
    root = download_imdb()
    train, test = load_split("train", root), load_split("test", root)
    for name, frame in (("train", train), ("test", test)):
        counts = frame["label"].value_counts().to_dict()
        if len(frame) != 25_000 or counts != {0: 12_500, 1: 12_500}:
            raise ValueError(f"Expected 25,000 balanced {name} reviews; found {len(frame)} ({counts}).")
    return train, test
