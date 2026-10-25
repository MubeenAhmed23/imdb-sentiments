import hashlib
import re
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

import pandas as pd

URLS = [
    "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    "https://dataset.bj.bcebos.com/imdb%2FaclImdb_v1.tar.gz",  # PaddlePaddle mirror
]
MD5 = "7c2ac02c03563afcf9b574c7e56c153a"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_imdb(data_dir: Path = DATA_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted = data_dir / "aclImdb"
    if extracted.exists():
        return extracted

    archive = data_dir / "aclImdb_v1.tar.gz"
    if not archive.exists():
        for url in URLS:
            print(f"Trying {url} ...")
            try:
                urllib.request.urlretrieve(url, archive)
                break
            except OSError as e:
                print(f"  failed: {e}")
                archive.unlink(missing_ok=True)
        else:
            raise RuntimeError(
                "All download sources failed. Download manually from one of:\n  "
                + "\n  ".join(URLS)
                + f"\nand save it as:\n  {archive}"
            )

    if _md5(archive) != MD5:
        archive.unlink()
        raise RuntimeError("Checksum mismatch: file is corrupted or not the original. Run again.")

    print("Extracting...")
    tmp_dir = data_dir / "_extracting"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    with tarfile.open(archive) as tar:
        tar.extractall(tmp_dir, filter="data")

    for attempt in range(10):
        try:
            (tmp_dir / "aclImdb").rename(extracted)
            break
        except PermissionError:
            print(f"  folder locked (antivirus/indexer?), retrying in 3s ({attempt + 1}/10)")
            time.sleep(3)
    else:
        raise RuntimeError(
            f"Could not rename {tmp_dir / 'aclImdb'} to {extracted}. "
            "Close programs using the folder and move it manually."
        )

    tmp_dir.rmdir()
    return extracted


def _read_movie_ids(urls_file: Path) -> list[str | None]:
    ids = []
    with open(urls_file, encoding="utf-8") as f:
        for line in f:
            match = re.search(r"tt\d+", line)
            ids.append(match.group(0) if match else None)
    return ids


def load_split(split: str, root: Path) -> pd.DataFrame:
    rows = []
    for label_name, label in [("pos", 1), ("neg", 0)]:
        folder = root / split / label_name
        movie_ids = _read_movie_ids(root / split / f"urls_{label_name}.txt")

        for path in folder.glob("*.txt"):
            review_id, rating = path.stem.split("_")
            rows.append({
                "review_id": f"{split}_{label_name}_{review_id}",
                "text": path.read_text(encoding="utf-8"),
                "label": label,
                "rating": int(rating),
                "movie_id": movie_ids[int(review_id)],
            })
    return pd.DataFrame(rows)


def load_imdb() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = download_imdb()
    train, test = load_split("train", root), load_split("test", root)
    if len(train) != 25_000 or len(test) != 25_000:
        raise ValueError(
            f"Expected 25,000 train and 25,000 test reviews, got {len(train)}/{len(test)}. "
            "Delete data/aclImdb and run again."
        )
    return train, test


if __name__ == "__main__":
    train, test = load_imdb()
    print(train.shape, test.shape)
    print(train.head())