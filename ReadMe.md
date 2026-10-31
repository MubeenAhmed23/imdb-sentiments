# IMDb Movie Review Sentiment Analysis

An end-to-end NLP project for binary sentiment classification on the Stanford IMDb Large Movie Review Dataset. It uses the dataset's official train/test split, English stop-word filtering (with negation terms retained), Snowball stemming, TF-IDF uni- and bigrams, and a linear Support Vector Machine.

## Requirements and setup

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The dataset downloads on the first run (about 84 MB compressed). The loader verifies the official MD5 checksum and both class folders in each split. If network access is unavailable, put the official `aclImdb_v1.tar.gz` archive in `data/` and run the pipeline again.

## Train and evaluate

From this directory:

```powershell
python -m src.train
```

The default experiment trains on stratified, reproducible samples of 1,000, 5,000, and 25,000 training reviews. Every scenario is evaluated on the same untouched 25,000-review official test set. Results are saved in `results/metrics.csv` and `results/metrics.json`; the full-data model, class-wise report, and confusion matrix are also saved there. To choose different training sizes:

```powershell
python -m src.train --sizes 500 2500 10000
```

## Classify a review

After training:

```powershell
python -m src.predict "A warm, funny film with a terrific cast and a satisfying ending."
```

The output is `positive` or `negative`. The Python API is `from src.predict import predict; predict("review text")`.

## Method and evaluation

Text is lowercased and tokenized, common English stop words are removed while negation cues such as *not* and *never* are retained, and tokens are stemmed. TF-IDF uses unigrams and bigrams. A linear SVM is trained using only the official training split; there is no fitting on the test set. The progressive-size experiment reports accuracy, precision, recall, F1, and fit time for the positive class. `results/metrics.csv` is the source of truth for measured results; do not report unrun or estimated scores.

The measured project report is in [`results/REPORT.md`](results/REPORT.md).
Upstream repository: [github.com/MubeenAhmed23/imdb-sentiments](https://github.com/MubeenAhmed23/imdb-sentiments).

## Submission files

- `submission/Project_Report.pdf` - rendered report for review/submission
- `submission/Project_Report.docx` - editable report source
- `results/REPORT.md` - concise report and measured outcomes

## Project structure

```text
src/data.py       verified dataset acquisition and loading
src/model.py      stemming, TF-IDF, and classifier pipeline
src/train.py      training-size experiments and evaluation artifacts
src/predict.py    command-line and Python prediction interface
notebooks/        exploratory analysis
results/          generated metrics, model, report, and figure
```

The dataset is from Maas et al. (2011), *Learning Word Vectors for Sentiment Analysis*, ACL. Dataset page: https://ai.stanford.edu/~amaas/data/sentiment/ . The academic task brief also lists the Stanford IMDb dataset as a suitable public benchmark.
