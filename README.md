# Fake News Classifier

Binary fake-news detection on **LIAR**, a dataset of 12.8k short political claims. Two fine-tuned
transformers (DistilBERT, RoBERTa) are compared against TF-IDF baselines, and the result is reported
at its real size.

**Best model: DistilBERT, validation macro-F1 0.558, +0.033 over the TF-IDF baseline.** Short claims
carry very little signal. The baseline barely beats chance, and the transformer gain is real but small.

## Results

Validation split (864 rows: 511 FAKE / 353 REAL). The primary metric is macro F1.

| Model | Training set | Macro F1 | Accuracy |
|---|---|---:|---:|
| Majority class (always FAKE) | — | 0.372 | 0.591 |
| TF-IDF + logistic regression | 8,242 oversampled | 0.525 | 0.538 |
| TF-IDF + calibrated LinearSVC | 8,242 oversampled | 0.531 | 0.551 |
| DistilBERT v2 (bs 64, lr 3e-5) | 8,242 oversampled | 0.549 | 0.588 |
| **DistilBERT v3 (dropout 0.2, wd 0.008)** | **4,998 undersampled** | **0.558** | 0.567 |
| RoBERTa-base | 4,998 undersampled | 0.372 | 0.409 |

What the runs showed:

- **Oversampling REAL backfired.** Validation loss climbed from 0.68 to 2.37 over 10 epochs as the model memorised the duplicated rows.
- **Undersampling FAKE fixed it.** v3 trained on 40% less data and was both the most stable run and the best scorer.
- **RoBERTa collapsed to a single class.** Recall sat at exactly 0.5 every epoch. Not diagnosed yet.

The full table, including AUC and the speaker-history analysis, is in [`RESULTS.md`](RESULTS.md).

## Known gaps

I'd rather list these than have someone find them:

- **No held-out test score yet.** The test split is tokenised but never evaluated, and validation also drove model selection.
- **Not a like-for-like comparison.** The baselines trained on the oversampled set; the best transformer trained on the undersampled one.
- **One experiment never ran.** A char-n-gram TF-IDF vectoriser was defined but never refit.

## How it works

1. **Labels.** LIAR's six truthfulness ratings collapse to two. `pants-fire` and `false` become FAKE; `mostly-true` and `true` become REAL. `barely-true` and `half-true` are dropped as too ambiguous. That takes 10,269 training rows down to 6,620 (4,121 FAKE / 2,499 REAL).
2. **Balancing.** Either oversample REAL or undersample FAKE (seed 42).
3. **Models.** TF-IDF baselines with scikit-learn. DistilBERT and RoBERTa fine-tuned with the Hugging Face `Trainer`, max length 128, selected on macro F1.

## Run it

```bash
pip install -e ".[notebook]"   # Python 3.12+
jupyter notebook Fake_News_NLP.ipynb
```

For a CUDA build of PyTorch, install it first from the PyTorch index (see the note in `pyproject.toml`).

## Structure

| Path | What it is |
|---|---|
| `Fake_News_NLP.ipynb` | The full experiment: data prep, baselines, transformer runs, evaluation |
| `src/common/` | Deterministic building blocks lifted from the notebook: data loading and balancing, labels, tokenisation, metrics |
| `RESULTS.md` | Every number, the findings and the open gaps |

`pyproject.toml` also declares extras for training with MLflow, serving with FastAPI, drift monitoring with Evidently, and tests. Those packages (`src/train`, `src/serve`, `src/monitor`, `tests/`) aren't built yet.

## Stack

Python · PyTorch · Hugging Face Transformers & Datasets · scikit-learn · SHAP / LIME
