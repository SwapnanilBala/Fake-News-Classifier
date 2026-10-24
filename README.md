# Fake News Classifier

Binary fake-news detection on **LIAR**, a dataset of 12.8k short political claims rated by
PolitiFact. Two fine-tuned transformers (DistilBERT, RoBERTa) are compared against TF-IDF baselines,
and every model is scored once on the held-out test split.

**Best model: DistilBERT on statement + speaker + party + context — test macro-F1 0.686, accuracy
0.703, AUC 0.763.** That is +0.034 over the best TF-IDF baseline on test (0.652) and far above the
0.365 floor of always predicting the majority class.

## Results

Validation is 799 rows (379 FAKE / 420 REAL); test is 802 rows (342 FAKE / 460 REAL). The primary
metric is macro F1. Epochs were picked on validation; the test split was scored once and never used
for a decision. "Rich text" is statement + speaker + party + context.

| Model | Training set | Val F1 | Test F1 | Test acc | Test AUC |
|---|---|---:|---:|---:|---:|
| Always predict REAL | — | 0.345 | 0.365 | 0.574 | — |
| TF-IDF word 1–5-gram + logistic regression | 7,298 oversampled | 0.672 | 0.651 | 0.658 | 0.705 |
| TF-IDF char 1–5-gram + calibrated LinearSVC, rich text | 7,298 oversampled | 0.656 | 0.652 | 0.662 | 0.731 |
| DistilBERT v2 (bs 64, lr 3e-5) | 7,298 oversampled | 0.672 | 0.629 | 0.641 | 0.701 |
| DistilBERT v3 (dropout 0.2), statement | 5,680 undersampled | 0.703 | 0.666 | 0.675 | 0.737 |
| **DistilBERT v3 (dropout 0.2), rich text** | **5,680 undersampled** | 0.701 | **0.686** | **0.703** | **0.763** |
| RoBERTa-base (bs 8, lr 2e-5), statement | 5,680 undersampled | **0.715** | 0.659 | 0.668 | 0.728 |

What the runs showed:

- **Rich text helps the transformer.** Best test F1, accuracy and AUC, and the smallest
  validation-to-test drop (0.701 → 0.686).
- **Validation picked the wrong winner.** RoBERTa tops validation (0.715) but drops to 0.659 on
  test. With ~800 rows per split and one seed per model, gaps of about ±0.02 are within noise.
- **Oversampling still overfits.** v2's validation loss climbs from 0.60 to 1.65 over 10 epochs.
  The epoch that validation F1 kept is the weakest transformer on test.
- **Single runs are noisy.** Re-running on identical data moved v2's test F1 from 0.667 to 0.629 and
  v3-statement's from 0.672 to 0.666. The rich-text and RoBERTa runs reproduced exactly.

The full table, run-to-run comparison and open issues are in [`RESULTS.md`](RESULTS.md).

## A label bug, fixed

Earlier versions of this repo reported a best validation F1 of 0.558. Those numbers were wrong. The
Hugging Face copy of LIAR stores labels as ids in the order `false, half-true, mostly-true, true,
barely-true, pants-fire`, not in truthfulness order, and the old mapping assumed truthfulness order.
"FAKE" was really false + half-true and "REAL" was barely-true + pants-fire. Labels are now mapped
by name, and every number above uses the corrected labels.

## Known gaps

- **One seed per model.** The ranking among the transformers is not statistically settled.
- **Runs aren't fully reproducible.** Each model is created before the `Trainer` seeds the RNG, so
  its starting weights depend on what ran earlier in the notebook.
- **`best_model` folders are overwritten unconditionally.** `distilbert_fakenews/best_model` holds
  v2, the weakest transformer. The best model exists only as a checkpoint in
  `distilbert_fakenews_v3/`, a folder both v3 runs write to.
- **Not a like-for-like comparison.** The baselines trained on the oversampled set; the best
  transformer trained on the undersampled one.
- **Speaker credit counts would leak the label.** Per the dataset's README they include the current
  statement's own rating, so they are left out as features.

## How it works

1. **Data.** The original UCSB LIAR release, committed in [`data/liar/`](data/liar/) as three TSV
   files (3 MB). They are read with pandas (quoting off, empty fields kept as `""`) and converted to
   Hugging Face Datasets. No Hub access is needed for the data.
2. **Labels.** Six ratings collapse to two, mapped by name. `pants-fire` and `false` become FAKE;
   `mostly-true` and `true` become REAL; `barely-true` and `half-true` are dropped as too ambiguous.
   That takes 10,269 training rows down to 6,489 (2,840 FAKE / 3,649 REAL).
3. **Balancing.** Either oversample FAKE to 7,298 rows or undersample REAL to 5,680 (seed 42).
4. **Models.** TF-IDF baselines with scikit-learn, using word and character 1–5-grams. DistilBERT
   and RoBERTa are fine-tuned with the Hugging Face `Trainer` at max length 128, keep their best
   epoch by validation macro F1, and are then scored once on test.

## Run it

```bash
pip install -e ".[notebook]"   # Python 3.12+
jupyter notebook Fake_News_NLP.ipynb
```

The data ships with the repo. The pretrained `distilbert-base-uncased` and `roberta-base` weights
download from the Hugging Face Hub on first use; no token is needed. For a CUDA build of PyTorch,
install it first from the PyTorch index (see the note in `pyproject.toml`).

## Structure

| Path | What it is |
|---|---|
| `Fake_News_NLP.ipynb` | The full experiment: data prep, baselines, transformer runs, validation and test evaluation |
| `data/liar/` | The LIAR TSVs plus the dataset's own README (column layout; research use only) |
| `src/common/` | Deterministic building blocks lifted from the notebook: data loading and balancing, labels, tokenisation, metrics |
| `RESULTS.md` | Every number, the findings and the open gaps |

`pyproject.toml` also declares extras for training with MLflow, serving with FastAPI, drift
monitoring with Evidently, and tests. Those packages (`src/train`, `src/serve`, `src/monitor`,
`tests/`) aren't built yet.

## Stack

Python · pandas · PyTorch · Hugging Face Transformers & Datasets · scikit-learn · SHAP / LIME
