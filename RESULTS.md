# Results

Binary fake-news classification on the **LIAR** dataset. LIAR's 6-way truthfulness rating is
collapsed to FAKE / REAL, dropping the ambiguous middle (`barely-true`, `half-true`):
10,269 → 6,620 training rows, imbalanced 4,121 FAKE / 2,499 REAL.

## Baseline metrics

All figures are **validation** (864 rows: 511 FAKE / 353 REAL). Primary metric is macro F1.

| Model | Training set | Macro F1 | Accuracy | AUC |
|---|---|---|---|---|
| TF-IDF (10k, 1–2gram) + LogisticRegression | 8,242 oversampled | 0.525 | 0.538 | 0.549 |
| TF-IDF + calibrated LinearSVC | 8,242 oversampled | 0.531 | 0.551 | 0.535 |
| DistilBERT v1 (bs 32) | 8,242 oversampled | 0.544 | 0.564 | — |
| DistilBERT v2 (bs 64, lr 3e-5, 10 ep) | 8,242 oversampled | 0.549 | 0.588 | — |
| **DistilBERT v3 (dropout 0.2, wd 0.008)** | 4,998 undersampled | **0.558** | 0.567 | — |
| RoBERTa-base (bs 8, lr 2e-5) | 4,998 undersampled | 0.372 | 0.409 | — |

Reference point: always predicting the majority class scores **0.372** macro F1 / 0.591 accuracy.

**Best model:** DistilBERT v3, macro F1 0.558 at epoch 3.

## Findings

- **TF-IDF barely beats chance.** AUC 0.549 / 0.535. Short political claims carry very little
  bag-of-words signal.
- **Oversampling REAL backfired.** v2's validation loss climbed 0.68 → 2.37 over 10 epochs while
  accuracy drifted up — it memorised the duplicated REAL rows. v1 peaked at epoch 1 and decayed.
- **Undersampling FAKE fixed it.** v3 is the flattest, most stable run and the best scorer,
  despite training on 40% less data.
- **RoBERTa collapsed to a single class.** Recall is exactly 0.5000 every epoch; accuracy alternates
  between 0.409 (=353/864, all-REAL) and 0.591 (=511/864, all-FAKE). Not diagnosed yet.
- **Speaker credit-history counts are near-useless.** Max |Pearson r| = 0.104
  (`pants_on_fire_counts`); the rest are < 0.06. Correctly left out as features.
- **Transformers beat TF-IDF by only ~0.03 macro F1.** The gain is real but small.

## Known gaps

- **No test-set evaluation.** The test split is tokenised but never scored. Validation also drove
  model selection (`metric_for_best_model="f1_macro"`), so there is no clean held-out estimate.
- **The rich-text / char-ngram experiment never ran.** The vectoriser was redefined
  (50k features, trigrams, `char_wb`) but never refit, so LinearSVC silently consumed the original
  10k / 1–2gram matrix. That 0.531 is the *old* feature set.
- **Artifact directories are tangled.** Two runs write to `distilbert_fakenews_v3/`, and the v2 run
  saves its best model into `distilbert_fakenews/`. On-disk checkpoints can't be attributed to a config.
- Baselines trained on the oversampled set, best transformer on the undersampled set — not a
  like-for-like comparison.
- No experiment tracking; notebook cells were edited in place, so v1's hyperparameters survive only
  in `trainer_state.json`.

## Status

- `src/common/` — shared, deterministic building blocks (data, labels, tokenisation, metrics). Done.
- `src/train/`, `src/serve/`, `src/monitor/`, `tests/` — declared in `pyproject.toml` extras
  (MLflow, FastAPI, Evidently, pytest), **not yet built**.
