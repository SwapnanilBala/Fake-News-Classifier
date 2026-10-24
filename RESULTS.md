# Results

Binary fake-news classification on the **LIAR** dataset (original UCSB release, committed in
`data/liar/`). LIAR's 6-way truthfulness rating is collapsed to FAKE / REAL by label name, dropping
the ambiguous middle (`barely-true`, `half-true`): 10,269 → 6,489 training rows, imbalanced
2,840 FAKE / 3,649 REAL. Validation is 799 rows (379 FAKE / 420 REAL); test is 802 rows
(342 FAKE / 460 REAL).

## Held-out results

Primary metric is macro F1. Each transformer keeps its best epoch by validation macro F1
(`load_best_model_at_end`); the test split is scored once afterwards and drove no decision.
"Rich text" is statement + speaker + party + context. One seed per model.

| Model | Training set | Val F1 | Val acc | Val AUC | Test F1 | Test acc | Test AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Always predict REAL | — | 0.345 | 0.526 | — | 0.365 | 0.574 | — |
| TF-IDF word 1–5-gram (10k) + LogisticRegression, statement | 7,298 oversampled | 0.672 | 0.673 | 0.732 | 0.651 | 0.658 | 0.705 |
| TF-IDF char_wb 1–5-gram (100k cap) + calibrated LinearSVC, rich text | 7,298 oversampled | 0.656 | 0.658 | 0.729 | 0.652 | 0.662 | 0.731 |
| DistilBERT v2 (bs 64, lr 3e-5, wd 0.01, 10 ep), statement | 7,298 oversampled | 0.672 | 0.675 | 0.722 | 0.629 | 0.641 | 0.701 |
| DistilBERT v3 (dropout 0.2, bs 32, lr 2e-5, wd 0.008), statement | 5,680 undersampled | 0.703 | 0.703 | 0.758 | 0.666 | 0.675 | 0.737 |
| **DistilBERT v3 (dropout 0.2, bs 32, lr 2e-5, wd 0.007), rich text** | **5,680 undersampled** | 0.701 | 0.706 | 0.771 | **0.686** | **0.703** | **0.763** |
| RoBERTa-base (dropout 0.2, bs 8, lr 2e-5), statement | 5,680 undersampled | **0.715** | **0.716** | 0.766 | 0.659 | 0.668 | 0.728 |

**Best held-out model: DistilBERT v3 on rich text** — test macro F1 0.686, accuracy 0.703,
AUC 0.763. That is +0.034 F1 over the best TF-IDF baseline on test and +0.321 over the
majority-class floor.

## Findings

- **Rich text is the clearest gain.** The rich-text DistilBERT has the best test F1, accuracy and
  AUC, and the smallest validation → test drop (−0.015). The statement-only v3 drops −0.037.
  Anything trained on rich text needs all four fields at inference time.
- **Validation and test disagree on the winner.** RoBERTa is best on validation (0.715) but third on
  test (0.659), the largest drop of any model (−0.056). Its kept epoch was the last one (5), with
  validation loss already rising from epoch 3. With ~800 rows per split, the standard error on
  accuracy is about ±0.016, so the transformer ranking is not settled by one seed each.
- **Oversampling still overfits.** v2's validation loss climbs 0.60 (epoch 2) → 1.65 (epoch 10)
  while validation F1 wanders between 0.64 and 0.67. Selecting by F1 kept epoch 8 (loss 1.50), and
  that checkpoint is the weakest transformer on test (0.629).
- **Transformers beat TF-IDF, modestly.** +0.007 to +0.034 test F1 for the undersampled
  transformers over the best baseline. The two baselines differ in classifier, n-gram type and input
  text, yet land 0.001 apart on test.
- **Word 1–5-grams vs 1–2-grams changed little.** Val F1 0.677 → 0.672, test 0.644 → 0.651. The
  10k-feature cap keeps mostly short n-grams: 3,311 unigrams, 4,572 bigrams, 1,570 trigrams,
  423 4-grams and 124 5-grams.
- **Speaker credit-history counts correlate with the label, but would leak it.** Pearson r with
  REAL: `pants_on_fire_counts` −0.221, `false_counts` −0.099, `mostly_true_counts` +0.097,
  `half_true_counts` +0.071, `barely_true_counts` −0.004. The LIAR README says these counts include
  the current statement, so they stay out of the features.

## Run-to-run variance

The notebook was run twice on identical data (once loaded from the Hugging Face cache, once from the
TSVs; verified row-for-row identical). The only code difference was the logistic-regression n-gram
range.

| Model | Run 1 val / test F1 | Run 2 val / test F1 |
|---|---|---|
| TF-IDF + LogisticRegression (1–2-gram → 1–5-gram) | 0.677 / 0.644 | 0.672 / 0.651 |
| TF-IDF char + LinearSVC, rich text | 0.656 / 0.652 | 0.656 / 0.652 |
| DistilBERT v2, oversampled | 0.679 / 0.667 | 0.672 / 0.629 |
| DistilBERT v3, statement | 0.706 / 0.672 | 0.703 / 0.666 |
| DistilBERT v3, rich text | 0.701 / 0.686 | 0.701 / 0.686 |
| RoBERTa, statement | 0.715 / 0.659 | 0.715 / 0.659 |

v2 and v3-statement diverge from epoch 1, so they started from different weights. Each model is
created *before* its `Trainer` calls `set_seed(42)`, so its initial classifier head depends on how
much randomness earlier cells consumed. v2 also early-stopped at epoch 6 in run 1 but trained all
10 epochs in run 2, which shifts everything after it. The rich-text and RoBERTa runs reproduced to
four decimals. Seeding immediately before each model is created would make every run repeatable.

## History: the label-mapping bug

Every number published before this table — including the old "best model 0.558" — was computed on
**wrong labels**.

- The Hugging Face copy of LIAR stores labels as `ClassLabel` ids whose order is
  `false, half-true, mostly-true, true, barely-true, pants-fire`. The notebook and
  `src/common/labels.py` assumed truthfulness order (`pants-fire, false, barely-true, …`).
- Effect: "FAKE" was false + half-true, "REAL" was barely-true + pants-fire, and mostly-true / true
  were dropped. The old 4,121 / 2,499 class split is exactly those counts.
- Giveaways in hindsight: `pants_on_fire_counts` correlated *positively* with "REAL", and every
  model sat near chance (TF-IDF AUC ~0.55).
- Fix: labels are mapped by name. The data now comes from the TSVs, where labels are strings.

The earlier multi-seed re-analysis, checkpoint-provenance notes and RoBERTa-collapse diagnosis were
all made on the wrong labels and have been retired. RoBERTa no longer collapses at bs 8. All
checkpoints trained on the wrong labels have been deleted.

## Other fixes in this round

- **Data is local.** `ucsbnlp/liar` now redirects to `ucsbai/liar`, which ships only a loading
  script that `datasets` 4.x will not run. The notebook only worked from a stale local cache. The
  original TSVs are now in `data/liar/`, read with `quoting=csv.QUOTE_NONE` (pandas' default
  quoting silently merges 29 training rows) and `keep_default_na=False`.
- **Test-set evaluation added** after every model, plus a summary table.
- **The rich-text / char-n-gram TF-IDF experiment now runs.** The vectoriser is refit on rich text
  for validation and test; it previously sat unfitted while LinearSVC reused the old matrix.
- **The rich-text DistilBERT run is evaluated on rich-text validation** and starts from a fresh
  model. It previously reused the already-trained statement model and scored on statement-only text.
- **Balancing is class-agnostic.** Oversampling and undersampling pick the minority and majority
  classes by count (`oversample_minority`, `undersample_majority`), since REAL is now the larger class.

## Known gaps

- **One seed per model**, and runs are not fully reproducible (see above).
- **`best_model` is overwritten unconditionally.** `distilbert_fakenews/best_model` holds v2, the
  weakest transformer. The best model lives only as `distilbert_fakenews_v3/checkpoint-356`, and
  both v3 runs write checkpoints with the same names into that folder.
- **Baselines trained on the oversampled set, the best transformers on the undersampled set** — not
  a like-for-like comparison.
- **Saved configs carry no label names**, so anything loading the notebook's checkpoints gets
  `LABEL_0`/`LABEL_1`. Runs from `src/common` set `id2label`/`label2id`.
- No experiment tracking; notebook cells are still edited in place.

## Status

- `src/common/` — shared, deterministic building blocks (data, labels, tokenisation, metrics). Done;
  loads from `data/liar/`.
- `src/train/`, `src/serve/`, `src/monitor/`, `tests/` — declared in `pyproject.toml` extras
  (MLflow, FastAPI, Evidently, pytest), **not yet built**.
