"""Dataset loading, label binarization, feature combination, and balancing.

Every function here is deterministic and side-effect-free: no training, no
file writes, no plotting. Given the same inputs (and seed) they always produce
the same datasets, so they are safe to import from train/serve/monitor and to
unit-test in isolation.

Lifted from the data-prep cells of Fake_News_NLP.ipynb.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets

from common.labels import FAKE, REAL, LIAR_TO_BINARY

# Default seed shared across balancing/shuffling so runs are reproducible.
SEED = 42

# Repo-local copy of the original UCSB LIAR release (train/valid/test TSVs).
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "liar"

# The TSVs have no header; column order is documented in data/liar/README.
LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job_title", "state_info",
    "party_affiliation", "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context",
]

# DatasetDict split name -> TSV file stem.
_SPLIT_FILES = {"train": "train", "validation": "valid", "test": "test"}


def read_liar_tsv(path: str | Path) -> pd.DataFrame:
    """Read one LIAR TSV into a DataFrame with named columns.

    Quoting is disabled because statements contain stray quote characters --
    with pandas' default quoting, 29 training rows silently merge into their
    neighbours. Empty fields (job_title, state_info, context) stay "" not NaN.
    """
    return pd.read_csv(
        path, sep="\t", header=None, names=LIAR_COLUMNS,
        quoting=csv.QUOTE_NONE, keep_default_na=False,
    )


def load_liar(data_dir: str | Path = DATA_DIR) -> DatasetDict:
    """Load the LIAR train/validation/test splits from the repo's TSVs.

    Reads local files only -- no Hugging Face Hub access. The Hub repo now
    ships just a loading script, which `datasets` 4.x no longer runs.
    """
    data_dir = Path(data_dir)
    return DatasetDict({
        split: Dataset.from_pandas(read_liar_tsv(data_dir / f"{stem}.tsv"))
        for split, stem in _SPLIT_FILES.items()
    })


def binarize(example: dict, mapping: dict[str, int | None] = LIAR_TO_BINARY) -> dict:
    """Add a `binary_label` field (0=FAKE, 1=REAL, or None for the middle).

    Use as a `.map()` function. Rows with a None label survive this step and
    are removed by :func:`drop_ambiguous`. Pass the mapping via
    ``fn_kwargs={"mapping": ...}`` so it is part of the `datasets` cache
    fingerprint -- otherwise edits to LIAR_TO_BINARY silently reuse stale
    cached results.
    """
    example["binary_label"] = mapping[example["label"]]
    return example


def drop_ambiguous(dataset):
    """Filter out rows whose `binary_label` is None (barely-true / half-true).

    Accepts a single split (`Dataset`) or a whole `DatasetDict`.
    """
    return dataset.filter(lambda x: x["binary_label"] is not None)


def prepare_binary(dataset: DatasetDict | None = None) -> DatasetDict:
    """Load (if needed), binarize, and drop the ambiguous middle in one call."""
    if dataset is None:
        dataset = load_liar()
    dataset = dataset.map(binarize, fn_kwargs={"mapping": LIAR_TO_BINARY})
    return drop_ambiguous(dataset)


def combine_features(example: dict) -> dict:
    """Add a `rich_text` field: statement + speaker + party + context.

    The richer string lets a model condition on metadata, not just the claim.
    Use as a `.map()` function. NOTE: anything trained on `rich_text` needs the
    same four fields at inference time -- keep that in mind for the serve schema.
    """
    example["rich_text"] = (
        f"{example['statement']} "
        f"speaker:{example['speaker']} "
        f"party:{example['party_affiliation']} "
        f"context:{example['context']}"
    )
    return example


def _count(split: Dataset, label: int) -> int:
    return split["binary_label"].count(label)


def _split_by_size(train: Dataset) -> tuple[Dataset, Dataset]:
    """Return (minority, majority) class subsets of `train`."""
    fake = train.filter(lambda x: x["binary_label"] == FAKE)
    real = train.filter(lambda x: x["binary_label"] == REAL)
    minority, majority = sorted([fake, real], key=len)
    return minority, majority


def oversample_minority(train: Dataset, seed: int = SEED) -> Dataset:
    """Balance by repeating minority rows up to the majority count, then shuffling.

    This is the v1 strategy from the notebook. It keeps every majority example
    but duplicates minority ones (FAKE on LIAR), so the model sees the same
    minority claims many times.
    """
    minority, majority = _split_by_size(train)

    target = len(majority)
    repeats = target // len(minority)
    remainder = target % len(minority)

    minority_oversampled = concatenate_datasets(
        [minority] * repeats
        + [minority.shuffle(seed=seed).select(range(remainder))]
    )
    return concatenate_datasets([majority, minority_oversampled]).shuffle(seed=seed)


def undersample_majority(train: Dataset, seed: int = SEED) -> Dataset:
    """Balance by sampling majority rows down to the minority count, then shuffling.

    This is the v2/v3 strategy the notebook settled on -- it avoids the recall
    collapse that oversampling caused by not showing duplicate minority claims.
    """
    minority, majority = _split_by_size(train)

    majority_sampled = majority.shuffle(seed=seed).select(range(len(minority)))
    return concatenate_datasets([majority_sampled, minority]).shuffle(seed=seed)


def build_balanced_train(
    train: Dataset,
    strategy: str = "undersample",
    seed: int = SEED,
) -> Dataset:
    """Return a class-balanced training split.

    `strategy` is "undersample" (default, the notebook's final choice) or
    "oversample".
    """
    if strategy == "undersample":
        return undersample_majority(train, seed=seed)
    if strategy == "oversample":
        return oversample_minority(train, seed=seed)
    raise ValueError(f"unknown balancing strategy: {strategy!r}")
