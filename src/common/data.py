"""Dataset loading, label binarization, feature combination, and balancing.

Every function here is deterministic and side-effect-free: no training, no
file writes, no plotting. Given the same inputs (and seed) they always produce
the same datasets, so they are safe to import from train/serve/monitor and to
unit-test in isolation.

Lifted from the data-prep cells of Fake_News_NLP.ipynb.
"""

from __future__ import annotations

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from common.labels import FAKE, REAL, LIAR_TO_BINARY

# Default seed shared across balancing/shuffling so runs are reproducible.
SEED = 42


def load_liar() -> DatasetDict:
    """Load the LIAR dataset (train/validation/test splits).

    Pulls the parquet conversion revision, which is the variant that loads
    cleanly without the legacy dataset script.
    """
    return load_dataset("ucsbnlp/liar", revision="refs/convert/parquet")


def binarize(example: dict) -> dict:
    """Add a `binary_label` field (0=FAKE, 1=REAL, or None for the middle).

    Use as a `.map()` function. Rows with a None label survive this step and
    are removed by :func:`drop_ambiguous`.
    """
    example["binary_label"] = LIAR_TO_BINARY[example["label"]]
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
    dataset = dataset.map(binarize)
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


def oversample_real(train: Dataset, seed: int = SEED) -> Dataset:
    """Balance by repeating REAL rows up to the FAKE count, then shuffling.

    This is the v1 strategy from the notebook. It keeps every FAKE example but
    duplicates REAL ones, so the model sees the same REAL claims many times.
    """
    fake = train.filter(lambda x: x["binary_label"] == FAKE)
    real = train.filter(lambda x: x["binary_label"] == REAL)

    target = len(fake)
    repeats = target // len(real)
    remainder = target % len(real)

    real_oversampled = concatenate_datasets(
        [real] * repeats
        + [real.shuffle(seed=seed).select(range(remainder))]
    )
    return concatenate_datasets([fake, real_oversampled]).shuffle(seed=seed)


def undersample_fake(train: Dataset, seed: int = SEED) -> Dataset:
    """Balance by sampling FAKE rows down to the REAL count, then shuffling.

    This is the v2/v3 strategy the notebook settled on -- it avoids the recall
    collapse that oversampling caused by not showing duplicate REAL claims.
    """
    fake = train.filter(lambda x: x["binary_label"] == FAKE)
    real = train.filter(lambda x: x["binary_label"] == REAL)

    fake_sampled = fake.shuffle(seed=seed).select(range(len(real)))
    return concatenate_datasets([fake_sampled, real]).shuffle(seed=seed)


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
        return undersample_fake(train, seed=seed)
    if strategy == "oversample":
        return oversample_real(train, seed=seed)
    raise ValueError(f"unknown balancing strategy: {strategy!r}")
