"""Shared, side-effect-free building blocks for the fake-news pipeline.

Everything train/serve/monitor needs to agree on lives here: how LIAR is
loaded, how the 6-way label collapses to binary, how the training split is
balanced, how text is tokenized, and how models are scored.
"""

from common.data import (
    SEED,
    binarize,
    build_balanced_train,
    combine_features,
    drop_ambiguous,
    load_liar,
    oversample_real,
    prepare_binary,
    undersample_fake,
)
from common.labels import FAKE, LABEL_NAMES, LIAR_TO_BINARY, REAL
from common.metrics import compute_metrics
from common.tokenize import (
    MAX_LENGTH,
    get_tokenizer,
    make_tokenize_fn,
    tokenize_for_trainer,
)

__all__ = [
    # labels
    "FAKE",
    "REAL",
    "LABEL_NAMES",
    "LIAR_TO_BINARY",
    # data
    "SEED",
    "load_liar",
    "binarize",
    "drop_ambiguous",
    "prepare_binary",
    "combine_features",
    "oversample_real",
    "undersample_fake",
    "build_balanced_train",
    # tokenize
    "MAX_LENGTH",
    "get_tokenizer",
    "make_tokenize_fn",
    "tokenize_for_trainer",
    # metrics
    "compute_metrics",
]
