"""Tokenizer construction and dataset tokenization helpers.

Generic over the model family: pass "distilbert-base-uncased" or "roberta-base"
and you get the matching fast tokenizer via AutoTokenizer. Pure helpers -- they
return new datasets and never mutate global state.

Lifted from the tokenize / tokenize_rich / tokenize_roberta cells of the notebook.
"""

from __future__ import annotations

from collections.abc import Callable

from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Sequence length used across all the notebook's transformer runs.
MAX_LENGTH = 128

# Columns the HF Trainer needs, and the label rename it expects.
_MODEL_COLUMNS = ["input_ids", "attention_mask", "binary_label"]


def get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load the fast tokenizer for any supported checkpoint."""
    return AutoTokenizer.from_pretrained(model_name)


def make_tokenize_fn(
    tokenizer: PreTrainedTokenizerBase,
    text_field: str = "statement",
    max_length: int = MAX_LENGTH,
) -> Callable[[dict], dict]:
    """Build a batched `.map()` function that tokenizes `text_field`.

    Use `text_field="rich_text"` to tokenize the statement+metadata string
    produced by :func:`common.data.combine_features`.
    """

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch[text_field],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return tokenize


def tokenize_for_trainer(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    text_field: str = "statement",
    max_length: int = MAX_LENGTH,
) -> Dataset:
    """Tokenize a split and shape it for `transformers.Trainer`.

    Tokenizes, drops everything except input_ids/attention_mask/label, renames
    `binary_label` -> `labels`, and sets the torch output format.
    """
    tokenized = dataset.map(
        make_tokenize_fn(tokenizer, text_field=text_field, max_length=max_length),
        batched=True,
    )
    tokenized = tokenized.select_columns(_MODEL_COLUMNS)
    tokenized = tokenized.rename_column("binary_label", "labels")
    tokenized.set_format("torch")
    return tokenized
