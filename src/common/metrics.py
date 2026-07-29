"""Evaluation metrics shared by the training runs.

`compute_metrics` is the callback handed to `transformers.Trainer`; the macro
F1 it returns is the notebook's primary model-selection metric. Pure function:
it reads logits + labels and returns a dict, nothing else.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(eval_pred) -> dict[str, float]:
    """Trainer metrics: accuracy plus macro precision/recall/F1.

    `eval_pred` is the `(logits, labels)` tuple Trainer passes in. Macro
    averaging weights FAKE and REAL equally, which is what we want on a class
    where missing fake news and false-flagging real news both matter.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
    }
