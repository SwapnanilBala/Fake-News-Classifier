"""Label definitions for the LIAR -> binary fake-news task.

LIAR ships a 6-way truthfulness rating. We collapse it to a binary
FAKE/REAL problem and drop the ambiguous middle (barely-true, half-true).

Original LIAR label ids:
    0 = pants-fire   -> FAKE
    1 = false        -> FAKE
    2 = barely-true  -> DROP (too ambiguous)
    3 = half-true    -> DROP (too ambiguous)
    4 = mostly-true  -> REAL
    5 = true         -> REAL
"""

# Binary class ids used everywhere downstream.
FAKE = 0
REAL = 1

# Human-readable names, indexed by binary class id. Used for report
# target_names, confusion-matrix tick labels, and API responses.
LABEL_NAMES = ["FAKE", "REAL"]

# 6-way LIAR id -> binary id, or None for the ambiguous middle that we drop.
LIAR_TO_BINARY: dict[int, int | None] = {
    0: FAKE,
    1: FAKE,
    2: None,
    3: None,
    4: REAL,
    5: REAL,
}
