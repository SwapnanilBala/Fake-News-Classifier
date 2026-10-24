"""Label definitions for the LIAR -> binary fake-news task.

LIAR ships a 6-way truthfulness rating. We collapse it to a binary
FAKE/REAL problem and drop the ambiguous middle (barely-true, half-true).

LIAR labels are strings in the source TSVs (data/liar/):
    pants-fire   -> FAKE
    false        -> FAKE
    barely-true  -> DROP (too ambiguous)
    half-true    -> DROP (too ambiguous)
    mostly-true  -> REAL
    true         -> REAL
"""

# Binary class ids used everywhere downstream.
FAKE = 0
REAL = 1

# Human-readable names, indexed by binary class id. Used for report
# target_names, confusion-matrix tick labels, and API responses.
LABEL_NAMES = ["FAKE", "REAL"]

# 6-way LIAR label -> binary id, or None for the ambiguous middle that we drop.
# Keyed by name, so an unknown label raises KeyError instead of mis-mapping.
LIAR_TO_BINARY: dict[str, int | None] = {
    "pants-fire": FAKE,
    "false": FAKE,
    "barely-true": None,
    "half-true": None,
    "mostly-true": REAL,
    "true": REAL,
}
