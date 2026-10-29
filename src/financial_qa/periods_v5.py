"""Recognize explicit subannual requests before applying annual comparison rules."""

import re

MONTH_NAMES = (
    ("january", "jan"),
    ("february", "feb"),
    ("march", "mar"),
    ("april", "apr"),
    ("may",),
    ("june", "jun"),
    ("july", "jul"),
    ("august", "aug"),
    ("september", "sept", "sep"),
    ("october", "oct"),
    ("november", "nov"),
    ("december", "dec"),
)
MONTH_NUMBERS = {name: number for number, names in enumerate(MONTH_NAMES, 1) for name in names}
MONTH_PATTERN = r"\b(?:" + "|".join(MONTH_NUMBERS) + r")\b"


def has_subannual_period(question: str) -> bool:
    """Detect periods that must never fall back to an unspecified prior year.

    Detection is deliberately broader than the compiler's supported named-month
    and annual-boundary pairs. Unsupported quarters and numeric dates must fail
    closed instead of silently becoming a year-on-year calculation.
    """
    return bool(
        re.search(
            MONTH_PATTERN + r"|\b(?:q[1-4]|h[12]|quarters?|quarterly|half.year|"
            r"beginning|opening|start|ending|closing|end)\b"
            + r"|\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b"
            + r"|\b\d{1,2}[-/]\d{1,2}[-/](?:19|20)\d{2}\b",
            question,
            re.I,
        )
    )
