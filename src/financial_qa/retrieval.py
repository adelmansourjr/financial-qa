"""Small BM25 retriever: no model weights or fitted test-label features."""

from __future__ import annotations

import math
import re
from collections import Counter

from .data import Document, Evidence

STOPWORDS = frozenset(
    "the a an of in on for what was were is are how much did to and by from".split()
)


def tokens(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOPWORDS]


def retrieve(
    document: Document, question: str, top_k: int = 6, mode: str = "legacy"
) -> list[Evidence]:
    if not 1 <= top_k <= 30:
        raise ValueError("top_k must be between 1 and 30")
    evidence = document.evidence()
    if not evidence:
        return []
    documents = [Counter(tokens(item.text)) for item in evidence]
    average_length = sum(sum(item.values()) for item in documents) / len(documents) or 1
    scores = []
    for index, counts in enumerate(documents):
        score = 0.0
        for term in set(tokens(question)):
            frequency = counts[term]
            containing = sum(term in other for other in documents)
            inverse = math.log(1 + (len(documents) - containing + 0.5) / (containing + 0.5))
            denominator = frequency + 1.2 * (0.25 + 0.75 * sum(counts.values()) / average_length)
            score += inverse * frequency * 2.2 / denominator
        scores.append((score, index))
    ranked = sorted(scores, key=lambda pair: (-pair[0], pair[1]))
    if mode == "complete":
        # Small financial tables fit the bounded catalog; retain all rows so
        # relationships and subtotals cannot disappear through lexical ranking.
        return evidence
    if mode == "coherent":
        from .context import table_context

        headers = set(table_context(document).header_rows)
        # Preserve headers separately, so several header rows cannot consume the evidence budget.
        body = [index for _, index in ranked if index not in headers][:top_k]
        chosen = sorted(headers | set(body))
        return [evidence[index] for index in chosen]
    chosen = [index for _, index in ranked[:top_k]]
    # Preserve the column header so selected numbers retain their year/unit context.
    if evidence[0].id == "R0" and 0 not in chosen:
        chosen = [0] + chosen[: top_k - 1]
    return [evidence[index] for index in chosen]
