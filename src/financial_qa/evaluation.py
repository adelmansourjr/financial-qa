"""Transparent numeric scoring, explicitly distinct from the official TAT-QA scorer."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
from typing import Any

import numpy as np

from .data import Example, parse_number
from .planner import Planner
from .service import answer

SCALES = {
    "": Decimal(1),
    "percent": Decimal("0.01"),
    "thousand": Decimal(1000),
    "million": Decimal(1_000_000),
    "billion": Decimal(1_000_000_000),
}


def numeric_match(
    predicted: str | None, predicted_scale: str, gold: object, gold_scale: str
) -> bool:
    expected = parse_number(str(gold))
    actual = parse_number(predicted) if predicted is not None else None
    if actual is None or expected is None:
        return False
    if predicted_scale not in SCALES or gold_scale not in SCALES:
        return False
    # Compare in the gold unit, with the dataset's customary two-decimal rounding.
    converted = actual * SCALES[predicted_scale] / SCALES[gold_scale]
    return abs(converted - expected) <= Decimal("0.005")


def evaluate(examples: list[Example], planner: Planner, limit: int = 100) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("Evaluation limit must be positive")
    eligible = [
        example
        for example in examples
        if example.answer_type == "arithmetic" and parse_number(str(example.answer)) is not None
    ][:limit]
    if not eligible:
        raise ValueError("Dataset contains no eligible arithmetic questions")
    records = []
    for example in eligible:
        prediction = answer(example.document, example.question, planner)
        scale = (prediction["plan"] or {}).get("scale", "")
        correct = numeric_match(prediction["answer"], scale, example.answer, example.scale)
        records.append(
            {
                "id": example.id,
                "document_id": example.document.id,
                "predicted": prediction["answer"],
                "predicted_scale": scale,
                "gold": example.answer,
                "gold_scale": example.scale,
                "correct": correct,
                "abstained": prediction["abstained"],
                "reason": prediction["reason"],
                "latency_ms": prediction["latency_ms"],
            }
        )
    answered = sum(not record["abstained"] for record in records)
    correct = sum(record["correct"] for record in records)
    return {
        "scorer": "numeric_with_scale_tolerance_0.005_in_gold_units; NOT official TAT-QA EM/F1",
        "selection": "first eligible arithmetic questions in pinned split; no gold-derived inputs",
        "dataset_examples": len(examples),
        "evaluated": len(records),
        "accuracy": correct / len(records),
        "coverage": answered / len(records),
        "answered_accuracy": correct / answered if answered else None,
        "p95_latency_ms": float(np.percentile([row["latency_ms"] for row in records], 95)),
        "abstention_reasons": dict(Counter(row["reason"] for row in records if row["abstained"])),
        "predictions": records,
    }
