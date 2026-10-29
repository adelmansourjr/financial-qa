"""Shared answer path for the CLI and HTTP API."""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

from .data import Document
from .executor import ExecutionError, execute
from .planner import Planner
from .retrieval import retrieve


def answer(document: Document, question: str, planner: Planner, top_k: int = 6) -> dict[str, Any]:
    start = time.perf_counter()
    evidence = retrieve(document, question, top_k, getattr(planner, "retrieval_mode", "legacy"))
    plan = None
    result: Decimal | None = None
    reason = ""
    try:
        plan = planner.plan(document, question, evidence)
        if plan.abstain:
            reason = plan.reason or "Insufficient evidence"
        else:
            result = execute(plan, evidence)
    except ExecutionError as error:
        reason = str(error)
    return {
        "answer": str(result) if result is not None else None,
        "abstained": result is None,
        "reason": reason,
        "plan": plan.model_dump() if plan else None,
        "evidence": [
            {
                "id": item.id,
                "text": item.text,
                "variables": {key: str(value) for key, value in item.variables.items()},
            }
            for item in evidence
        ],
        "latency_ms": round((time.perf_counter() - start) * 1000, 3),
        "validation": "Arithmetic and citation checks do not establish semantic correctness.",
    }
