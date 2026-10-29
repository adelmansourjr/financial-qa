"""Frozen document-disjoint evaluation and paired document-bootstrap comparisons."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .data import Example, parse_number
from .evaluation import numeric_match
from .planner import Planner
from .service import answer
from .storage import write_json


def eligible(examples: list[Example]) -> list[Example]:
    return [
        example
        for example in examples
        if example.answer_type == "arithmetic" and parse_number(str(example.answer)) is not None
    ]


def freeze_manifest(examples: list[Example], path: Path, size: int = 50, seed: int = 71) -> dict:
    if size < 1:
        raise ValueError("Manifest size must be positive")
    if path.exists():
        manifest = json.loads(path.read_text())
        validate_manifest(examples, manifest, size, seed)
        return manifest
    old = eligible(examples)[:100]
    excluded = {example.document.id for example in old}
    groups = {}
    for example in eligible(examples):
        if example.document.id not in excluded:
            groups.setdefault(example.document.id, []).append(example)
    if len(groups) < size:
        raise ValueError("Not enough previously unused documents")
    rng = np.random.default_rng(seed)
    documents = rng.permutation(sorted(groups))[:size]
    chosen = [groups[doc][int(rng.integers(len(groups[doc])))] for doc in documents]
    manifest = {
        "version": 1,
        "seed": seed,
        "size": size,
        "protocol": (
            "One numeric arithmetic question per seeded sampled document; "
            "all documents of the original first100 arithmetic development questions excluded."
        ),
        "status": (
            "Development holdout from system tuning; "
            "public benchmark contamination remains possible."
        ),
        "excluded_document_ids": sorted(excluded),
        "examples": [{"id": example.id, "document_id": example.document.id} for example in chosen],
    }
    write_json(path, manifest)
    return manifest


def validate_manifest(examples: list[Example], manifest: dict, size: int, seed: int) -> None:
    if manifest.get("version", 1) not in {1, 2, 3}:
        raise ValueError("Unsupported frozen manifest version")
    selected = manifest.get("examples", [])
    ids = [row["id"] for row in selected]
    docs = [row["document_id"] for row in selected]
    expected_excluded = {example.document.id for example in eligible(examples)[:100]}
    lookup = {example.id: example.document.id for example in eligible(examples)}
    if size < 1 or manifest.get("size") != size or manifest.get("seed") != seed:
        raise ValueError("Manifest size or seed does not match the frozen protocol")
    if len(ids) != size or len(set(ids)) != size or len(set(docs)) != size:
        raise ValueError("Manifest must contain unique questions and documents")
    excluded = set(manifest.get("excluded_document_ids", []))
    if manifest.get("version", 1) in {2, 3}:
        # Later versions expand exclusions after prior sets were inspected; they cannot remove
        # any legacy document. Its exact frozen artifact hash is in each report.
        if not expected_excluded <= excluded:
            raise ValueError("Later manifests must exclude every legacy evaluation document")
        expected_excluded = excluded
    if excluded != expected_excluded:
        raise ValueError("Manifest exclusion set differs from the legacy evaluation documents")
    if set(docs) & expected_excluded or any(
        lookup.get(identifier) != doc for identifier, doc in zip(ids, docs, strict=True)
    ):
        raise ValueError("Manifest contains unknown, excluded, or mismatched examples")


def interval(values: list[float], groups: list[str], seed: int = 71) -> dict:
    if not values or len(values) != len(groups):
        raise ValueError("Need nonempty values and matching document groups")
    unique = sorted(set(groups))
    grouped = {
        group: [value for value, document in zip(values, groups, strict=True) if document == group]
        for group in unique
    }
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(1000):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        samples.append(np.mean([value for group in chosen for value in grouped[group]]))
    return {
        "mean": float(np.mean(values)),
        "ci95": list(map(float, np.quantile(samples, [0.025, 0.975]))),
        "bootstrap_unit": "document",
        "replicates": 1000,
    }


def evaluate_manifest(examples: list[Example], planner: Planner, manifest: dict) -> dict:
    validate_manifest(examples, manifest, manifest["size"], manifest["seed"])
    lookup = {example.id: example for example in examples}
    records = []
    for index, selected in enumerate(manifest["examples"]):
        example = lookup[selected["id"]]
        if example.document.id != selected["document_id"]:
            raise ValueError("Manifest document identity mismatch")
        prediction = answer(example.document, example.question, planner)
        scale = (prediction["plan"] or {}).get("scale", "")
        correct = numeric_match(prediction["answer"], scale, example.answer, example.scale)
        failure = (
            "correct"
            if correct
            else "abstained"
            if prediction["abstained"]
            else "wrong_unit_only"
            if numeric_match(prediction["answer"], example.scale, example.answer, example.scale)
            else "wrong_value_or_operation"
        )
        records.append(
            {
                "id": example.id,
                "document_id": example.document.id,
                "correct": correct,
                "predicted": prediction["answer"],
                "predicted_scale": scale,
                "gold": example.answer,
                "gold_scale": example.scale,
                "abstained": prediction["abstained"],
                "failure": failure,
                "reason": prediction["reason"],
                "plan": prediction["plan"],
                "latency_ms": prediction["latency_ms"],
            }
        )
        print(
            json.dumps(
                {
                    "completed": index + 1,
                    "total": len(manifest["examples"]),
                    "backend": type(planner).__name__,
                    "correct": correct,
                    "abstained": prediction["abstained"],
                }
            ),
            flush=True,
        )
    answered = sum(not record["abstained"] for record in records)
    return {
        "backend": type(planner).__name__,
        "scorer": "Local scale-aware numeric tolerance; not official TAT-QA EM/F1",
        "evaluated": len(records),
        "accuracy": sum(record["correct"] for record in records) / len(records),
        "accuracy_interval": interval(
            [float(row["correct"]) for row in records], [row["document_id"] for row in records]
        ),
        "coverage": answered / len(records),
        "coverage_interval": interval(
            [float(not row["abstained"]) for row in records],
            [row["document_id"] for row in records],
        ),
        "answered_accuracy_interval": interval(
            [float(row["correct"]) for row in records if not row["abstained"]],
            [row["document_id"] for row in records if not row["abstained"]],
        )
        if answered
        else None,
        "answered_accuracy": sum(record["correct"] for record in records) / answered
        if answered
        else None,
        "failures": dict(Counter(row["failure"] for row in records)),
        "abstention_reasons": dict(Counter(row["reason"] for row in records if row["abstained"])),
        "p95_latency_ms": float(np.percentile([row["latency_ms"] for row in records], 95)),
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "predictions": records,
    }


def compare(first: dict, second: dict) -> dict:
    baseline = {row["id"]: row for row in first["predictions"]}
    if set(baseline) != {row["id"] for row in second["predictions"]}:
        raise ValueError("Paired comparison requires identical question sets")
    differences = [
        float(row["correct"]) - float(baseline[row["id"]]["correct"])
        for row in second["predictions"]
    ]
    return {
        "first": first["backend"],
        "second": second["backend"],
        "second_minus_first_accuracy": interval(
            differences, [row["document_id"] for row in second["predictions"]]
        ),
    }
