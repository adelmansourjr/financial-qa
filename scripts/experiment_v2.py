"""Frozen selective-v2 experiment; training diagnostics and blind final scoring.

Run from the repository directory. No sampling or policy fitting occurs during
final evaluation. Existing reports are not overwritten without explicit paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from financial_qa.benchmark import compare, eligible, evaluate_manifest, interval
from financial_qa.cli import make_planner
from financial_qa.data import load_examples
from financial_qa.storage import write_json


def freeze() -> None:
    """Exclude every publicly explored legacy/v1/demo document before scoring."""
    target = Path("benchmarks/dev-holdout-v2.json")
    if target.exists():
        raise ValueError("Refusing to replace an existing frozen manifest")
    examples = load_examples(Path("data/tatqa_dev.json"))
    prior = json.loads(Path("benchmarks/dev-holdout-v1.json").read_text())
    excluded = set(prior["excluded_document_ids"])
    excluded.update(row["document_id"] for row in prior["examples"])
    demos = [x for x in examples if x.answer_type == "arithmetic"][:40]
    excluded.update(x.document.id for x in demos)
    groups = {}
    for example in eligible(examples):
        if example.document.id not in excluded:
            groups.setdefault(example.document.id, []).append(example)
    rng = np.random.default_rng(211)
    documents = rng.permutation(sorted(groups))[:100]
    chosen = [groups[doc][int(rng.integers(len(groups[doc])))] for doc in documents]
    write_json(
        target,
        {
            "version": 2,
            "seed": 211,
            "size": len(chosen),
            "available_documents": len(groups),
            "protocol": (
                "One numeric arithmetic question per document, seeded sample. "
                "Excludes all original100 numeric arithmetic documents, all v1 "
                "documents, all40 UI demo documents. Frozen before final scoring."
            ),
            "status": "Public development evaluation; pretraining contamination remains possible.",
            "excluded_document_ids": sorted(excluded),
            "examples": [{"id": x.id, "document_id": x.document.id} for x in chosen],
        },
    )
    print(f"Frozen {len(chosen)} documents; {len(excluded)} excluded")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["freeze", "development", "validation", "final", "compare"]
    )
    parser.add_argument("--backend", default="selective")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
        return
    if args.stage == "compare":
        reports = {
            name: json.loads(Path(f"reports/v2-final-{name}.json").read_text())
            for name in ("baseline", "header", "semantic", "selective", "hybrid", "agreement")
        }
        write_json(
            args.output or Path("reports/v2-comparison.json"),
            {
                "comparisons": [
                    compare(reports["header"], report)
                    for name, report in reports.items()
                    if name != "header"
                ],
                "note": (
                    "Paired document bootstrap; local numeric scorer, not official TATQA. "
                    "All methods use the same frozen100 documents."
                ),
            },
        )
        return
    output = args.output or Path(f"reports/v2-{args.stage}-{args.backend}.json")
    if output.exists():
        raise ValueError("Report already exists; use --output with a new replay path")
    cache_path = Path(f"artifacts/v2-{args.stage}-cache")
    cache_before = len(list(cache_path.glob("*.json")))
    if args.stage == "final":
        examples = load_examples(Path("data/tatqa_dev.json"))
        manifest = json.loads(Path("benchmarks/dev-holdout-v2.json").read_text())
    else:
        examples = load_examples(Path("data/tatqa_train.json"))
        selected = json.loads(Path("benchmarks/train-v2.json").read_text())[args.stage]
        # Training selection is intentionally separate from dev-only manifest
        # validation. It is never presented as the blind final evaluation.
        manifest = {"examples": selected}
    planner = make_planner(
        args.backend,
        "http://127.0.0.1:18080/v1",
        "qwen2.5-1.5b-q4",
        cache_path,
    )
    if args.stage != "final":
        # Training records can overlap legacy positions in their own split.
        # Direct scoring uses the same service/scorer and aggregation, without
        # pretending the dev manifest protocol applies to this training sample.
        from collections import Counter

        from financial_qa.evaluation import numeric_match
        from financial_qa.service import answer

        lookup = {x.id: x for x in examples}
        records = []
        for index, row in enumerate(manifest["examples"]):
            x = lookup[row["id"]]
            prediction = answer(x.document, x.question, planner)
            scale = (prediction["plan"] or {}).get("scale", "")
            records.append(
                {
                    "id": x.id,
                    "document_id": x.document.id,
                    "correct": numeric_match(prediction["answer"], scale, x.answer, x.scale),
                    "abstained": prediction["abstained"],
                    "predicted": prediction["answer"],
                    "predicted_scale": scale,
                    "gold": x.answer,
                    "gold_scale": x.scale,
                    "reason": prediction["reason"],
                    "plan": prediction["plan"],
                    "latency_ms": prediction["latency_ms"],
                }
            )
            print(
                f"{args.stage} {args.backend} {index + 1}/{len(manifest['examples'])}", flush=True
            )
        accepted = sum(not row["abstained"] for row in records)
        report = {
            "backend": args.backend,
            "evaluated": len(records),
            "accuracy": sum(row["correct"] for row in records) / len(records),
            "coverage": accepted / len(records),
            "answered_accuracy": sum(row["correct"] for row in records) / accepted
            if accepted
            else None,
            "reasons": dict(Counter(row["reason"] for row in records if row["abstained"])),
            "accuracy_interval": interval(
                [float(row["correct"]) for row in records], [row["document_id"] for row in records]
            ),
            "predictions": records,
        }
    else:
        report = evaluate_manifest(examples, planner, manifest)
    report["protocol"] = "selective-v2, training-only choices; local scale-aware numeric scorer"
    report.update(
        {
            "backend": args.backend,
            "planner_class": type(planner).__name__,
            "cache_entries_before": cache_before,
            "cache_entries_after": len(list(cache_path.glob("*.json"))),
            "latency_status": "actual deterministic execution"
            if args.backend in {"baseline", "header", "semantic"}
            else "live CPU inference"
            if cache_before == 0
            else "cache replay or mixed cache/live requests; not a pure live inference latency",
            "scorer": (
                "Local scale-aware numeric tolerance0.005 in gold units; not official TATQA EM/F1"
            ),
        }
    )
    write_json(output, report)
    print(
        json.dumps({key: value for key, value in report.items() if key != "predictions"}, indent=2)
    )


if __name__ == "__main__":
    main()
