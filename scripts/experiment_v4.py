"""Measure layout-aware proposals without reusing inspected QA test outcomes.

Development reuses the existing 100 training-development contexts. Freeze selects
200 previously unused public-training contexts, excluding every historical train
context. This is a project-held-out arithmetic study, not official test data or
an independent filing-level calibration certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np

from financial_qa import benchmark as statistics
from financial_qa import evaluation as scoring
from financial_qa.benchmark import interval
from financial_qa.calibration import file_sha256
from financial_qa.cli import load_runtime_profile
from financial_qa.data import load_examples, parse_number
from financial_qa.evaluation import numeric_match
from financial_qa.reasoning_v3 import ProgramPlanner, RuntimeProfile, runtime_signature
from financial_qa.reasoning_v4 import LayoutProgramPlanner
from financial_qa.service import answer
from financial_qa.storage import write_json

MANIFEST = Path("benchmarks/train-holdout-v4.json")
FREEZE = Path("benchmarks/policy-v4.json")
MODEL_SHA = "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"
MODEL_PATH = Path("local-runtime/qwen3-4b-q4.gguf")


def environment_identity() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "pydantic": version("pydantic"),
        "platform": platform.platform(),
    }


def verify_runtime(planner, *, verify_bytes: bool = False) -> None:
    """Check trusted local launch identity; metadata is not server attestation."""
    properties = planner.client.properties()
    if runtime_signature(properties) != profile().runtime:
        raise ValueError("Model server runtime differs from the recorded profile")
    aliases = properties.get("model_alias", [])
    if isinstance(aliases, str):
        aliases = [aliases]
    server_path = Path(properties.get("model_path", ""))
    if not server_path.is_absolute():
        server_path = Path.cwd() / server_path
    if server_path.resolve() != MODEL_PATH.resolve() or profile().model not in aliases:
        raise ValueError("Model path or alias differs from the trusted local launch")
    if verify_bytes and file_sha256(MODEL_PATH) != MODEL_SHA:
        raise ValueError("Actual model bytes differ from the pinned model")


def verify_freeze() -> tuple[dict, dict]:
    """Enforce source/data/scorer identity for held-out execution and summary."""
    frozen = json.loads(FREEZE.read_text())
    manifest = json.loads(MANIFEST.read_text())
    if (
        frozen["runner_sha256"] != file_sha256(Path(__file__))
        or frozen["manifest_sha256"] != file_sha256(MANIFEST)
        or manifest["data_sha256"] != file_sha256(Path("data/tatqa_train.json"))
        or frozen["criterion_sha256"] != file_sha256(Path(scoring.__file__))
        or frozen["statistics_sha256"] != file_sha256(Path(statistics.__file__))
        or frozen["environment"] != environment_identity()
        or any(frozen["methods"][method] != identity(method) for method in ("baseline", "layout"))
    ):
        raise ValueError("Source/data/scorer/policy changed after freeze")
    selected = manifest["examples"]
    if (
        len(selected) != 200
        or len({r["id"] for r in selected}) != 200
        or len({r["document_id"] for r in selected}) != 200
        or set(manifest["excluded_context_ids"]) & {r["document_id"] for r in selected}
    ):
        raise ValueError("Held-out membership/count/exclusion differs from protocol")
    return frozen, manifest


def profile() -> RuntimeProfile:
    return RuntimeProfile(
        "qwen3-4b-q4-nonthinking",
        MODEL_SHA,
        runtime=load_runtime_profile(Path("benchmarks/runtime-v3.json")),
    )


def planner_for(method: str, stage: str):
    cache = Path(f"artifacts/v4-{stage}-cache")
    if method == "layout":
        return LayoutProgramPlanner(profile(), cache_dir=cache)
    return ProgramPlanner(profile(), cache_dir=cache, policy="model")


def identity(method: str) -> dict:
    planner = planner_for(method, "identity")
    return planner.identity() if method == "layout" else profile().identity()


def freeze() -> None:
    """Select labels only by predeclared eligibility; freeze before any scoring."""
    if MANIFEST.exists() or FREEZE.exists():
        raise ValueError("Version4 manifest/policy already frozen; refusing replacement")
    development = json.loads(Path("reports/v4-development-layout.json").read_text())
    if development["evaluated"] != 100 or development["identity"] != identity("layout"):
        raise ValueError("Complete development under the final inference identity first")
    previous = json.loads(Path("benchmarks/train-v3-400.json").read_text())
    excluded = set(previous["previously_used_document_ids"])
    excluded.update(
        row["document_id"] for stage in ("development", "calibration") for row in previous[stage]
    )
    by_document = {}
    for example in load_examples(Path("data/tatqa_train.json")):
        if (
            example.document.id not in excluded
            and example.answer_type == "arithmetic"
            and parse_number(str(example.answer)) is not None
        ):
            by_document.setdefault(example.document.id, []).append(example)
    rng = random.Random(409)
    documents = sorted(by_document)
    rng.shuffle(documents)
    if len(documents) < 200:
        raise ValueError("Insufficient unused eligible contexts")
    selected = [
        rng.choice(sorted(by_document[doc], key=lambda item: item.id)) for doc in documents[:200]
    ]
    manifest = {
        "version": 4,
        "seed": 409,
        "scope": "200 project-held-out public-training contexts; one arithmetic question each",
        "unused_eligible_contexts_before_selection": len(documents),
        "excluded_context_ids": sorted(excluded),
        "examples": [{"id": item.id, "document_id": item.document.id} for item in selected],
        "data_sha256": file_sha256(Path("data/tatqa_train.json")),
    }
    write_json(MANIFEST, manifest)
    write_json(
        FREEZE,
        {
            "methods": {method: identity(method) for method in ("baseline", "layout")},
            "manifest_sha256": file_sha256(MANIFEST),
            "runner_sha256": file_sha256(Path(__file__)),
            "criterion": "Unconditional scale-aware numeric accuracy; tolerance0.005 in gold units",
            "criterion_sha256": file_sha256(Path(scoring.__file__)),
            "statistics_sha256": file_sha256(Path(statistics.__file__)),
            "environment": environment_identity(),
            "comparison": "Paired context bootstrap1000, accuracy layout minus frozen v3 model",
            "accuracy_acceptance": (
                "Positive unconditional accuracy gain with the paired95% interval above zero; "
                "a point gain alone is inconclusive"
            ),
            "selection": (
                "No model/prompt/threshold changes after scoring; no automatic default promotion"
            ),
            "risk_status": "Not a new risk calibration; v3 guarded mode remains unchanged",
            "limitations": [
                "Public training labels can be in model pretraining",
                "Context IDs do not establish unique independent filings",
                "Project-held-out arithmetic subset is not an official benchmark test",
            ],
        },
    )
    print(f"Frozen {len(selected)} unused contexts; no outcomes scored.")


def run(stage: str, method: str) -> None:
    output = Path(f"reports/v4-{stage}-{method}.json")
    checkpoint = Path(f"artifacts/v4-{stage}-{method}-checkpoint.json")
    if output.exists():
        raise ValueError("Refusing to overwrite an existing result")
    planner = planner_for(method, stage)
    cache_before = len(list(planner.cache_dir.glob("*.json")))
    expected = identity(method)
    verify_runtime(planner, verify_bytes=True)
    criterion = file_sha256(Path(scoring.__file__))
    data = Path("data/tatqa_train.json")
    if stage == "development":
        selected = json.loads(Path("benchmarks/train-v3-400.json").read_text())[stage]
    else:
        frozen, manifest = verify_freeze()
        selected = manifest["examples"]
    lookup = {item.id: item for item in load_examples(data)}
    records = []
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if (
            saved.get("criterion_sha256") != criterion
            or saved["identity"] != expected
            or [r["id"] for r in saved["predictions"]]
            != [r["id"] for r in selected[: len(saved["predictions"])]]
        ):
            raise ValueError("Checkpoint identity or membership differs")
        records = saved["predictions"]
    for row in selected[len(records) :]:
        if identity(method) != expected or file_sha256(Path(scoring.__file__)) != criterion:
            raise ValueError("Prediction source changed during run")
        example = lookup[row["id"]]
        if example.document.id != row["document_id"]:
            raise ValueError("Question/context membership differs")
        prediction = answer(example.document, example.question, planner)
        scale = (prediction["plan"] or {}).get("scale", "")
        records.append(
            {
                "id": example.id,
                "document_id": example.document.id,
                "correct": numeric_match(
                    prediction["answer"], scale, example.answer, example.scale
                ),
                "predicted": prediction["answer"],
                "predicted_scale": scale,
                "gold": example.answer,
                "gold_scale": example.scale,
                "abstained": prediction["abstained"],
                "reason": prediction["reason"],
                "plan": prediction["plan"],
                "latency_ms": prediction["latency_ms"],
            }
        )
        write_json(
            checkpoint,
            {"identity": expected, "criterion_sha256": criterion, "predictions": records},
        )
        if len(records) % 10 == 0 or len(records) == len(selected):
            verify_runtime(planner)
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "method": method,
                        "completed": len(records),
                        "total": len(selected),
                    }
                ),
                flush=True,
            )
    accepted = sum(not row["abstained"] for row in records)
    correct = sum(row["correct"] for row in records)
    report = {
        "scope": "Training-development only"
        if stage == "development"
        else json.loads(MANIFEST.read_text())["scope"],
        "identity": expected,
        "criterion_sha256": criterion,
        "freeze_sha256": file_sha256(FREEZE) if stage == "heldout" else None,
        "method": method,
        "backend": "layout-program-v4" if method == "layout" else "program-v3-model",
        "manifest_sha256": hashlib.sha256(
            json.dumps(json.loads(MANIFEST.read_text()), sort_keys=True).encode()
        ).hexdigest()
        if stage == "heldout"
        else None,
        "cache_entries_before": cache_before,
        "cache_entries_after": len(list(planner.cache_dir.glob("*.json"))),
        "new_cached_model_responses": len(list(planner.cache_dir.glob("*.json"))) - cache_before,
        "evaluated": len(records),
        "correct": correct,
        "answered": accepted,
        "accuracy": correct / len(records),
        "coverage": accepted / len(records),
        "answered_accuracy": correct / accepted if accepted else None,
        "p95_latency_ms": float(np.percentile([row["latency_ms"] for row in records], 95)),
        "latency_note": (
            "Run can resume/replay cached requests; not pure live latency unless cache starts empty"
        ),
        "predictions": records,
    }
    write_json(output, report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"predictions", "identity"}}, indent=2
        )
    )


def summarize() -> None:
    frozen, manifest = verify_freeze()
    source = {example.id: example for example in load_examples(Path("data/tatqa_train.json"))}
    reports = {
        method: json.loads(Path(f"reports/v4-heldout-{method}.json").read_text())
        for method in ("baseline", "layout")
    }
    rows = {method: report["predictions"] for method, report in reports.items()}
    for method, report in reports.items():
        records = report["predictions"]
        if (
            report["identity"] != frozen["methods"][method]
            or report["criterion_sha256"] != frozen["criterion_sha256"]
            or report["freeze_sha256"] != file_sha256(FREEZE)
            or report["evaluated"] != 200
            or [{"id": r["id"], "document_id": r["document_id"]} for r in records]
            != manifest["examples"]
        ):
            raise ValueError("Result identity/membership differs from frozen evaluation")
        if any(
            r["gold"] != source[r["id"]].answer or r["gold_scale"] != source[r["id"]].scale
            for r in records
        ):
            raise ValueError("Reported reference labels differ from pinned source")
        correct = sum(
            numeric_match(r["predicted"], r["predicted_scale"], r["gold"], r["gold_scale"])
            for r in records
        )
        if (
            any(
                type(r["correct"]) is not bool
                or type(r["abstained"]) is not bool
                or r["abstained"] != (r["predicted"] is None)
                for r in records
            )
            or correct != report["correct"]
            or correct != sum(r["correct"] for r in records)
            or report["accuracy"] != correct / 200
            or report["answered"] != sum(not r["abstained"] for r in records)
            or report["coverage"] != report["answered"] / 200
            or report["answered_accuracy"]
            != (correct / report["answered"] if report["answered"] else None)
        ):
            raise ValueError("Result aggregates differ from recomputed outcomes")
    if [r["id"] for r in rows["baseline"]] != [r["id"] for r in rows["layout"]]:
        raise ValueError("Paired outcomes differ in membership")
    differences = [
        float(b["correct"]) - float(a["correct"])
        for a, b in zip(rows["baseline"], rows["layout"], strict=True)
    ]
    result = {
        "scope": manifest["scope"],
        "freeze_sha256": file_sha256(FREEZE),
        "methods": {
            method: {k: v for k, v in report.items() if k not in {"identity", "predictions"}}
            for method, report in reports.items()
        },
        "paired_accuracy_gain": interval(differences, [r["document_id"] for r in rows["layout"]]),
        "accuracy_intervals": {
            method: interval(
                [float(r["correct"]) for r in records], [r["document_id"] for r in records]
            )
            for method, records in rows.items()
        },
        "limitations": frozen["limitations"],
        "risk_status": "No new calibration certificate; guarded v3 continues to abstain",
    }
    output = Path("reports/v4-summary.json")
    if output.exists():
        raise ValueError("Refusing to replace a completed summary")
    write_json(output, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["development", "freeze", "heldout", "summarize"])
    parser.add_argument("--method", choices=["baseline", "layout"], default="layout")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    elif args.stage == "summarize":
        summarize()
    else:
        run(args.stage, args.method)


if __name__ == "__main__":
    main()
