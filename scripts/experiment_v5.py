"""Development-first v5 comparison on prospectively reserved source groups.

Never tune on completed evaluation outcomes. Development reuses the original100
contexts; the new paired sample is reserved only after the candidate is fixed.
Reports and checkpoints bind inference, source, scorer, runner and membership.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from financial_qa import benchmark, evaluation
from financial_qa.calibration import file_sha256
from financial_qa.data import load_examples
from financial_qa.evaluation import numeric_match
from financial_qa.reasoning_v4 import LayoutProgramPlanner
from financial_qa.reasoning_v5 import CandidateProgramPlanner
from financial_qa.service import answer
from financial_qa.source_reservations import freeze_reservation, validate_reservation
from financial_qa.storage import write_json

DATA = Path("data/tatqa_train.json")
MEMBERSHIP = Path("benchmarks/future/v5-comparison.json")
POLICY = Path("benchmarks/policy-v5.json")
SIZE, SEED = 150, 202610547


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def engine(method, stage):
    cls = CandidateProgramPlanner if method == "candidate" else LayoutProgramPlanner
    return cls(module("experiment_v4").profile(), cache_dir=Path(f"artifacts/v5-{stage}-cache"))


def identity() -> dict:
    return {
        "methods": {
            name: engine(name, "identity").identity() for name in ("baseline", "candidate")
        },
        "data_sha256": file_sha256(DATA),
        "scorer_sha256": file_sha256(Path(evaluation.__file__)),
        "statistics_sha256": file_sha256(Path(benchmark.__file__)),
        "runner_sha256": file_sha256(Path(__file__)),
        "runtime_driver_sha256": file_sha256(Path(__file__).with_name("experiment_v4.py")),
        "environment": module("experiment_v4").environment_identity(),
    }


def validate_rows(rows, selected, lookup, *, complete=False):
    expected = [{"id": r["id"], "document_id": r["document_id"]} for r in selected]
    if (complete and len(rows) != len(expected)) or len(rows) > len(expected):
        raise ValueError("Prediction count differs from frozen membership")
    if [{"id": r["id"], "document_id": r["document_id"]} for r in rows] != expected[: len(rows)]:
        raise ValueError("Prediction membership differs from frozen order")
    for row in rows:
        source = lookup[row["id"]]
        if (
            source.document.id != row["document_id"]
            or row["gold"] != source.answer
            or row["gold_scale"] != source.scale
            or type(row["correct"]) is not bool
            or type(row["abstained"]) is not bool
            or row["abstained"] != (row["predicted"] is None)
            or row["correct"]
            != numeric_match(row["predicted"], row["predicted_scale"], source.answer, source.scale)
            or type(row["latency_ms"]) not in (int, float)
            or not np.isfinite(row["latency_ms"])
            or row["latency_ms"] < 0
        ):
            raise ValueError("Prediction labels/correctness/availability/latency are inconsistent")


def validate_report(report, stage, method, selected, lookup, expected):
    rows = report["predictions"]
    validate_rows(rows, selected, lookup, complete=True)
    correct = sum(r["correct"] for r in rows)
    answered = sum(not r["abstained"] for r in rows)
    if (
        report["identity"] != expected
        or report["method"] != method
        or report["stage"] != stage
        or type(report["evaluated"]) is not int
        or report["evaluated"] != len(rows)
        or type(report["correct"]) is not int
        or report["correct"] != correct
        or type(report["answered"]) is not int
        or report["answered"] != answered
        or report["accuracy"] != correct / len(rows)
        or report["coverage"] != answered / len(rows)
        or report["answered_accuracy"] != (correct / answered if answered else None)
        or report["p95_latency_ms"] != float(np.percentile([r["latency_ms"] for r in rows], 95))
    ):
        raise ValueError("Report stage/method/aggregates differ from outcomes")


def source_membership(freeze=False):
    guard = module("audit_leakage")
    documents, stages, source_identity = guard.sources()
    examples = load_examples(DATA)
    consumed = {uid for ids in stages.values() for uid in ids}
    # This experiment is a new registered consumer. Unknown manual/legacy model
    # calls still require updating the explicit consumption ledger first.
    entries = guard.read_reservations(guard.LEDGER, source_identity, examples, documents, consumed)
    all_stages = {
        **stages,
        **{
            f"reserved_{e['index']}": [r["document_id"] for r in e["sampling"]["examples"]]
            for e in entries
        },
    }
    guard.verify_consumed_ledger(all_stages)
    operation = freeze_reservation if freeze else validate_reservation
    return operation(
        guard.LEDGER, MEMBERSHIP, source_identity, examples, documents, consumed, SIZE, SEED
    )


def policy_contract(current: dict, reservation_sha256: str, reservation_entry_sha256: str) -> dict:
    """Define the complete prospective contract in code, including stopping rules.

    Its containing runner is fingerprinted before evaluation. Exact equality
    rejects changed thresholds, omitted limitations, and unrecognized additions.
    """
    return {
        "identity": current,
        "reservation_sha256": reservation_sha256,
        "reservation_entry_sha256": reservation_entry_sha256,
        "size": SIZE,
        "seed": SEED,
        "scope": f"{SIZE} prospectively reserved source-group-disjoint public-training contexts",
        "primary": "Unconditional scale-aware numeric accuracy; abstentions count incorrect",
        "accuracy_target": 0.80,
        "improvement_target": "Paired context-bootstrap95% accuracy-gain interval above zero",
        "selection": "No inference changes or repeated sample selection after outcomes",
        "secondary": "Coverage, accuracy among answered questions and live end-to-end latency",
        "limitations": [
            "Exact source groups are conservative proxies, not verified independent filings",
            "Public data may be in model pretraining",
            "No new accepted-error certificate",
        ],
    }


def freeze() -> None:
    if POLICY.exists() or MEMBERSHIP.exists():
        raise ValueError("Preserve prospective membership and policy; refusing replacement")
    development = json.loads(Path("reports/v5-development-candidate.json").read_text())
    current = identity()
    selected = json.loads(Path("benchmarks/train-v3-400.json").read_text())["development"]
    lookup = {e.id: e for e in load_examples(DATA)}
    validate_report(development, "development", "candidate", selected, lookup, current)
    if len(selected) != 100:
        raise ValueError("Original development membership must contain100 contexts")
    reserved = source_membership(freeze=True)
    write_json(
        POLICY,
        policy_contract(current, file_sha256(MEMBERSHIP), reserved["entry_sha256"]),
    )
    print("Frozen source-group comparison; no evaluation outcomes scored", flush=True)


def verify_freeze():
    frozen = json.loads(POLICY.read_text())
    current = identity()
    reservation_sha256 = file_sha256(MEMBERSHIP)
    if frozen.get("identity") != current or frozen.get("reservation_sha256") != reservation_sha256:
        raise ValueError("Frozen inference/scorer/environment/source identity changed")
    membership = source_membership()
    if frozen != policy_contract(current, reservation_sha256, membership["entry_sha256"]):
        raise ValueError("Frozen prospective policy contract changed")
    return frozen, membership["sampling"]["examples"]


def run(stage: str, method: str) -> None:
    output = Path(f"reports/v5-{stage}-{method}.json")
    checkpoint = Path(f"artifacts/v5-{stage}-{method}-checkpoint.json")
    if output.exists():
        raise ValueError("Preserve completed outcomes; refusing replacement")
    frozen, selected = (
        verify_freeze()
        if stage == "comparison"
        else (None, json.loads(Path("benchmarks/train-v3-400.json").read_text())["development"])
    )
    policy_sha256 = file_sha256(POLICY) if frozen else None
    expected = identity()
    selected_identity = [{"id": r["id"], "document_id": r["document_id"]} for r in selected]
    lookup = {e.id: e for e in load_examples(DATA)}
    planner = engine(method, stage)
    runtime = module("experiment_v4")
    runtime.verify_runtime(planner, verify_bytes=True)
    cached_before = len(list(planner.cache_dir.glob("*.json")))
    rows = []
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved["identity"] != expected or saved["membership"] != selected_identity:
            raise ValueError("Checkpoint inference or membership changed")
        rows = saved["predictions"]
        validate_rows(rows, selected, lookup)
        if [{"id": r["id"], "document_id": r["document_id"]} for r in rows] != selected_identity[
            : len(rows)
        ]:
            raise ValueError("Checkpoint prediction prefix differs")
    for selection in selected[len(rows) :]:
        if identity() != expected:
            raise ValueError("Inference/scorer/environment changed during evaluation")
        example = lookup[selection["id"]]
        if example.document.id != selection["document_id"]:
            raise ValueError("Question does not belong to selected source context")
        prediction = answer(example.document, example.question, planner)
        scale = (prediction["plan"] or {}).get("scale", "")
        rows.append(
            {
                "id": example.id,
                "document_id": example.document.id,
                "predicted": prediction["answer"],
                "predicted_scale": scale,
                "correct": numeric_match(
                    prediction["answer"], scale, example.answer, example.scale
                ),
                "gold": example.answer,
                "gold_scale": example.scale,
                "abstained": prediction["abstained"],
                "reason": prediction["reason"],
                "plan": prediction["plan"],
                "latency_ms": prediction["latency_ms"],
            }
        )
        write_json(
            checkpoint, {"identity": expected, "membership": selected_identity, "predictions": rows}
        )
        if len(rows) % 5 == 0:
            runtime.verify_runtime(planner)
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "method": method,
                        "completed": len(rows),
                        "total": len(selected),
                        "correct": sum(r["correct"] for r in rows),
                    }
                ),
                flush=True,
            )
    if identity() != expected:
        raise ValueError("Inference identity changed before finalizing results")
    runtime.verify_runtime(planner)
    validate_rows(rows, selected, lookup, complete=True)
    if frozen:
        # Revalidate after the last request/runtime check, including resumed runs
        # that produce no new predictions. Never bind an edited ending policy.
        final_frozen, final_selected = verify_freeze()
        if (
            final_frozen != frozen
            or final_selected != selected
            or file_sha256(POLICY) != policy_sha256
        ):
            raise ValueError("Frozen policy or source membership changed during evaluation")
    correct, answered = sum(r["correct"] for r in rows), sum(not r["abstained"] for r in rows)
    write_json(
        output,
        {
            "identity": expected,
            "method": method,
            "stage": stage,
            "scope": frozen["scope"]
            if frozen
            else "Original100 reused training-development contexts",
            "policy_sha256": policy_sha256,
            "evaluated": len(rows),
            "correct": correct,
            "answered": answered,
            "accuracy": correct / len(rows),
            "coverage": answered / len(rows),
            "answered_accuracy": correct / answered if answered else None,
            "p95_latency_ms": float(np.percentile([r["latency_ms"] for r in rows], 95)),
            "cached_responses_before": cached_before,
            "cached_responses_after": len(list(planner.cache_dir.glob("*.json"))),
            "latency_scope": "Includes cache replay after resume; new requests counted separately",
            "predictions": rows,
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "correct": correct,
                "evaluated": len(rows),
                "answered": answered,
            }
        ),
        flush=True,
    )


def summarize() -> None:
    frozen, selected = verify_freeze()
    policy_sha256 = file_sha256(POLICY)
    output = Path("reports/v5-summary.json")
    if output.exists():
        raise ValueError("Preserve completed comparison summary")
    reports = {
        name: json.loads(Path(f"reports/v5-comparison-{name}.json").read_text())
        for name in ("baseline", "candidate")
    }
    lookup = {e.id: e for e in load_examples(DATA)}
    expected_rows = [{"id": r["id"], "document_id": r["document_id"]} for r in selected]
    for method, report in reports.items():
        validate_report(report, "comparison", method, selected, lookup, frozen["identity"])
        rows = report["predictions"]
        if report["identity"] != frozen["identity"] or report["policy_sha256"] != policy_sha256:
            raise ValueError("Prediction report differs from frozen identity")
        if [{"id": r["id"], "document_id": r["document_id"]} for r in rows] != expected_rows:
            raise ValueError("Comparison membership differs")
        if any(
            type(r["correct"]) is not bool
            or type(r["abstained"]) is not bool
            or r["gold"] != lookup[r["id"]].answer
            or r["gold_scale"] != lookup[r["id"]].scale
            or r["abstained"] != (r["predicted"] is None)
            or r["correct"]
            != numeric_match(r["predicted"], r["predicted_scale"], r["gold"], r["gold_scale"])
            for r in rows
        ):
            raise ValueError("Prediction correctness/labels are inconsistent")
        if (
            report["correct"] != sum(r["correct"] for r in rows)
            or report["accuracy"] != report["correct"] / SIZE
        ):
            raise ValueError("Reported aggregate differs from outcomes")
    differences = [
        float(b["correct"]) - float(a["correct"])
        for a, b in zip(
            reports["baseline"]["predictions"], reports["candidate"]["predictions"], strict=True
        )
    ]
    gain = benchmark.interval(
        differences, [r["document_id"] for r in reports["candidate"]["predictions"]]
    )
    result = {
        "scope": frozen["scope"],
        "policy_sha256": policy_sha256,
        "methods": {
            name: {k: v for k, v in report.items() if k not in {"identity", "predictions"}}
            for name, report in reports.items()
        },
        "paired_accuracy_gain": gain,
        "accuracy_target": frozen["accuracy_target"],
        "accuracy_target_met": reports["candidate"]["accuracy"] >= frozen["accuracy_target"],
        "improvement_target_met": gain["ci95"][0] > 0,
        "limitations": frozen["limitations"],
    }
    final_frozen, final_selected = verify_freeze()
    if final_frozen != frozen or final_selected != selected or file_sha256(POLICY) != policy_sha256:
        raise ValueError("Frozen policy or source membership changed during summary")
    write_json(output, result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["development", "freeze", "comparison", "summarize"])
    parser.add_argument("--method", choices=["baseline", "candidate"], default="candidate")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    elif args.stage == "summarize":
        summarize()
    else:
        run(args.stage, args.method)


if __name__ == "__main__":
    main()
