"""Freeze and test one training-developed policy on300 fresh reference contexts.

Run freeze before opening calibration outcomes. All contexts exclude the608
historical training contexts and200 new comparison contexts. Calibration is a
single fixed test, not a policy/threshold search or permission to retune failures.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

from financial_qa import evaluation as scoring
from financial_qa.calibration import file_sha256
from financial_qa.data import load_examples, parse_number
from financial_qa.evaluation import numeric_match
from financial_qa.executor import ExecutionError
from financial_qa.program_cache_v4 import read_program_cache
from financial_qa.reasoning_v4 import LayoutProgramPlanner
from financial_qa.selective_v4 import (
    CALIBRATION_CONTEXTS,
    acceptance_plan,
    certificate_from_evidence,
    eligible_question,
    policy_identity,
)
from financial_qa.service import answer
from financial_qa.storage import write_json

MANIFEST = Path("benchmarks/calibration-v4.json")
FREEZE = Path("benchmarks/calibration-policy-v4.json")
REPORT = Path("reports/v4-temporal-calibration.json")
CERTIFICATE = Path("reports/v4-temporal-certificate.json")
CHECKPOINT = Path("artifacts/v4-temporal-calibration-checkpoint.json")
DATA = Path("data/tatqa_train.json")


def experiment_module():
    path = Path(__file__).with_name("experiment_v4.py")
    spec = importlib.util.spec_from_file_location("frozen_v4_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def engine() -> LayoutProgramPlanner:
    return LayoutProgramPlanner(
        experiment_module().profile(), cache_dir=Path("artifacts/v4-calibration-cache")
    )


def identity() -> dict:
    return {
        **policy_identity(engine()),
        "manifest_sha256": file_sha256(MANIFEST),
        "data_sha256": file_sha256(DATA),
    }


def read_checkpoint(path: Path) -> dict:
    """Full300-row checkpoints need a larger bounded budget than raw responses."""
    with path.open("rb") as stream:
        raw = stream.read(256_001)
    if len(raw) > 256_000:
        raise ValueError("Calibration checkpoint exceeds the metadata budget")
    try:
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("Expected a checkpoint object")
        return result
    except (ValueError, RecursionError) as error:
        raise ValueError("Calibration checkpoint is malformed") from error


def freeze() -> None:
    if MANIFEST.exists() or FREEZE.exists():
        raise ValueError("Preserve the predeclared calibration files; already frozen")
    experiment = experiment_module()
    _, comparison = experiment.verify_freeze()
    exclusions = set(comparison["excluded_context_ids"])
    exclusions.update(row["document_id"] for row in comparison["examples"])
    grouped = {}
    for example in load_examples(DATA):
        if (
            example.document.id not in exclusions
            and example.answer_type == "arithmetic"
            and parse_number(str(example.answer)) is not None
        ):
            grouped.setdefault(example.document.id, []).append(example)
    rng = random.Random(419)
    contexts = sorted(grouped)
    rng.shuffle(contexts)
    if len(contexts) < CALIBRATION_CONTEXTS:
        raise ValueError("Insufficient unused calibration contexts")
    selected = [
        rng.choice(sorted(grouped[doc], key=lambda example: example.id))
        for doc in contexts[:CALIBRATION_CONTEXTS]
    ]
    write_json(
        MANIFEST,
        {
            "version": 4,
            "seed": 419,
            "scope": "300 new project-held-out public-training reference contexts",
            "excluded_context_ids": sorted(exclusions),
            "examples": [{"id": e.id, "document_id": e.document.id} for e in selected],
            "data_sha256": file_sha256(DATA),
        },
    )
    write_json(
        FREEZE,
        {
            "identity": identity(),
            "runner_sha256": file_sha256(Path(__file__)),
            "scorer_sha256": file_sha256(Path(scoring.__file__)),
            "comparison_runner_sha256": file_sha256(Path(experiment.__file__)),
            "policy_count": 1,
            "target_error": 0.10,
            "alpha": 0.05,
            "minimum_accepted": 20,
            "selection": (
                "Single training-developed temporal-agreement gate; no retuning after calibration"
            ),
            "development_only_feasibility": {"evaluated": 100, "accepted": 30, "correct": 29},
            "assumptions": (
                "IID reference contexts and matching future distribution; "
                "context IDs do not establish independent source filings; "
                "public training data may occur in pretraining"
            ),
        },
    )
    print(f"Frozen one policy and {len(selected)} fresh calibration contexts; no outcomes scored.")


def verify() -> tuple[dict, dict]:
    frozen = read_program_cache(FREEZE)
    manifest = read_program_cache(MANIFEST)
    rows = manifest["examples"]
    if (
        frozen["identity"] != identity()
        or frozen["runner_sha256"] != file_sha256(Path(__file__))
        or frozen["scorer_sha256"] != file_sha256(Path(scoring.__file__))
        or frozen["comparison_runner_sha256"] != file_sha256(Path(experiment_module().__file__))
        or len(rows) != CALIBRATION_CONTEXTS
        or len({r["id"] for r in rows}) != CALIBRATION_CONTEXTS
        or len({r["document_id"] for r in rows}) != CALIBRATION_CONTEXTS
        or set(manifest["excluded_context_ids"]) & {r["document_id"] for r in rows}
    ):
        raise ValueError("Calibration source/scorer/policy/membership changed after freeze")
    return frozen, manifest


def run() -> None:
    if REPORT.exists() or CERTIFICATE.exists():
        raise ValueError("Preserve completed calibration outcomes; refusing replacement")
    frozen, manifest = verify()
    planner = engine()
    experiment = experiment_module()
    experiment.verify_runtime(planner, verify_bytes=True)
    lookup = {e.id: e for e in load_examples(DATA)}
    records = []
    cache_before = len(list(planner.cache_dir.glob("*.json")))
    if CHECKPOINT.exists():
        saved = read_checkpoint(CHECKPOINT)
        if saved["freeze_sha256"] != file_sha256(FREEZE) or [
            r["id"] for r in saved["predictions"]
        ] != [r["id"] for r in manifest["examples"][: len(saved["predictions"])]]:
            raise ValueError("Calibration checkpoint differs from frozen run")
        records = saved["predictions"]
    for row in manifest["examples"][len(records) :]:
        example = lookup[row["id"]]
        if example.document.id != row["document_id"]:
            raise ValueError("Calibration source/context membership differs")
        base_evaluated = eligible_question(example.question)
        proposed = (
            answer(example.document, example.question, planner)
            if base_evaluated
            else {
                "abstained": True,
                "answer": None,
                "plan": None,
                "latency_ms": 0.0,
            }
        )
        from financial_qa.executor import Plan

        gate = None
        if not proposed["abstained"]:
            try:
                gate = acceptance_plan(
                    example.document,
                    example.question,
                    example.document.evidence(),
                    Plan.model_validate(proposed["plan"]),
                )
            except ExecutionError:
                pass
        accepted = gate is not None and not gate.abstain
        records.append(
            {
                "id": example.id,
                "document_id": example.document.id,
                "accepted": accepted,
                "correct": accepted
                and numeric_match(proposed["answer"], gate.scale, example.answer, example.scale),
                "predicted": proposed["answer"] if accepted else None,
                "predicted_scale": gate.scale if accepted else "",
                "gold": example.answer,
                "gold_scale": example.scale,
                "base_evaluated": base_evaluated,
                "base_abstained": proposed["abstained"] if base_evaluated else None,
                "latency_ms": proposed["latency_ms"],
            }
        )
        write_json(CHECKPOINT, {"freeze_sha256": file_sha256(FREEZE), "predictions": records})
        if len(records) % 10 == 0:
            verify()
            experiment.verify_runtime(planner)
            print(
                json.dumps(
                    {
                        "stage": "v4-fresh-calibration",
                        "completed": len(records),
                        "total": CALIBRATION_CONTEXTS,
                    }
                ),
                flush=True,
            )
    evidence = [
        {key: row[key] for key in ("id", "document_id", "accepted", "correct")} for row in records
    ]
    certificate = certificate_from_evidence(evidence, frozen["identity"], manifest["examples"])
    write_json(
        REPORT,
        {
            "scope": manifest["scope"],
            "freeze_sha256": file_sha256(FREEZE),
            "identity": frozen["identity"],
            "predictions": records,
            "cache_entries_before": cache_before,
            "cache_entries_after": len(list(planner.cache_dir.glob("*.json"))),
            "risk_summary": {
                k: v
                for k, v in certificate.items()
                if k not in {"identity", "membership", "evidence"}
            },
        },
    )
    write_json(CERTIFICATE, certificate)
    print(
        json.dumps(
            {
                k: v
                for k, v in certificate.items()
                if k not in {"identity", "membership", "evidence"}
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "run"])
    if parser.parse_args().stage == "freeze":
        freeze()
    else:
        run()


if __name__ == "__main__":
    main()
