"""Run frozen development, independent calibration and final86-doc experiments.

The entire finite policy grid is fixed before calibration. Each document is a
statistical unit. Raw model requests are cached so all policies share identical
program proposals. Interrupted stages retain atomic per-document checkpoints.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from financial_qa.benchmark import interval
from financial_qa.calibration import POLICIES, calibrate, file_sha256, save_certificate
from financial_qa.data import load_examples
from financial_qa.evaluation import numeric_match
from financial_qa.reasoning_v3 import ProgramPlanner, RuntimeProfile, runtime_signature
from financial_qa.service import answer
from financial_qa.storage import write_json

MODEL_SHA = "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"


def profile() -> RuntimeProfile:
    return RuntimeProfile(
        "qwen3-4b-q4-nonthinking",
        MODEL_SHA,
        runtime=json.loads(Path("benchmarks/runtime-v3.json").read_text()),
    )


def identity() -> dict:
    return {
        **profile().identity(),
        "calibration_manifest_sha256": file_sha256(Path("benchmarks/train-v3-400.json")),
        "calibration_data_sha256": file_sha256(Path("data/tatqa_train.json")),
        "calibration_membership": json.loads(Path("benchmarks/train-v3-400.json").read_text())[
            "calibration"
        ],
    }


def run(stage: str, policy: str) -> None:
    output = Path(f"reports/v3-{stage}-{policy}.json")
    checkpoint = Path(f"artifacts/v3-{stage}-{policy}-checkpoint.json")
    if output.exists():
        raise ValueError("Refusing to overwrite an existing result")
    manifest_path = Path(
        "benchmarks/dev-holdout-v3.json" if stage == "final" else "benchmarks/train-v3-400.json"
    )
    manifest = json.loads(manifest_path.read_text())
    selected = manifest["examples"] if stage == "final" else manifest[stage]
    expected_count = {"development": 100, "calibration": 400, "final": 86}[stage]
    if len(selected) != expected_count:
        raise ValueError("Frozen manifest count differs from the declared stage size")
    if len({row["id"] for row in selected}) != len(selected) or len(
        {row["document_id"] for row in selected}
    ) != len(selected):
        raise ValueError("Frozen manifest has duplicate question/document identities")
    data = Path("data/tatqa_dev.json" if stage == "final" else "data/tatqa_train.json")
    lookup = {x.id: x for x in load_examples(data)}
    cache = Path(f"artifacts/v3-{stage}-cache")
    cache_before = len(list(cache.glob("*.json")))
    planner = ProgramPlanner(profile(), cache_dir=cache, policy=policy)
    if policy != "semantic" and runtime_signature(planner.client.properties()) != profile().runtime:
        raise ValueError("Active server runtime differs from the frozen experiment profile")
    run_identity = {
        **identity(),
        "policy": policy,
        "stage": stage,
        "manifest_sha256": file_sha256(manifest_path),
    }
    records = []
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved["identity"] != run_identity:
            raise ValueError("Checkpoint identity differs from the frozen run")
        records = saved["predictions"]
        if [row["id"] for row in records] != [row["id"] for row in selected[: len(records)]]:
            raise ValueError("Checkpoint membership differs from the manifest prefix")
    for row in selected[len(records) :]:
        example = lookup[row["id"]]
        if example.document.id != row["document_id"]:
            raise ValueError("Manifest and source document identity differ")
        prediction = answer(example.document, example.question, planner)
        scale = (prediction["plan"] or {}).get("scale", "")
        correct = numeric_match(prediction["answer"], scale, example.answer, example.scale)
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
                "reason": prediction["reason"],
                "plan": prediction["plan"],
                "latency_ms": prediction["latency_ms"],
            }
        )
        write_json(checkpoint, {"identity": run_identity, "predictions": records})
        print(
            json.dumps(
                {
                    "stage": stage,
                    "policy": policy,
                    "completed": len(records),
                    "total": len(selected),
                }
            ),
            flush=True,
        )
    accepted = [row for row in records if not row["abstained"]]
    cache_after = len(list(cache.glob("*.json")))
    report = {
        "cache_entries_before": cache_before,
        "cache_entries_after": cache_after,
        "new_cached_model_responses": cache_after - cache_before,
        "latency_status": "actual deterministic execution"
        if policy == "semantic"
        else "live local model generation plus validation/rejections"
        if cache_before == 0
        else "cached replay or mixed replay/live; not pure live model latency",
        "manifest_sha256": __import__("hashlib")
        .sha256(json.dumps(manifest, sort_keys=True).encode())
        .hexdigest(),
        "identity": run_identity,
        "backend": "program-v3-" + policy,
        "scorer": "Local scale-aware numeric tolerance0.005 in gold units; not official TATQA",
        "evaluated": len(records),
        "accuracy": sum(row["correct"] for row in records) / len(records),
        "coverage": len(accepted) / len(records),
        "answered_accuracy": sum(row["correct"] for row in accepted) / len(accepted)
        if accepted
        else None,
        "accuracy_interval": interval(
            [float(row["correct"]) for row in records], [row["document_id"] for row in records]
        ),
        "p95_latency_ms": float(np.percentile([row["latency_ms"] for row in records], 95)),
        "abstention_reasons": dict(Counter(row["reason"] for row in records if row["abstained"])),
        "predictions": records,
    }
    write_json(output, report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"predictions", "identity"}},
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["development", "calibration", "final", "freeze", "certify"]
    )
    parser.add_argument("--policy", choices=POLICIES, default="model")
    args = parser.parse_args()
    if args.stage == "freeze":
        path = Path("benchmarks/policy-v3.json")
        if path.exists():
            raise ValueError("Policy grid already frozen")
        write_json(
            path,
            {
                "identity": identity(),
                "policies": list(POLICIES),
                "target_error": 0.10,
                "familywise_alpha": 0.05,
                "minimum_accepted": 20,
                "selection": (
                    "Maximum calibration coverage among policies with exact binomial "
                    "upper error <=0.10 after Bonferroni alpha/5; "
                    "failclosed if none qualify"
                ),
                "final_manifest_sha256": file_sha256(Path("benchmarks/dev-holdout-v3.json")),
            },
        )
    elif args.stage == "certify":
        frozen = json.loads(Path("benchmarks/policy-v3.json").read_text())
        if frozen["identity"] != identity():
            raise ValueError("Prediction code/profile changed after finite-grid freeze")
        reports = {
            policy: json.loads(Path(f"reports/v3-calibration-{policy}.json").read_text())
            for policy in POLICIES
        }
        for policy, report in reports.items():
            if (
                report["identity"].get("stage") != "calibration"
                or report["identity"].get("policy") != policy
            ):
                raise ValueError(
                    "Calibration report stage or policy label differs from the declared grid"
                )
            if any(report["identity"][key] != value for key, value in identity().items()):
                raise ValueError("Calibration outputs do not share the frozen inference identity")
        certificate = calibrate(reports, identity())
        save_certificate(Path("reports/v3-risk-certificate.json"), certificate)
        print(json.dumps(certificate, indent=2))
    else:
        if args.stage != "development":
            frozen = json.loads(Path("benchmarks/policy-v3.json").read_text())
            if frozen["identity"] != identity():
                raise ValueError("Prediction code/profile changed after calibration freeze")
        run(args.stage, args.policy)


if __name__ == "__main__":
    main()
