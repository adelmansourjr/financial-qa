"""Summarize immutable v3 outputs; never fit a policy on final outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from financial_qa.benchmark import compare, interval
from financial_qa.calibration import POLICIES, code_fingerprint, digest
from financial_qa.evaluation import numeric_match
from financial_qa.storage import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    deadline = time.monotonic() + 7200
    while not all(Path(f"reports/v3-final-{policy}.json").exists() for policy in POLICIES):
        if not args.wait or time.monotonic() >= deadline:
            raise ValueError("The complete frozen final evaluation is not available")
        time.sleep(3)
    frozen = json.loads(Path("benchmarks/policy-v3.json").read_text())
    if frozen["identity"]["prediction_code_sha256"] != code_fingerprint():
        raise ValueError("Prediction code drifted after calibration freeze")
    reports = {
        policy: json.loads(Path(f"reports/v3-final-{policy}.json").read_text())
        for policy in POLICIES
    }
    calibration = {
        policy: json.loads(Path(f"reports/v3-calibration-{policy}.json").read_text())
        for policy in POLICIES
    }
    methods = {}
    for policy, report in reports.items():
        records = report["predictions"]
        for key, expected in frozen["identity"].items():
            if (
                report["identity"].get(key) != expected
                or calibration[policy]["identity"].get(key) != expected
            ):
                raise ValueError("A measured result does not match the frozen inference identity")
        accepted = [row for row in records if not row["abstained"]]
        methods[policy] = {
            key: value for key, value in report.items() if key not in {"identity", "predictions"}
        }
        methods[policy]["coverage_interval"] = interval(
            [float(not row["abstained"]) for row in records],
            [row["document_id"] for row in records],
        )
        methods[policy]["answered_accuracy_interval"] = (
            interval(
                [float(row["correct"]) for row in accepted],
                [row["document_id"] for row in accepted],
            )
            if accepted
            else None
        )
        failures = []
        for row in records:
            failures.append(
                "correct"
                if row["correct"]
                else "abstained"
                if row["abstained"]
                else "wrong_unit_only"
                if numeric_match(
                    row["predicted"], row["gold_scale"], row["gold"], row["gold_scale"]
                )
                else "wrong_value_or_operation"
            )
        methods[policy]["failures"] = dict(Counter(failures))
    certificate = json.loads(Path("reports/v3-risk-certificate.json").read_text())
    criterion_audit = json.loads(Path("reports/v3-criterion-prefreeze-audit.json").read_text())
    current_scorer_hash = hashlib.sha256(
        Path("src/financial_qa/evaluation.py").read_bytes()
    ).hexdigest()
    if criterion_audit["evaluation_py_sha256"] != current_scorer_hash:
        raise ValueError("Numeric correctness criterion changed during the experiment")
    criterion = {
        "risk_certificate_sha256": certificate["artifact_sha256"],
        "evaluation_py_sha256": current_scorer_hash,
        "criterion": criterion_audit["criterion"],
        "scorer_matches_previous_source_snapshot": criterion_audit["matches_prior_source_snapshot"],
        "calibration_prediction_sha256": {
            policy: digest(report["predictions"]) for policy, report in calibration.items()
        },
        "final_prediction_sha256": {
            policy: digest(report["predictions"]) for policy, report in reports.items()
        },
        "note": (
            "Companion integrity audit. Risk is defined by local numeric correctness, "
            "not human financial truth."
        ),
    }
    criterion_path = Path("reports/v3-criterion-binding.json")
    if not criterion_path.exists():
        write_json(criterion_path, criterion)
    selected = certificate["selected_policy"]
    if selected:
        risk = {**reports[selected], "backend": "risk-controlled-v3", "selection": selected}
    else:
        records = [
            {
                **row,
                "predicted": None,
                "predicted_scale": "",
                "abstained": True,
                "correct": False,
                "reason": "No policy met independent calibration target",
                "plan": None,
            }
            for row in reports["model"]["predictions"]
        ]
        risk = {
            "backend": "risk-controlled-v3",
            "selection": None,
            "manifest_sha256": reports["model"]["manifest_sha256"],
            "accuracy": 0.0,
            "coverage": 0.0,
            "answered_accuracy": None,
            "evaluated": len(records),
            "predictions": records,
        }
    risk["evaluation_mode"] = (
        "Frozen calibration selection applied to identical saved policy predictions; "
        "no additional model calls"
    )
    risk["calibration_artifact_sha256"] = certificate["artifact_sha256"]
    risk_path = Path("reports/v3-final-risk-controlled.json")
    if not risk_path.exists():
        write_json(risk_path, risk)
    summary = {
        "scope": (
            "All 86 previously uninspected eligible public-development documents; "
            "one numeric arithmetic question per document"
        ),
        "frozen_prediction_code_verified": True,
        "policy_freeze_sha256": hashlib.sha256(
            Path("benchmarks/policy-v3.json").read_bytes()
        ).hexdigest(),
        "final_manifest_sha256": reports["model"]["manifest_sha256"],
        "methods": methods,
        "criterion_binding_sha256": digest(criterion),
        "criterion_binding_path": str(criterion_path),
        "paired_comparisons": [
            compare(reports["semantic"], reports[policy])
            for policy in POLICIES
            if policy != "semantic"
        ],
        "risk_control": {
            key: value for key, value in certificate.items() if key not in {"evidence", "identity"}
        },
        "risk_controlled_final": {
            key: value for key, value in risk.items() if key not in {"predictions", "identity"}
        },
        "calibration": {
            policy: {
                "accuracy": report["accuracy"],
                "coverage": report["coverage"],
                "answered_accuracy": report["answered_accuracy"],
                "p95_latency_ms": report["p95_latency_ms"],
            }
            for policy, report in calibration.items()
        },
        "raw_cache_entries": {
            stage: len(list(Path(f"artifacts/v3-{stage}-cache").glob("*.json")))
            for stage in ("development", "calibration", "final")
        },
        "latency_note": (
            "The first model pass is live generation. Other policies replay "
            "identical raw model outputs; their measured latency is not fresh LLM "
            "inference latency."
        ),
        "risk_note": (
            "Error upper bound applies only under stated fixed-policy IID document "
            "assumptions. No guarantee under distribution shift and no individual-answer "
            "confidence claim."
        ),
    }
    destination = Path("reports/v3-summary.json")
    if destination.exists():
        raise ValueError("Refusing to overwrite a measured summary")
    write_json(destination, summary)
    print(
        json.dumps(
            {"completed": True, "summary": str(destination), "summary_sha256": digest(summary)}
        )
    )


if __name__ == "__main__":
    main()
