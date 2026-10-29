"""Secondary official TAT-QA arithmetic scoring of the completed frozen v5 study.

Uses the existing hash-verified upstream evaluator. No downloads, fitting, model
calls, policy selection or changes to primary prediction reports occur here.
Run from financial-qa after experiment_v5.py summarize.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from financial_qa.calibration import file_sha256
from financial_qa.data import REVISION, load_examples
from financial_qa.storage import write_json


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def score_method(report: dict, questions: dict, metric_type) -> dict:
    """Score every checked row, including abstentions, with the upstream metric."""
    overall, accepted = metric_type(), metric_type()
    for row in report["predictions"]:
        question = questions[row["id"]]
        if question["answer_type"] != "arithmetic":
            raise ValueError("Official secondary scoring is restricted to arithmetic")
        prediction = None if row["abstained"] else str(row["predicted"])
        overall(ground_truth=question, prediction=prediction, pred_scale=row["predicted_scale"])
        if prediction is not None:
            accepted(
                ground_truth=question, prediction=prediction, pred_scale=row["predicted_scale"]
            )
    em, f1, scale, _ = overall.get_overall_metric()
    accepted_em, accepted_f1, _, _ = accepted.get_overall_metric()
    return {
        "questions": report["evaluated"],
        "answered": report["answered"],
        "coverage": report["coverage"],
        "official_em": em,
        "official_f1": f1,
        "official_scale_accuracy": scale,
        "official_em_among_answered": accepted_em if report["answered"] else None,
        "official_f1_among_answered": accepted_f1 if report["answered"] else None,
        "primary_local_accuracy": report["accuracy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-dir", type=Path, default=Path("artifacts/official-evaluator"))
    args = parser.parse_args()
    output = Path("reports/v5-official-secondary.json")
    if output.exists():
        raise ValueError("Preserve completed secondary evidence; refusing replacement")
    experiment, adapter = module("experiment_v5"), module("official_score")
    policy, selected = experiment.verify_freeze()
    summary_path = Path("reports/v5-summary.json")
    summary = json.loads(summary_path.read_text())
    policy_sha = file_sha256(experiment.POLICY)
    if summary["policy_sha256"] != policy_sha:
        raise ValueError("Primary summary differs from frozen policy")
    metric_type = adapter.load_evaluator(args.evaluator_dir, allow_download=False)
    lookup = {example.id: example for example in load_examples(experiment.DATA)}
    questions = {
        question["uid"]: question
        for context in json.loads(experiment.DATA.read_text())
        for question in context["questions"]
    }
    results = {}
    for name in ("baseline", "candidate"):
        path = Path(f"reports/v5-comparison-{name}.json")
        report = json.loads(path.read_text())
        experiment.validate_report(report, "comparison", name, selected, lookup, policy["identity"])
        if report["policy_sha256"] != policy_sha or summary["methods"][name] != {
            key: value for key, value in report.items() if key not in {"identity", "predictions"}
        }:
            raise ValueError("Primary report and summary differ")
        results[name] = {
            "prediction_report_sha256": file_sha256(path),
            **score_method(report, questions, metric_type),
        }
    if experiment.verify_freeze() != (policy, selected):
        raise ValueError("Policy or source membership changed during secondary scoring")
    write_json(
        output,
        {
            "scope": policy["scope"],
            "purpose": "Secondary official arithmetic-subset EM/F1; no model selection",
            "dataset_revision": REVISION,
            "data_sha256": file_sha256(experiment.DATA),
            "policy_sha256": policy_sha,
            "primary_summary_sha256": file_sha256(summary_path),
            "scoring_source_sha256": file_sha256(Path(__file__)),
            "evaluator_adapter_sha256": file_sha256(Path(adapter.__file__)),
            "evaluator_sources_sha256": adapter.SOURCES,
            "methods": results,
            "limitations": (
                "Project source-group comparison on public training data; "
                "not a hidden-test leaderboard"
            ),
        },
    )
    print(output)


if __name__ == "__main__":
    main()
