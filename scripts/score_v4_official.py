"""Secondary official scoring of the frozen public-training arithmetic study.

This uses the verified upstream arithmetic evaluator without changing prediction
reports or selecting a model. No evaluator or dependency downloads occur here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from financial_qa.calibration import file_sha256
from financial_qa.data import REVISION
from financial_qa.storage import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-dir", type=Path, default=Path("artifacts/official-evaluator"))
    args = parser.parse_args()
    output = Path("reports/v4-official-secondary.json")
    if output.exists():
        raise ValueError("Refusing to overwrite completed secondary evidence")
    adapter_path = Path(__file__).with_name("official_score.py")
    spec = importlib.util.spec_from_file_location("verified_secondary_adapter", adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    metric = adapter.load_evaluator(args.evaluator_dir, allow_download=False)
    manifest = json.loads(Path("benchmarks/train-holdout-v4.json").read_text())
    frozen = json.loads(Path("benchmarks/policy-v4.json").read_text())
    source = Path("data/tatqa_train.json")
    if (
        file_sha256(source) != manifest["data_sha256"]
        or file_sha256(Path("benchmarks/train-holdout-v4.json")) != frozen["manifest_sha256"]
    ):
        raise ValueError("Source or manifest differs from the frozen study")
    for method in ("baseline", "layout"):
        report = json.loads(Path(f"reports/v4-heldout-{method}.json").read_text())
        if report["identity"] != frozen["methods"][method] or report[
            "freeze_sha256"
        ] != file_sha256(Path("benchmarks/policy-v4.json")):
            raise ValueError("Prediction identity differs from the frozen study")
    questions = {
        q["uid"]: q for context in json.loads(source.read_text()) for q in context["questions"]
    }
    write_json(
        output,
        {
            "scope": manifest["scope"],
            "purpose": "Secondary official arithmetic-subset EM/F1; no model selection",
            "dataset_revision": REVISION,
            "data_sha256": file_sha256(source),
            "policy_freeze_sha256": file_sha256(Path("benchmarks/policy-v4.json")),
            "adapter_sha256": file_sha256(adapter_path),
            "evaluator_sources_sha256": adapter.SOURCES,
            "methods": [
                adapter.score_report(
                    Path(f"reports/v4-heldout-{method}.json"), manifest, questions, metric
                )
                for method in ("baseline", "layout")
            ],
            "limitations": (
                "Project-held-out public-training arithmetic study; not a hidden-test leaderboard"
            ),
        },
    )
    print(output)


if __name__ == "__main__":
    main()
