"""Secondary scoring with the unmodified, pinned official TAT-QA evaluator.

Run from the project directory after prediction/policy freezing. This command
never fits, selects, or calls a model. Scorer source stays in ignored artifacts;
downloads are opt-in, HTTPS-only, content-checked, and pinned to the data revision.
Install requirements-evaluation.txt to supply the evaluator's optional dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from financial_qa.data import REVISION
from financial_qa.storage import download, write_json

SOURCES = {
    "tatqa_utils.py": "a84bb2f960737cf0a53733637a674cc4b20ef030a2be6a4b21dc2c4356f415ec",
    "tatqa_metric.py": "2aeeac479f89f8c76300af1cc0e8d098eb86af84bc386b38b6ab4af484a6dea8",
}
SOURCE_ROOT = f"https://raw.githubusercontent.com/NExTplusplus/TAT-QA/{REVISION}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_evaluator(directory: Path, allow_download: bool):
    """Load only the reviewed upstream source bytes, never an arbitrary scorer file."""
    for filename, expected in SOURCES.items():
        path = directory / filename
        if not path.exists() and allow_download:
            download(f"{SOURCE_ROOT}/{filename}", path, max_bytes=100_000)
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise ValueError(f"Missing or changed pinned scorer: {path}; use --download-scorer")
    try:
        # Explicit modules ensure an unrelated installed tatqa_utils cannot shadow
        # the verified sibling. No evaluator source is copied into the package.
        for filename in SOURCES:
            name = Path(filename).stem
            spec = importlib.util.spec_from_file_location(name, directory / filename)
            if spec is None or spec.loader is None:
                raise ValueError(f"Cannot load pinned evaluator module: {name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return sys.modules["tatqa_metric"].TaTQAEmAndF1
    except ModuleNotFoundError as error:
        raise ValueError("Install requirements-evaluation.txt for official scoring") from error


def score_report(report_path: Path, manifest: dict, questions: dict, metric_type) -> dict:
    """Score exactly the frozen IDs, counting abstentions as incorrect official answers."""
    report = json.loads(report_path.read_text())
    rows = report["predictions"]
    identifiers = [row["id"] for row in rows]
    expected = {row["id"] for row in manifest["examples"]}
    if len(identifiers) != len(expected) or set(identifiers) != expected:
        raise ValueError("Predictions must contain the identical frozen question set exactly once")
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if report["manifest_sha256"] != digest:
        raise ValueError("Prediction report does not match the frozen manifest")
    overall, accepted = metric_type(), metric_type()
    accepted_count = 0
    for row in rows:
        if type(row["abstained"]) is not bool:
            raise ValueError("Each prediction needs an explicit boolean abstention status")
        gold = questions[row["id"]]
        if gold["answer_type"] != "arithmetic":
            raise ValueError("This secondary evaluation is restricted to arithmetic questions")
        if not row["abstained"] and row["predicted"] is None:
            raise ValueError("A non-abstained prediction cannot have a missing numeric answer")
        prediction = None if row["abstained"] else str(row["predicted"])
        scale = row["predicted_scale"]
        overall(ground_truth=gold, prediction=prediction, pred_scale=scale)
        if prediction is not None:
            accepted(ground_truth=gold, prediction=prediction, pred_scale=scale)
            accepted_count += 1
    em, f1, scale_accuracy, _ = overall.get_overall_metric()
    accepted_em, accepted_f1, _, _ = accepted.get_overall_metric()
    return {
        "backend": report["backend"],
        "prediction_report": report_path.name,
        "prediction_report_sha256": sha256(report_path),
        "questions": len(rows),
        "answered": accepted_count,
        "coverage": accepted_count / len(rows),
        "official_em": em,
        "official_f1": f1,
        "official_scale_accuracy": scale_accuracy,
        "official_em_among_answered": accepted_em if accepted_count else None,
        "official_f1_among_answered": accepted_f1 if accepted_count else None,
        "primary_local_accuracy": report["accuracy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--data", type=Path, default=Path("data/tatqa_dev.json"))
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/dev-holdout-v2.json"))
    parser.add_argument("--evaluator-dir", type=Path, default=Path("artifacts/official-evaluator"))
    parser.add_argument("--download-scorer", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/v2-official-secondary.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve the existing secondary report; choose a new --output path")
    evaluator = load_evaluator(args.evaluator_dir, args.download_scorer)
    manifest = json.loads(args.manifest.read_text())
    questions = {
        question["uid"]: question
        for document in json.loads(args.data.read_text())
        for question in document["questions"]
    }
    result = {
        "purpose": "Secondary metric diagnostic after policy freezing; no model selection",
        "scope": "Frozen arithmetic-only development subset, not the full benchmark leaderboard",
        "dataset_revision": REVISION,
        "data_sha256": sha256(args.data),
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "evaluator_sources": {
            filename: {"url": f"{SOURCE_ROOT}/{filename}", "sha256": digest}
            for filename, digest in SOURCES.items()
        },
        "methods": [score_report(path, manifest, questions, evaluator) for path in args.reports],
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
