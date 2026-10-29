"""Audit consumed QA sources and guard future content-group sampling.

This never generates model answers, edits historical manifests, removes observed
errors or recertifies an inspected calibration set. Future sampling freezes only
source membership; model/scorer/protocol freezing remains a separate requirement.
Run from the project directory, after fetching the documented train/dev sources.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from financial_qa.calibration import file_sha256
from financial_qa.data import ROOT_URL, Document, load_examples
from financial_qa.source_reservations import (
    freeze_reservation,
    read_reservations,
    validate_reservation,
)
from financial_qa.split_integrity import (
    GROUPING_VERSION,
    content_groups,
    overlap_audit,
)
from financial_qa.storage import write_json

HISTORY = (
    Path("benchmarks/train-v3-400.json"),
    Path("benchmarks/train-holdout-v4.json"),
    Path("benchmarks/calibration-v4.json"),
)
# These are completed consumed sets, not editable candidates for another search.
# New completed experiments require an explicit ledger revision before sampling.
EXPECTED_HISTORY_SHA256 = (
    "fe87b6ed9c49517da70a2a0041145c368e0d10931ff1280a521ae40c73452956",
    "c021bfcbb480db1442b20541ddd281769ed5eaf2d2e6d8a60b6f7e5e90431eb0",
    "85f20654ae89f680276bb60b63558b94687540537f0e30bdffb0bebb243e1562",
)
TRAIN = Path("data/tatqa_train.json")
DEV = Path("data/tatqa_dev.json")
EXPECTED_SOURCE_SHA256 = {
    TRAIN: "2df6e722cdbaaa37efcbfb280f5c9a15be29a6ec18f618ef936fe63cc6d07c69",
    DEV: "8da095a819af6db3c14877c6df2d4d29960e41d1a63dd1fa853507bd2a616af5",
}
LEDGER = Path("benchmarks/source-reservations.json")


def sources() -> tuple[list[Document], dict[str, list[str]], dict]:
    """Verify saved source identities before trusting their question memberships."""
    if any(
        file_sha256(path) != expected
        for path, expected in zip(HISTORY, EXPECTED_HISTORY_SHA256, strict=True)
    ):
        raise ValueError("Completed consumed-source ledger changed; preserve historical manifests")
    previous, comparison, calibration = [json.loads(path.read_text()) for path in HISTORY]
    if any(manifest["data_sha256"] != file_sha256(TRAIN) for manifest in (comparison, calibration)):
        raise ValueError("Training bytes differ from the completed manifest identities")
    documents = []
    dev_ids = []
    for source in (TRAIN, DEV):
        receipt = json.loads(source.with_suffix(source.suffix + ".receipt.json").read_text())
        split = "train" if source == TRAIN else "dev"
        if (
            file_sha256(source) != EXPECTED_SOURCE_SHA256[source]
            or receipt["sha256"] != EXPECTED_SOURCE_SHA256[source]
            or type(receipt.get("bytes")) is not int
            or receipt["bytes"] != source.stat().st_size
            or receipt.get("url") != f"{ROOT_URL}/tatqa_dataset_{split}.json"
        ):
            raise ValueError("Pinned source receipt differs from actual data")
        for record in json.loads(source.read_text()):
            document = Document(
                record["table"]["uid"],
                tuple(tuple(str(cell) for cell in row) for row in record["table"]["table"]),
                tuple(row["text"] for row in record["paragraphs"]),
            )
            documents.append(document)
            if source == DEV:
                dev_ids.append(document.id)
    stages = {
        "historical_training_contexts": sorted(previous["previously_used_document_ids"]),
        "prompt_development": [row["document_id"] for row in previous["development"]],
        "calibration_v3": [row["document_id"] for row in previous["calibration"]],
        "comparison_v4": [row["document_id"] for row in comparison["examples"]],
        "calibration_v4": [row["document_id"] for row in calibration["examples"]],
        # Public development has been repeatedly inspected. Conservatively retire
        # the complete split, including contexts without a selected numeric item.
        "retired_public_development": dev_ids,
    }
    identity = {
        "grouping_version": GROUPING_VERSION,
        "source_sha256": {str(path): file_sha256(path) for path in (TRAIN, DEV)},
        "history_sha256": {str(path): file_sha256(path) for path in HISTORY},
        "guard_sha256": file_sha256(
            Path(__file__).resolve().parents[1] / "src/financial_qa/split_integrity.py"
        ),
        "runner_sha256": file_sha256(Path(__file__)),
        "reservation_guard_sha256": file_sha256(
            Path(__file__).resolve().parents[1] / "src/financial_qa/source_reservations.py"
        ),
    }
    return documents, stages, identity


def observed_training_contexts() -> set[str]:
    """Catch additional consumed sources when a later study adds prediction reports."""
    examples = load_examples(TRAIN)
    questions = {e.id: e.document.id for e in examples}
    known = {e.document.id for e in examples}
    observed = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, str) and key in {
                    "id",
                    "question_id",
                    "document_id",
                    "context_id",
                }:
                    if item in known:
                        observed.add(item)
                    if item in questions:
                        observed.add(questions[item])
                elif isinstance(item, (list, dict)):
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    paths = [
        *Path("reports").rglob("*.json"),
        *Path("artifacts").rglob("*checkpoint*.json"),
    ]
    for path in sorted(paths):
        visit(json.loads(path.read_text()))
    return observed


def verify_consumed_ledger(stages: dict[str, list[str]]) -> set[str]:
    consumed = {identifier for identifiers in stages.values() for identifier in identifiers}
    observed = observed_training_contexts()
    if observed - consumed:
        raise ValueError("Prediction reports expose additional consumed sources; update the ledger")
    return observed


def audit() -> dict:
    documents, stages, identity = sources()
    reservations = read_reservations(
        LEDGER,
        identity,
        load_examples(TRAIN),
        documents,
        {identifier for identifiers in stages.values() for identifier in identifiers},
    )
    for entry in reservations:
        stages[f"reserved_future_{entry['index']}"] = [
            row["document_id"] for row in entry["sampling"]["examples"]
        ]
    report = overlap_audit(documents, stages)
    groups = content_groups(documents)
    consumed = {identifier for identifiers in stages.values() for identifier in identifiers}
    blocked_groups = {groups[identifier] for identifier in consumed}
    observed = verify_consumed_ledger(stages)
    candidates = {
        example.document.id
        for example in load_examples(TRAIN)
        if example.answer_type == "arithmetic"
    }
    available = {groups[i] for i in candidates if groups[i] not in blocked_groups}
    report.update(
        {
            "identity": identity,
            "status": "source_overlap_detected"
            if not report["source_group_disjoint"]
            else "checked_source_keys_disjoint",
            "within_stage_repeated_source_groups": {
                stage: len(ids) - len({groups[i] for i in ids}) for stage, ids in stages.items()
            },
            "future_sampling": {
                "consumed_contexts": len(consumed),
                "consumed_source_groups": len(blocked_groups),
                "unused_arithmetic_source_groups_before_numeric_eligibility": len(available),
                "policy": "Exclude complete consumed components; one question per unused group",
                "new_evaluation_sample_created": False,
                "reported_training_contexts_checked_against_consumption_ledger": len(observed),
            },
        }
    )
    development = json.loads(Path("reports/v4-development-layout.json").read_text())
    comparison = json.loads(Path("reports/v4-summary.json").read_text())
    report["fit_diagnostics"] = {
        "local_llm_weight_training": False,
        "what_was_selected": "Prompt, symbolic grammar and acceptance policy on development only",
        "development_contexts": development["evaluated"],
        "development_accuracy": development["accuracy"],
        "comparison_contexts": comparison["methods"]["layout"]["evaluated"],
        "comparison_accuracy": comparison["methods"]["layout"]["accuracy"],
        "comparison_minus_development_accuracy": (
            comparison["methods"]["layout"]["accuracy"] - development["accuracy"]
        ),
        "interpretation": (
            "Descriptive different-context gap, not a causal diagnosis of over/underfitting. "
            "Prompt selection can overfit; no local weight-training learning curve exists."
        ),
    }
    report["limitations"] = [
        "Historical ID-disjoint samples contain repeated tables/paragraphs; "
        "independent-filings and IID calibration assumptions are unverified",
        "Connected source groups conservatively include boilerplate; "
        "they can miss paraphrases, edited tables and unknown common filings",
        "Public train/dev contents may be in model pretraining; this audit cannot certify absence",
        "All completed evaluation/calibration contexts are consumed; "
        "they cannot be reused as fresh tuning data",
        "No historical row is removed or risk bound recomputed after observing outcomes",
    ]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["audit", "freeze", "validate"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--require-disjoint", action="store_true", help="Fail if historical source groups overlap"
    )
    args = parser.parse_args()
    if args.stage == "audit":
        output = args.output or Path("reports/leakage_audit.json")
        if output.exists():
            parser.error("Preserve completed audit/evidence; choose a new output path")
        report = audit()
        write_json(output, report)
        print(json.dumps({"status": report["status"], "report": str(output)}))
        if args.require_disjoint and not report["source_group_disjoint"]:
            parser.exit(2, "Historical sources overlap; do not assert independent samples.\n")
        return
    if args.output is None or args.size is None or args.seed is None:
        parser.error("freeze/validate require explicit --output, --size and --seed")
    documents, stages, identity = sources()
    consumed = {identifier for identifiers in stages.values() for identifier in identifiers}
    examples = load_examples(TRAIN)
    entries = read_reservations(LEDGER, identity, examples, documents, consumed)
    registered_stages = {
        **stages,
        **{
            f"reserved_future_{entry['index']}": [
                row["document_id"] for row in entry["sampling"]["examples"]
            ]
            for entry in entries
        },
    }
    verify_consumed_ledger(registered_stages)
    if args.stage == "freeze":
        freeze_reservation(
            LEDGER, args.output, identity, examples, documents, consumed, args.size, args.seed
        )
    else:
        validate_reservation(
            LEDGER, args.output, identity, examples, documents, consumed, args.size, args.seed
        )
    print(f"Source-group membership {args.stage} passed; no outcomes evaluated")


if __name__ == "__main__":
    main()
