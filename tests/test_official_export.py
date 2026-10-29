"""Contracts for the optional secondary scorer; no upstream download is needed."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "official_score", Path(__file__).parents[1] / "scripts/official_score.py"
)
assert SPEC is not None and SPEC.loader is not None
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)


def prediction_report(tmp_path, rows):
    manifest = {"examples": [{"id": "q0"}, {"id": "q1"}]}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    report = tmp_path / "predictions.json"
    report.write_text(
        json.dumps(
            {"backend": "fixture", "manifest_sha256": digest, "predictions": rows, "accuracy": 0.5}
        )
    )
    return report, manifest


def test_secondary_scorer_preserves_zero_answer_and_abstention_denominator(tmp_path):
    calls = []

    class Metric:
        def __init__(self):
            self.inputs = []
            calls.append(self.inputs)

        def __call__(self, **values):
            self.inputs.append(values)

        def get_overall_metric(self):
            return 0.5, 0.5, 1.0, 0.0

    path, manifest = prediction_report(
        tmp_path,
        [
            {"id": "q0", "abstained": False, "predicted": 0, "predicted_scale": ""},
            {"id": "q1", "abstained": True, "predicted": None, "predicted_scale": ""},
        ],
    )
    gold = {identifier: {"answer_type": "arithmetic"} for identifier in ("q0", "q1")}
    result = SCORER.score_report(path, manifest, gold, Metric)
    assert len(calls[0]) == 2 and len(calls[1]) == 1
    assert calls[0][0]["prediction"] == "0"
    assert calls[0][1]["prediction"] is None
    assert result["answered"] == 1 and result["coverage"] == 0.5


def test_secondary_scorer_rejects_duplicate_or_unfrozen_questions(tmp_path):
    path, manifest = prediction_report(tmp_path, [{"id": "q0"}, {"id": "q0"}])
    with pytest.raises(ValueError, match="identical frozen"):
        SCORER.score_report(path, manifest, {}, None)


def test_secondary_scorer_refuses_modified_upstream_code(tmp_path):
    for filename in SCORER.SOURCES:
        (tmp_path / filename).write_text("raise RuntimeError('must never execute')")
    with pytest.raises(ValueError, match="changed pinned scorer"):
        SCORER.load_evaluator(tmp_path, allow_download=False)
