"""Enforce frozen study identities and recompute outcomes before publication."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def experiment():
    path = Path(__file__).resolve().parents[1] / "scripts" / "experiment_v4.py"
    spec = importlib.util.spec_from_file_location("experiment_v4_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def heldout(tmp_path, monkeypatch, experiment):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports").mkdir()
    membership = [{"id": f"question-{i}", "document_id": f"context-{i}"} for i in range(200)]
    frozen = {
        "methods": {"baseline": {"version": 3}, "layout": {"version": 4}},
        "criterion_sha256": "scorer",
        "limitations": ["synthetic test"],
    }
    manifest = {"scope": "synthetic200", "examples": membership}
    monkeypatch.setattr(experiment, "verify_freeze", lambda: (frozen, manifest))
    monkeypatch.setattr(experiment, "file_sha256", lambda _path: "freeze")
    monkeypatch.setattr(
        experiment,
        "load_examples",
        lambda _path: [SimpleNamespace(id=row["id"], answer=2, scale="") for row in membership],
    )
    for method in ("baseline", "layout"):
        records = [
            {
                **row,
                "predicted": "2",
                "predicted_scale": "",
                "gold": 2,
                "gold_scale": "",
                "correct": True,
                "abstained": False,
            }
            for row in membership
        ]
        report = {
            "identity": frozen["methods"][method],
            "criterion_sha256": "scorer",
            "freeze_sha256": "freeze",
            "evaluated": 200,
            "correct": 200,
            "answered": 200,
            "accuracy": 1.0,
            "coverage": 1.0,
            "answered_accuracy": 1.0,
            "predictions": records,
        }
        (tmp_path / f"reports/v4-heldout-{method}.json").write_text(json.dumps(report))
    return tmp_path


def test_summary_recomputes_paired_frozen_membership(experiment, heldout):
    experiment.summarize()
    result = json.loads((heldout / "reports/v4-summary.json").read_text())
    assert result["paired_accuracy_gain"]["mean"] == 0
    assert result["methods"]["layout"]["evaluated"] == 200


@pytest.mark.parametrize("mutation", ["membership", "scorer", "labels", "aggregate"])
def test_summary_rejects_mutated_reports(experiment, heldout, mutation):
    path = heldout / "reports/v4-heldout-layout.json"
    report = json.loads(path.read_text())
    if mutation == "membership":
        report["predictions"][0]["document_id"] = "different-context"
    elif mutation == "scorer":
        report["criterion_sha256"] = "changed-scorer"
    elif mutation == "labels":
        report["predictions"][0]["gold"] = 999
    else:
        report["accuracy"] = 0.99
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        experiment.summarize()
    assert not (heldout / "reports/v4-summary.json").exists()


@pytest.fixture
def frozen_files(tmp_path, monkeypatch, experiment):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/tatqa_train.json").write_text("[]")
    scorer = tmp_path / "scorer.py"
    scorer.write_text("# original scorer\n")
    monkeypatch.setattr(experiment, "scoring", SimpleNamespace(__file__=str(scorer)))
    monkeypatch.setattr(experiment, "identity", lambda method: {"method": method})
    monkeypatch.setattr(experiment, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(experiment, "FREEZE", tmp_path / "freeze.json")
    manifest = {
        "examples": [{"id": str(i), "document_id": f"doc-{i}"} for i in range(200)],
        "excluded_context_ids": ["old-doc"],
        "data_sha256": experiment.file_sha256(Path("data/tatqa_train.json")),
    }
    experiment.MANIFEST.write_text(json.dumps(manifest))
    frozen = {
        "runner_sha256": experiment.file_sha256(Path(experiment.__file__)),
        "manifest_sha256": experiment.file_sha256(experiment.MANIFEST),
        "criterion_sha256": experiment.file_sha256(scorer),
        "statistics_sha256": experiment.file_sha256(Path(experiment.statistics.__file__)),
        "environment": experiment.environment_identity(),
        "methods": {method: experiment.identity(method) for method in ("baseline", "layout")},
    }
    experiment.FREEZE.write_text(json.dumps(frozen))
    return scorer


def test_frozen_scorer_drift_is_rejected(experiment, frozen_files):
    experiment.verify_freeze()
    frozen_files.write_text("# changed scoring tolerance\n")
    with pytest.raises(ValueError, match="scorer"):
        experiment.verify_freeze()


def test_duplicate_contexts_fail_even_if_manifest_rehashed(experiment, frozen_files):
    manifest = json.loads(experiment.MANIFEST.read_text())
    manifest["examples"][1]["document_id"] = manifest["examples"][0]["document_id"]
    experiment.MANIFEST.write_text(json.dumps(manifest))
    frozen = json.loads(experiment.FREEZE.read_text())
    frozen["manifest_sha256"] = experiment.file_sha256(experiment.MANIFEST)
    experiment.FREEZE.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="membership"):
        experiment.verify_freeze()
