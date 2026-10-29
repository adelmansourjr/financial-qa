"""Synthetic protocol contracts; never inspect real reserve or evaluation outcomes."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def experiment():
    path = Path(__file__).resolve().parents[1] / "scripts" / "experiment_v5.py"
    spec = importlib.util.spec_from_file_location("experiment_v5_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def study(tmp_path, monkeypatch, experiment):
    """A tiny, fully synthetic study with no model, dataset or network access."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "artifacts").mkdir()
    selected = [{"id": f"q{i}", "document_id": f"source{i}"} for i in range(4)]
    identity = {"fixture": "source-bound identity"}
    monkeypatch.setattr(experiment, "SIZE", 4)
    frozen = experiment.policy_contract(identity, "reservation-hash", "reservation-entry-hash")
    examples = [
        SimpleNamespace(
            id=row["id"],
            document=SimpleNamespace(id=row["document_id"]),
            answer=2,
            scale="",
            question="Synthetic question",
        )
        for row in selected
    ]
    rows = [
        {
            **row,
            "predicted": "2",
            "predicted_scale": "",
            "gold": 2,
            "gold_scale": "",
            "correct": True,
            "abstained": False,
            "reason": "",
            "plan": {"expression": "2", "scale": ""},
            "latency_ms": 1.0,
        }
        for row in selected
    ]
    monkeypatch.setattr(experiment, "SIZE", 4)
    monkeypatch.setattr(experiment, "verify_freeze", lambda: (frozen, selected))
    monkeypatch.setattr(experiment, "identity", lambda: identity)
    monkeypatch.setattr(experiment, "file_sha256", lambda _path: "policy-hash")
    monkeypatch.setattr(experiment, "load_examples", lambda _path: examples)
    for method in ("baseline", "candidate"):
        report = {
            "identity": identity,
            "method": method,
            "stage": "comparison",
            "scope": frozen["scope"],
            "policy_sha256": "policy-hash",
            "evaluated": 4,
            "correct": 4,
            "answered": 4,
            "accuracy": 1.0,
            "coverage": 1.0,
            "answered_accuracy": 1.0,
            "p95_latency_ms": 1.0,
            "predictions": copy.deepcopy(rows),
        }
        (tmp_path / f"reports/v5-comparison-{method}.json").write_text(json.dumps(report))
    return SimpleNamespace(
        root=tmp_path, rows=rows, selected=selected, identity=identity, frozen=frozen
    )


def test_summary_accepts_consistent_synthetic_reports(experiment, study):
    experiment.summarize()
    result = json.loads((study.root / "reports/v5-summary.json").read_text())
    assert result["methods"]["candidate"]["evaluated"] == 4
    assert result["methods"]["candidate"]["coverage"] == 1
    assert result["paired_accuracy_gain"]["mean"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "method",
        "stage",
        "evaluated",
        "answered",
        "coverage",
        "answered_accuracy",
        "p95_latency",
        "row_latency_nan",
        "row_latency_negative",
        "row_correct_not_bool",
        "row_abstained_not_bool",
        "row_labels",
        "row_source",
        "row_membership_order",
        "row_prediction",
        "row_answer_abstention",
    ],
)
def test_summary_rejects_inconsistent_report_contract(experiment, study, mutation):
    path = study.root / "reports/v5-comparison-candidate.json"
    report = json.loads(path.read_text())
    if mutation == "method":
        report["method"] = "baseline"
    elif mutation == "stage":
        report["stage"] = "development"
    elif mutation == "evaluated":
        report["evaluated"] = 99
    elif mutation == "answered":
        report["answered"] = 1
    elif mutation == "coverage":
        report["coverage"] = 0.25
    elif mutation == "answered_accuracy":
        report["answered_accuracy"] = 0.5
    elif mutation == "p95_latency":
        report["p95_latency_ms"] = 999.0
    elif mutation == "row_latency_nan":
        report["predictions"][0]["latency_ms"] = float("nan")
    elif mutation == "row_latency_negative":
        report["predictions"][0]["latency_ms"] = -1.0
    elif mutation == "row_correct_not_bool":
        report["predictions"][0]["correct"] = 1
    elif mutation == "row_abstained_not_bool":
        report["predictions"][0]["abstained"] = 0
    elif mutation == "row_labels":
        report["predictions"][0]["gold"] = 3
    elif mutation == "row_source":
        report["predictions"][0]["document_id"] = "different-source"
    elif mutation == "row_membership_order":
        report["predictions"].reverse()
    elif mutation == "row_prediction":
        report["predictions"][0]["predicted"] = "3"
    elif mutation == "row_answer_abstention":
        report["predictions"][0]["abstained"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        experiment.summarize()
    assert not (study.root / "reports/v5-summary.json").exists()


@pytest.fixture
def resume(study, monkeypatch, experiment):
    """Resume a complete synthetic checkpoint without ever invoking inference."""
    cache = study.root / "artifacts/cache"
    cache.mkdir()
    planner = SimpleNamespace(cache_dir=cache)
    runtime_calls = []
    runtime = SimpleNamespace(verify_runtime=lambda *args, **kwargs: runtime_calls.append(kwargs))
    monkeypatch.setattr(experiment, "engine", lambda *_args: planner)
    monkeypatch.setattr(experiment, "module", lambda _name: runtime)
    monkeypatch.setattr(
        experiment, "answer", lambda *_args: pytest.fail("Completed checkpoint reran inference")
    )
    (study.root / "reports/v5-comparison-candidate.json").unlink()
    checkpoint = study.root / "artifacts/v5-comparison-candidate-checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "identity": study.identity,
                "membership": study.selected,
                "predictions": study.rows,
            }
        )
    )
    return SimpleNamespace(**vars(study), checkpoint=checkpoint, runtime_calls=runtime_calls)


def test_valid_checkpoint_revalidates_runtime_before_final_report(experiment, resume):
    experiment.run("comparison", "candidate")
    report = json.loads((resume.root / "reports/v5-comparison-candidate.json").read_text())
    assert report["correct"] == 4
    assert len(resume.runtime_calls) >= 2


@pytest.mark.parametrize("mutation", ["correct", "gold", "abstained", "latency", "membership"])
def test_resumed_checkpoint_does_not_trust_saved_row_claims(experiment, resume, mutation):
    saved = json.loads(resume.checkpoint.read_text())
    row = saved["predictions"][0]
    if mutation == "correct":
        row["correct"] = 1
    elif mutation == "gold":
        row["gold"] = 3
    elif mutation == "abstained":
        row["abstained"] = True
    elif mutation == "latency":
        row["latency_ms"] = float("inf")
    else:
        row["document_id"] = "changed-source"
    resume.checkpoint.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        experiment.run("comparison", "candidate")
    assert not (resume.root / "reports/v5-comparison-candidate.json").exists()


def test_resume_checks_final_identity_even_when_no_new_sample_runs(experiment, resume, monkeypatch):
    calls = []

    def changing_identity():
        calls.append(None)
        return resume.identity if len(calls) == 1 else {"changed": "during-run"}

    monkeypatch.setattr(experiment, "identity", changing_identity)
    with pytest.raises(ValueError):
        experiment.run("comparison", "candidate")
    assert not (resume.root / "reports/v5-comparison-candidate.json").exists()


@pytest.fixture
def frozen_contract(experiment, tmp_path, monkeypatch):
    """An actual policy verifier with synthetic identities and reservation only."""
    monkeypatch.chdir(tmp_path)
    experiment.POLICY.parent.mkdir(parents=True)
    experiment.MEMBERSHIP.parent.mkdir(parents=True)
    experiment.MEMBERSHIP.write_text("synthetic reserved membership")
    current = {"fixture": "unchanged inference identity"}
    selected = [{"id": "q1", "document_id": "source1"}]
    membership = {"entry_sha256": "entry-hash", "sampling": {"examples": selected}}
    monkeypatch.setattr(experiment, "identity", lambda: current)
    monkeypatch.setattr(experiment, "source_membership", lambda: membership)
    policy = experiment.policy_contract(
        current, experiment.file_sha256(experiment.MEMBERSHIP), "entry-hash"
    )
    experiment.POLICY.write_text(json.dumps(policy))
    return SimpleNamespace(policy=policy, selected=selected)


def test_policy_verifier_accepts_the_complete_prospective_contract(experiment, frozen_contract):
    assert experiment.verify_freeze() == (frozen_contract.policy, frozen_contract.selected)


@pytest.mark.parametrize(
    "field",
    [
        "accuracy_target",
        "size",
        "seed",
        "scope",
        "primary",
        "improvement_target",
        "selection",
        "secondary",
        "limitations",
        "reservation_entry_sha256",
        "unknown_field",
    ],
)
def test_policy_verifier_rejects_changed_criteria_or_added_fields(
    experiment, frozen_contract, field
):
    policy = copy.deepcopy(frozen_contract.policy)
    if field == "accuracy_target":
        policy[field] = 0.5
    elif field == "limitations":
        policy[field] = []
    elif field in {"size", "seed"}:
        policy[field] += 1
    else:
        policy[field] = "changed after outcomes"
    experiment.POLICY.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="policy contract changed"):
        experiment.verify_freeze()


def test_policy_verifier_rejects_a_missing_contract_field(experiment, frozen_contract):
    del frozen_contract.policy["selection"]
    experiment.POLICY.write_text(json.dumps(frozen_contract.policy))
    with pytest.raises(ValueError, match="policy contract changed"):
        experiment.verify_freeze()


@pytest.mark.parametrize("mutation", ["policy", "membership", "policy_bytes"])
def test_resumed_run_rechecks_freeze_before_writing(experiment, resume, monkeypatch, mutation):
    calls = []

    def changing_freeze():
        calls.append(None)
        policy, selected = copy.deepcopy(resume.frozen), copy.deepcopy(resume.selected)
        if len(calls) > 1:
            if mutation == "policy":
                policy["accuracy_target"] = 0.5
            elif mutation == "membership":
                selected.reverse()
        return policy, selected

    monkeypatch.setattr(experiment, "verify_freeze", changing_freeze)
    if mutation == "policy_bytes":
        hashes = []

        def changed_policy_bytes(_path):
            hashes.append(None)
            return "first-bytes" if len(hashes) == 1 else "edited-bytes"

        monkeypatch.setattr(experiment, "file_sha256", changed_policy_bytes)
    with pytest.raises(ValueError, match="changed during evaluation"):
        experiment.run("comparison", "candidate")
    assert len(calls) == 2
    assert not (resume.root / "reports/v5-comparison-candidate.json").exists()


def test_summary_rechecks_freeze_before_writing(experiment, study, monkeypatch):
    calls = []

    def changed_policy():
        calls.append(None)
        policy = copy.deepcopy(study.frozen)
        if len(calls) > 1:
            policy["accuracy_target"] = 0.5
        return policy, study.selected

    monkeypatch.setattr(experiment, "verify_freeze", changed_policy)
    with pytest.raises(ValueError, match="changed during summary"):
        experiment.summarize()
    assert len(calls) == 2
    assert not (study.root / "reports/v5-summary.json").exists()


def test_identity_fingerprints_the_reused_runtime_verification_driver(experiment, monkeypatch):
    paths = []

    def recorded_hash(path):
        paths.append(path.name)
        return path.name + "-hash"

    monkeypatch.setattr(experiment, "file_sha256", recorded_hash)
    monkeypatch.setattr(
        experiment,
        "engine",
        lambda *_args: SimpleNamespace(identity=lambda: {"model": "synthetic"}),
    )
    monkeypatch.setattr(
        experiment,
        "module",
        lambda _name: SimpleNamespace(environment_identity=lambda: {"python": "synthetic"}),
    )
    result = experiment.identity()
    assert result["runtime_driver_sha256"] == "experiment_v4.py-hash"
    assert "experiment_v4.py" in paths
