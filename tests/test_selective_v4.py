"""Selective policy support, tamper checks and fail-closed serving behavior."""

import json

import pytest

from financial_qa.data import Document
from financial_qa.executor import Plan
from financial_qa.reasoning_v3 import RuntimeProfile
from financial_qa.reasoning_v4 import LayoutProgramPlanner
from financial_qa.selective_v4 import (
    TemporalRiskPlanner,
    acceptance_plan,
    certificate_from_evidence,
    validate_certificate,
)


@pytest.fixture
def document():
    return Document(
        "synthetic",
        (("USD million", "2023", "2022"), ("Revenue", "120", "100"), ("Costs", "30", "20")),
        (),
    )


def test_gate_accepts_supported_same_row_agreement_and_rejects_unsupported_selection(document):
    question = "What was the percentage change in revenue from 2022 to 2023?"
    plan = Plan(expression="(r1c1-r1c2)/r1c2*100", scale="percent", citations=["R1"])
    assert acceptance_plan(document, question, document.evidence(), plan) == plan
    wrong_metric = Plan(expression="(r2c1-r2c2)/r2c2*100", scale="percent", citations=["R2"])
    assert acceptance_plan(document, question, document.evidence(), wrong_metric).abstain
    assert acceptance_plan(
        document, "What is the share of revenue?", document.evidence(), plan
    ).abstain


def evidence(accepted=100, errors=0):
    membership = [{"id": f"q-{i}", "document_id": f"context-{i}"} for i in range(300)]
    rows = [
        {**row, "accepted": i < accepted, "correct": errors <= i < accepted}
        for i, row in enumerate(membership)
    ]
    return membership, rows


def test_support_and_error_bound_are_both_required():
    membership, rows = evidence(20)
    small = certificate_from_evidence(rows, {}, membership)
    assert not small["certified"] and small["error_upper_bound"] > 0.10
    membership, rows = evidence(100)
    strong = certificate_from_evidence(rows, {}, membership)
    assert strong["certified"] and strong["error_upper_bound"] < 0.10
    membership, rows = evidence(100, 20)
    assert not certificate_from_evidence(rows, {}, membership)["certified"]


@pytest.mark.parametrize("mutation", ["truncate", "duplicate", "boolean", "impossible"])
def test_bad_calibration_evidence_cannot_qualify(mutation):
    membership, rows = evidence()
    if mutation == "truncate":
        membership, rows = membership[:100], rows[:100]
    elif mutation == "duplicate":
        rows[1]["document_id"] = rows[0]["document_id"]
    elif mutation == "boolean":
        rows[0]["accepted"] = 1
    else:
        rows[-1]["correct"] = True
    with pytest.raises(ValueError):
        certificate_from_evidence(rows, {}, membership)


def test_rehashed_counts_cannot_replace_recomputed_evidence(tmp_path):
    from financial_qa.calibration import digest, file_sha256

    membership, rows = evidence()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"examples": membership}))
    data = tmp_path / "source.json"
    data.write_text("[]")
    identity = {"engine": "synthetic"}
    full_identity = {
        **identity,
        "manifest_sha256": file_sha256(manifest),
        "data_sha256": file_sha256(data),
    }
    cert = certificate_from_evidence(rows, full_identity, membership)
    path = tmp_path / "certificate.json"
    path.write_text(json.dumps(cert))
    assert validate_certificate(path, identity, manifest, data)["certified"]
    cert["accepted"] = 200
    cert["artifact_sha256"] = digest({k: v for k, v in cert.items() if k != "artifact_sha256"})
    path.write_text(json.dumps(cert))
    with pytest.raises(ValueError, match="recomputed"):
        validate_certificate(path, identity, manifest, data)


def test_unavailable_certificate_starts_without_calling_a_model(document, tmp_path):
    engine = LayoutProgramPlanner(RuntimeProfile("model", "a" * 64))
    engine.client.request = lambda _body: pytest.fail("Unavailable policy called the model")
    guarded = TemporalRiskPlanner(
        engine,
        tmp_path / "absent.json",
        tmp_path / "manifest.json",
        tmp_path / "data.json",
        tmp_path / "model.gguf",
    )
    assert guarded.engine is None and not guarded.state["certified"]
    assert guarded.plan(document, "Revenue change from 2022 to 2023?", document.evidence()).abstain


def test_v4_guarded_cli_missing_configuration_is_explicitly_fail_closed(tmp_path):
    from financial_qa.cli import inference_metadata, make_planner

    guarded = make_planner(
        "risk-controlled-v4", None, None, runtime_profile=tmp_path / "absent-runtime.json"
    )
    assert guarded.state["version"] == 4 and guarded.state["certified"] is False
    assert guarded.state["state"].endswith("fail_closed")
    assert inference_metadata(guarded, "risk-controlled-v4")["generation"] is None


def test_calibration_checkpoint_supports_full_evidence_but_bounds_actual_reads(tmp_path):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/calibrate_v4.py"
    spec = importlib.util.spec_from_file_location("calibration_runner_under_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"predictions": [{"details": "x" * 400}] * 300}))
    assert checkpoint.stat().st_size > 100_000
    assert len(runner.read_checkpoint(checkpoint)["predictions"]) == 300
    checkpoint.write_bytes(b" " * 256_001)
    with pytest.raises(ValueError, match="budget"):
        runner.read_checkpoint(checkpoint)


@pytest.mark.parametrize("drift", ["alias", "runtime"])
def test_guarded_runtime_drift_revokes_even_after_metadata_is_restored(tmp_path, document, drift):
    import hashlib

    from financial_qa.calibration import file_sha256
    from financial_qa.reasoning_v3 import runtime_signature
    from financial_qa.selective_v4 import policy_identity

    model = tmp_path / "synthetic.gguf"
    model.write_bytes(b"Synthetic test bytes, never an actual language model")
    properties = {
        "model_path": str(model),
        "model_alias": ["test-model"],
        "build_info": "synthetic",
        "chat_template": "synthetic-template",
        "default_generation_settings": {"n_ctx": 8192, "params": {"reasoning_format": "none"}},
    }
    profile = RuntimeProfile(
        "test-model",
        hashlib.sha256(model.read_bytes()).hexdigest(),
        runtime=runtime_signature(properties),
    )
    engine = LayoutProgramPlanner(profile)
    engine.client.properties = lambda: properties
    proposal = Plan(expression="(r1c1-r1c2)/r1c2*100", citations=["R1"], scale="percent")
    engine.plan = lambda *_args: proposal
    membership, rows = evidence()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"examples": membership}))
    source = tmp_path / "source.json"
    source.write_text("[]")
    identity = {
        **policy_identity(engine),
        "manifest_sha256": file_sha256(manifest),
        "data_sha256": file_sha256(source),
    }
    certificate = tmp_path / "certificate.json"
    certificate.write_text(json.dumps(certificate_from_evidence(rows, identity, membership)))
    guarded = TemporalRiskPlanner(engine, certificate, manifest, source, model)
    assert guarded.state["certified"]
    question = "Percentage change in revenue from 2022 to 2023?"
    assert guarded.plan(document, question, document.evidence()) == proposal
    original = json.loads(json.dumps(properties))
    if drift == "alias":
        properties["model_alias"] = ["different-model"]
    else:
        properties["build_info"] = "different-build"
    assert guarded.plan(document, question, document.evidence()).abstain
    assert guarded.state["state"] == "runtime_changed_fail_closed"
    properties.clear()
    properties.update(original)
    assert guarded.engine is None
    assert guarded.plan(document, question, document.evidence()).abstain


@pytest.mark.parametrize(
    "question",
    [
        "What percentage of assets is debt in 2023?",
        "What was the change in revenue?",
        "What was the change in revenue from 2020 to 2023 including 2021?",
    ],
)
def test_necessary_question_prefilter_agrees_with_full_gate_without_a_model(document, question):
    from financial_qa.selective_v4 import eligible_question

    proposal = Plan(expression="r1c1-r1c2", citations=["R1"], scale="million")
    assert not eligible_question(question)
    assert acceptance_plan(document, question, document.evidence(), proposal).abstain


def test_prefilter_can_reject_unsupported_question_without_generation(tmp_path, document):
    guarded = TemporalRiskPlanner.__new__(TemporalRiskPlanner)
    guarded.engine = type(
        "Engine", (), {"plan": lambda *_args: pytest.fail("Unneeded model call")}
    )()
    guarded.state = {"version": 4, "certified": True}
    guarded._check_runtime = lambda _engine: None
    result = guarded.plan(
        document, "Revenue as a share of total assets in 2023?", document.evidence()
    )
    assert result.abstain
