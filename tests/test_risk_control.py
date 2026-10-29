"""Selective-risk evidence must fail closed, including apparently perfect small pilots."""

import json

import pytest

from financial_qa.calibration import (
    POLICIES,
    calibrate,
    digest,
    save_certificate,
    upper_error_bound,
    validate_certificate,
)
from financial_qa.data import Document
from financial_qa.executor import ExecutionError, Plan, execute
from financial_qa.model_client import NoRedirect, SafeModelClient, validate_endpoint
from financial_qa.reasoning_v3 import Program, compile_program, financial_catalog


def reports(count, errors=0):
    rows = [
        {"id": f"q{i}", "document_id": f"d{i}", "abstained": False, "correct": i >= errors}
        for i in range(count)
    ]
    return {policy: {"predictions": rows} for policy in POLICIES}


def test_exact_zero_error_bound_and_small_pilot_is_not_enough():
    assert upper_error_bound(0, 100, 0.01) == pytest.approx(1 - 0.01**0.01)
    assert upper_error_bound(0, 0, 0.01) == 1
    assert upper_error_bound(10, 10, 0.01) == 1
    assert not calibrate(reports(10), {})["certified"]
    assert calibrate(reports(100), {})["certified"]


def test_exact_bound_matches_scipy_when_available():
    scipy = pytest.importorskip("scipy.stats")
    for accepted, errors in [(25, 1), (80, 3), (200, 15), (400, 0)]:
        expected = scipy.beta.ppf(0.99, errors + 1, accepted - errors)
        assert upper_error_bound(errors, accepted, 0.01) == pytest.approx(expected)


def test_grid_membership_and_duplicate_document_rejected():
    sample = reports(100)
    del sample[POLICIES[-1]]
    with pytest.raises(ValueError, match="predeclared"):
        calibrate(sample, {})
    sample = reports(100)
    sample[POLICIES[0]]["predictions"][0]["document_id"] = "d1"
    with pytest.raises(ValueError, match="unique document"):
        calibrate(sample, {})


def test_certificate_identity_and_checksum_fail_closed(tmp_path):
    path = tmp_path / "certificate.json"
    identity = {"model_sha256": "modelA", "prompt_sha256": "promptA"}
    certificate = calibrate(reports(100), identity)
    save_certificate(path, certificate)
    assert validate_certificate(path, identity)["certified"]
    with pytest.raises(ValueError, match="identity"):
        validate_certificate(path, {"model_sha256": "modelB"})
    certificate["target_error"] = 0.001
    path.write_text(json.dumps(certificate))
    with pytest.raises(ValueError, match="checksum"):
        validate_certificate(path, identity)
    with pytest.raises(ValueError, match="overwrite"):
        save_certificate(path, certificate)


def test_no_policy_certifies_with_frequent_errors():
    result = calibrate(reports(200, 30), {})
    assert result["selected_policy"] is None
    assert result["state"] == "uncertified_fail_closed"
    assert all(not row["certified"] for row in result["policies"].values())


def program(steps):
    return Program.model_validate({"steps": steps, "abstain": False})


def test_multistep_ratio_of_sums_preserves_source_grounding():
    document = Document(
        "sum-ratio",
        (("USD million", "2023"), ("Debt A", "20"), ("Debt B", "30"), ("Assets", "200")),
        (),
    )
    catalog = financial_catalog(document, document.evidence())
    plan = compile_program(
        program(
            [
                {"operation": "sum", "operands": ["r1c1", "r2c1"]},
                {"operation": "percent_ratio", "operands": ["s0", "r3c1"]},
            ]
        ),
        catalog,
        "What percentage of assets is debt in 2023?",
    )
    assert execute(plan, document.evidence()) == 25
    assert plan.scale == "percent"
    assert plan.citations == ["R1", "R2", "R3"]


def test_program_rejects_forward_references_doublecount_and_currency_mix():
    document = Document(
        "currencies", (("", "2023"), ("USD revenue", "$120"), ("EUR revenue", "€100")), ()
    )
    catalog = financial_catalog(document, document.evidence())
    with pytest.raises(ExecutionError, match="future"):
        compile_program(
            program([{"operation": "sum", "operands": ["s0", "r1c1"]}]), catalog, "Total revenue?"
        )
    with pytest.raises(ExecutionError, match="currencies"):
        compile_program(
            program([{"operation": "difference", "operands": ["r1c1", "r2c1"]}]),
            catalog,
            "Revenue difference?",
        )
    with pytest.raises(ExecutionError, match="double-counts"):
        compile_program(
            program(
                [
                    {"operation": "lookup", "operands": ["r1c1"]},
                    {"operation": "sum", "operands": ["s0", "r1c1"]},
                ]
            ),
            catalog,
            "Revenue total?",
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://user:secret@127.0.0.1:18081/v1",
        "http://127.0.0.1:18081/v1?forward=yes",
        "https://127.0.0.1:18081/v1",
        "http://example.com:18081/v1",
        "http://127.0.0.1:18081/v1#hidden",
    ],
)
def test_private_documents_only_use_clean_loopback_endpoint(url):
    with pytest.raises(ValueError):
        validate_endpoint(url)


def test_redirect_is_rejected_before_followup_request():
    handler = NoRedirect()
    with pytest.raises(ExecutionError, match="redirects"):
        handler.redirect_request(None, None, 302, "Moved", {}, "https://example.com/steal")


def test_model_transport_bounds_response(monkeypatch):
    client = SafeModelClient("http://127.0.0.1:18081/v1")

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, limit):
            return b"x" * limit

    monkeypatch.setattr(client.opener, "open", lambda *args, **kwargs: Response())
    with pytest.raises(ExecutionError, match="response size"):
        client.request(b"{}")


@pytest.mark.parametrize(
    "errors,accepted,alpha",
    [
        (True, 100, 0.01),
        (0, 100.5, 0.01),
        (0.1, 100, 0.01),
        (0, 100, float("nan")),
        (0, 100, float("inf")),
        (0, 100, True),
    ],
)
def test_binomial_rejects_nonexact_counts_and_invalid_tail(errors, accepted, alpha):
    with pytest.raises(ValueError):
        upper_error_bound(errors, accepted, alpha)


def test_certificate_recomputes_counts_bound_and_selection_even_with_updated_checksum(tmp_path):
    certificate = calibrate(reports(100), {})
    path = tmp_path / "evidence.json"
    certificate["policies"]["semantic"]["error_upper_bound"] = 0
    certificate["artifact_sha256"] = digest(
        {k: v for k, v in certificate.items() if k != "artifact_sha256"}
    )
    path.write_text(json.dumps(certificate))
    with pytest.raises(ValueError, match="recomputed"):
        validate_certificate(path, {})


def test_calibration_requires_exact_manifest_and_strict_boolean_outcomes():
    expected = [{"id": f"q{i}", "document_id": f"d{i}"} for i in range(101)]
    with pytest.raises(ValueError, match="manifest"):
        calibrate(reports(100), {"calibration_membership": expected})
    invalid = reports(100)
    invalid["model"]["predictions"][0]["correct"] = "false"
    with pytest.raises(ValueError, match="boolean"):
        calibrate(invalid, {})


@pytest.mark.parametrize("payload", ["null", "[]", '"string"'])
def test_malformed_certificate_shapes_fail_closed(tmp_path, payload):
    path = tmp_path / "malformed.json"
    path.write_text(payload)
    with pytest.raises(ValueError, match="object"):
        validate_certificate(path, {})


def test_program_cache_corruption_abstains_instead_of_raising(monkeypatch, tmp_path):
    from financial_qa.reasoning_v3 import ProgramPlanner, RuntimeProfile
    from financial_qa.service import answer

    client = ProgramPlanner(
        RuntimeProfile("test", "weight-hash"), cache_dir=tmp_path, policy="model"
    )
    content = {
        "steps": [{"operation": "difference", "operands": ["r1c1", "r1c2"]}],
        "abstain": False,
    }
    raw = json.dumps({"choices": [{"message": {"content": json.dumps(content)}}]}).encode()
    monkeypatch.setattr(client.client, "request", lambda _: raw)
    document = Document("cache", (("USD million", "2023", "2022"), ("Revenue", "120", "100")), ())
    assert answer(document, "Revenue change from2022to2023?", client)["answer"] == "20"
    cache = next(tmp_path.glob("*.json"))
    cache.write_text('{"response":null}')
    result = answer(document, "Revenue change from2022to2023?", client)
    assert result["abstained"]
    assert "cache identity" in result["reason"]


def test_risk_controlled_runtime_drift_fails_closed(monkeypatch, tmp_path):
    import hashlib

    from financial_qa.reasoning_v3 import RiskControlledPlanner, RuntimeProfile, runtime_signature

    model = tmp_path / "weights.gguf"
    model.write_bytes(b"test weights")
    data = tmp_path / "data.json"
    data.write_text("[]")
    members = [{"id": f"q{i}", "document_id": f"d{i}"} for i in range(400)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"calibration": members}))
    properties = {
        "model_alias": "test",
        "model_path": str(model),
        "build_info": "testbuild",
        "chat_template": "test template",
        "default_generation_settings": {"n_ctx": 8192, "params": {"reasoning_format": "none"}},
    }
    profile = RuntimeProfile(
        "test",
        hashlib.sha256(model.read_bytes()).hexdigest(),
        runtime=runtime_signature(properties),
    )
    identity = {
        **profile.identity(),
        "calibration_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "calibration_data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "calibration_membership": members,
    }
    certificate_path = tmp_path / "certificate.json"
    save_certificate(certificate_path, calibrate(reports(400), identity))
    monkeypatch.setattr(SafeModelClient, "properties", lambda _: properties)
    engine = RiskControlledPlanner(certificate_path, profile, model, manifest, data)
    assert engine.state["certified"]
    properties["default_generation_settings"]["n_ctx"] = 4096
    document = Document("runtime", (("", "2023"), ("Revenue", "100")), ())
    assert engine.plan(document, "Revenue in2023?", document.evidence()).abstain
    assert engine.state["state"] == "runtime_changed_fail_closed"


@pytest.mark.parametrize("change", ["alias", "path"])
def test_serving_only_model_identity_drift_fails_closed_in_api(tmp_path, change):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from financial_qa.api import create_app
    from financial_qa.risk_serving import ServingIdentityGuard

    properties = {"model_path": str(tmp_path / "model.gguf"), "model_alias": "expected"}
    calls = []

    class Planner:
        engine = SimpleNamespace(
            client=SimpleNamespace(properties=lambda: properties),
            profile=SimpleNamespace(model="expected"),
        )
        state = {"certified": True, "state": "certified_under_iid_assumptions"}

        def plan(self, *_):
            calls.append(True)
            return Plan(abstain=True, reason="underlying decline")

    planner = Planner()
    guard = ServingIdentityGuard(planner, tmp_path / "model.gguf")
    document = Document("serving", (("", "2023"), ("Revenue", "100")), ())
    guard.plan(document, "Revenue in2023?", document.evidence())
    assert len(calls) == 1
    properties["model_alias" if change == "alias" else "model_path"] = "changed"
    assert guard.plan(document, "Revenue in2023?", document.evidence()).abstain
    assert len(calls) == 1
    assert planner.engine is None
    state = TestClient(create_app(guard)).get("/health").json()["reasoning"]
    assert not state["certified"]
    assert state["state"] == "model_identity_changed_fail_closed"
