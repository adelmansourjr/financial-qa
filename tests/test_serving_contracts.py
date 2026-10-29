"""Serving admission and CLI configuration regressions, without live model calls."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from fastapi.testclient import TestClient

from financial_qa.api import create_app
from financial_qa.benchmark import validate_manifest
from financial_qa.cli import (
    inference_metadata,
    load_runtime_profile,
    main,
    make_planner,
    model_defaults,
)
from financial_qa.data import Document, Example
from financial_qa.executor import Plan
from financial_qa.model_client import SafeModelClient

VALID_RUNTIME = {
    "build_info": "synthetic-test-runtime",
    "n_ctx": 8192,
    "chat_template_sha256": "a" * 64,
    "default_generation_sha256": "b" * 64,
    "reasoning_format": "none",
}

REQUEST = {"question": "What is revenue in 2023?", "table": [["", "2023"], ["Revenue", "120"]]}


def test_v5_cli_defaults_and_metadata_identify_experimental_candidate(tmp_path):
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps(VALID_RUNTIME))
    planner = make_planner("program-v5", None, None, runtime_profile=runtime)
    assert model_defaults("program-v5", "benchmark") == (
        "http://127.0.0.1:18081/v1",
        "qwen3-4b-q4-nonthinking",
    )
    metadata = inference_metadata(planner, "program-v5")
    assert metadata["model"] == planner.profile.model
    assert metadata["generation"]["version"] == 5
    assert metadata["generation"]["deterministic_v5_sha256"]
    assert metadata["reasoning"]["certified"] is False
    assert metadata["reasoning"]["state"] == "experimental_v5_not_risk_calibrated"
    with pytest.raises(ValueError, match="model policy"):
        make_planner("program-v5", None, None, program_policy="hybrid", runtime_profile=runtime)


def test_v5_cli_preserves_explicit_endpoint_without_loading_model(tmp_path):
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps(VALID_RUNTIME))
    planner = make_planner(
        "program-v5", "http://127.0.0.1:18888/v1", "custom-alias", runtime_profile=runtime
    )
    assert planner.client.base_url == "http://127.0.0.1:18888/v1"
    assert planner.profile.model == "custom-alias"
    with pytest.raises(ValueError, match="loopback"):
        make_planner("program-v5", "https://example.org/v1", None, runtime_profile=runtime)


def test_single_slot_inference_rejects_overlap_without_blocking_health():
    entered, release = Event(), Event()
    calls = []

    class SlowPlanner:
        def plan(self, *_):
            calls.append(True)
            entered.set()
            assert release.wait(5)
            return Plan(expression="r1c1", citations=["R1"])

    with TestClient(create_app(SlowPlanner(), data_path=Path("/nonexistent"))) as client:
        with ThreadPoolExecutor(max_workers=1) as pool:
            running = pool.submit(client.post, "/answer", json=REQUEST)
            try:
                assert entered.wait(3)
                assert client.get("/health").status_code == 200
                rejected = client.post("/answer", json=REQUEST)
                assert rejected.status_code == 503
                assert rejected.headers["retry-after"] == "1"
                assert len(calls) == 1
            finally:
                release.set()
            assert running.result(timeout=5).json()["answer"] == "120"
        assert client.post("/answer", json=REQUEST).status_code == 200


def test_inference_slot_is_released_after_planner_failure():
    class RecoveringPlanner:
        calls = 0

        def plan(self, *_):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("Unexpected planner failure")
            return Plan(expression="r1c1", citations=["R1"])

    with TestClient(create_app(RecoveringPlanner()), raise_server_exceptions=False) as client:
        assert client.post("/answer", json=REQUEST).status_code == 500
        response = client.post("/answer", json=REQUEST)
        assert response.status_code == 200
        assert response.json()["answer"] == "120"


@pytest.mark.parametrize("profile_contents", [None, "null", "[]", "{invalid"])
def test_unavailable_risk_configuration_starts_fail_closed(tmp_path, profile_contents):
    profile = tmp_path / "runtime.json"
    if profile_contents is not None:
        profile.write_text(profile_contents)
    engine = make_planner("risk-controlled", None, None, runtime_profile=profile)
    client = TestClient(create_app(engine, data_path=tmp_path / "missing-data.json"))
    state = client.get("/health").json()["reasoning"]
    assert not state["certified"]
    assert state["state"] == "configuration_unavailable_fail_closed"
    prediction = client.post("/answer", json=REQUEST).json()
    assert prediction["abstained"]
    assert prediction["answer"] is None


def test_cli_program_evaluation_defaults_and_metadata_are_actual_profile(monkeypatch, tmp_path):
    captured = {}

    def request(self, body):
        captured["endpoint"] = self.base_url
        captured["request"] = json.loads(body)
        program = {
            "steps": [{"operation": "difference", "operands": ["r1c1", "r1c2"]}],
            "abstain": False,
        }
        return json.dumps({"choices": [{"message": {"content": json.dumps(program)}}]}).encode()

    monkeypatch.setattr(SafeModelClient, "request", request)
    source = [
        {
            "table": {
                "uid": "source",
                "table": [["USD million", "2023", "2022"], ["Revenue", "120", "100"]],
            },
            "paragraphs": [],
            "questions": [
                {
                    "uid": "question",
                    "question": "Revenue change from 2022 to 2023?",
                    "answer": 20,
                    "scale": "million",
                    "answer_type": "arithmetic",
                }
            ],
        }
    ]
    data, runtime, output = (
        tmp_path / name for name in ("data.json", "runtime.json", "result.json")
    )
    data.write_text(json.dumps(source))
    runtime.write_text(json.dumps(VALID_RUNTIME))
    monkeypatch.setattr(
        "sys.argv",
        [
            "financial-qa",
            "evaluate",
            "--backend",
            "program-v3",
            "--data",
            str(data),
            "--runtime-profile",
            str(runtime),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output",
            str(output),
        ],
    )
    main()
    report = json.loads(output.read_text())
    assert captured["endpoint"] == "http://127.0.0.1:18081/v1"
    assert captured["request"]["model"] == report["model"] == "qwen3-4b-q4-nonthinking"
    assert report["generation"]["temperature"] == captured["request"]["temperature"] == 0.7
    assert report["generation"]["max_tokens"] == captured["request"]["max_tokens"] == 240
    assert report["generation"]["model_sha256"]
    assert report["accuracy"] == 1


def test_program_cli_preserves_explicit_endpoint_and_model(tmp_path):
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps(VALID_RUNTIME))
    planner = make_planner(
        "program-v3", "http://127.0.0.1:8080/v1", "custom-alias", runtime_profile=runtime
    )
    assert planner.client.base_url == "http://127.0.0.1:8080/v1"
    assert planner.profile.model == "custom-alias"
    with pytest.raises(ValueError, match="loopback"):
        make_planner(
            "program-v3", "http://example.com:8080/v1", "custom-alias", runtime_profile=runtime
        )


def test_v3_manifest_admitted_without_relaxing_exclusion_or_membership():
    examples = [
        Example(
            Document(f"d{i}", (("", "2023"), ("Revenue", "100")), ()),
            f"q{i}",
            "Revenue?",
            100,
            "",
            "arithmetic",
        )
        for i in range(160)
    ]
    manifest = {
        "version": 3,
        "seed": 311,
        "size": 1,
        "excluded_document_ids": [f"d{i}" for i in range(150)],
        "examples": [{"id": "q150", "document_id": "d150"}],
    }
    validate_manifest(examples, manifest, 1, 311)
    invalid = {**manifest, "examples": [{"id": "q149", "document_id": "d149"}]}
    with pytest.raises(ValueError, match="excluded"):
        validate_manifest(examples, invalid, 1, 311)


def test_cli_benchmark_accepts_v3_manifest_with_v3_defaults(monkeypatch, tmp_path):
    captured = {}

    def request(self, body):
        captured["url"] = self.base_url
        content = {
            "steps": [{"operation": "difference", "operands": ["r1c1", "r1c2"]}],
            "abstain": False,
        }
        return json.dumps({"choices": [{"message": {"content": json.dumps(content)}}]}).encode()

    monkeypatch.setattr(SafeModelClient, "request", request)
    records = [
        {
            "table": {
                "uid": f"d{i}",
                "table": [["USD million", "2023", "2022"], ["Revenue", "120", "100"]],
            },
            "paragraphs": [],
            "questions": [
                {
                    "uid": f"q{i}",
                    "question": "Revenue change from 2022 to 2023?",
                    "answer": 20,
                    "scale": "million",
                    "answer_type": "arithmetic",
                }
            ],
        }
        for i in range(102)
    ]
    data, manifest, runtime, output = (
        tmp_path / name for name in ("data.json", "manifest.json", "runtime.json", "report.json")
    )
    data.write_text(json.dumps(records))
    manifest.write_text(
        json.dumps(
            {
                "version": 3,
                "seed": 311,
                "size": 1,
                "excluded_document_ids": [f"d{i}" for i in range(101)],
                "examples": [{"id": "q101", "document_id": "d101"}],
            }
        )
    )
    runtime.write_text(json.dumps(VALID_RUNTIME))
    monkeypatch.setattr(
        "sys.argv",
        [
            "financial-qa",
            "benchmark",
            "--backend",
            "program-v3",
            "--data",
            str(data),
            "--manifest",
            str(manifest),
            "--runtime-profile",
            str(runtime),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output",
            str(output),
        ],
    )
    main()
    report = json.loads(output.read_text())
    assert captured["url"] == "http://127.0.0.1:18081/v1"
    assert report["model"] == "qwen3-4b-q4-nonthinking"
    assert report["protocol"] == "program-v3"
    assert report["accuracy"] == 1
    assert report["generation"]["max_tokens"] == 240


def test_runtime_signature_rejects_wrong_types_fields_and_oversize_metadata(tmp_path):
    profile = tmp_path / "runtime.json"
    invalid = [
        {"bad": 0},
        {**VALID_RUNTIME, "n_ctx": True},
        {**VALID_RUNTIME, "n_ctx": 8192.0},
        {**VALID_RUNTIME, "n_ctx": 0},
        {**VALID_RUNTIME, "chat_template_sha256": "not-a-hash"},
        {**VALID_RUNTIME, "reasoning_format": None},
        {**VALID_RUNTIME, "extra": "unrecognized"},
    ]
    for payload in invalid:
        profile.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="Runtime profile"):
            make_planner("program-v3", None, None, runtime_profile=profile)
        controlled = make_planner("risk-controlled", None, None, runtime_profile=profile)
        assert controlled.state["state"] == "configuration_unavailable_fail_closed"
    profile.write_bytes(b" " * 16_385)
    with pytest.raises(ValueError, match="size budget"):
        load_runtime_profile(profile)
    assert not make_planner("risk-controlled", None, None, runtime_profile=profile).state[
        "certified"
    ]
    profile.write_text("[" * 1200 + "]" * 1200)
    assert (
        make_planner("risk-controlled", None, None, runtime_profile=profile).state["state"]
        == "configuration_unavailable_fail_closed"
    )


def test_runtime_signature_accepts_recorded_and_compatible_explicit_profiles(tmp_path):
    recorded = Path(__file__).parents[1] / "benchmarks/runtime-v3.json"
    assert load_runtime_profile(recorded) == json.loads(recorded.read_text())
    profile = tmp_path / "future.json"
    compatible = {**VALID_RUNTIME, "build_info": "future-compatible-build", "n_ctx": 32768}
    profile.write_text(json.dumps(compatible))
    assert (
        make_planner("program-v3", None, None, runtime_profile=profile).profile.runtime
        == compatible
    )
