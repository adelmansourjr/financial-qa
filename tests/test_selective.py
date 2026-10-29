"""Regression tests for semantic scope, unit propagation and artifact safety."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from financial_qa.api import create_app
from financial_qa.data import Document
from financial_qa.executor import ExecutionError
from financial_qa.selective import (
    SelectiveLLM,
    SemanticBaseline,
    source_catalog,
    verify_selection,
)
from financial_qa.service import answer
from financial_qa.storage import download
from financial_qa.structured import Calculation


@pytest.fixture
def document():
    return Document(
        "scope",
        (
            ("$ millions", "2023", "2022"),
            ("Revenue", "120", "100"),
            ("Diluted earnings per share", "2.50", "2.00"),
            ("Tax percent", "20%", "18%"),
        ),
        (),
    )


def test_year_average_never_invents_an_observation(document):
    catalog = source_catalog(document, document.evidence())
    proposal = Calculation(operation="average", operands=["r1c1", "r1c2"], scale="million")
    with pytest.raises(ExecutionError, match="years|unspecified"):
        verify_selection(
            proposal, catalog, "What is average revenue in2023?".replace("in2023", "in 2023")
        )
    assert answer(document, "What is average revenue in 2023?", SemanticBaseline())["abstained"]
    assert (
        answer(
            document,
            "What is average revenue in 2023 and2022?".replace("and2022", "and 2022"),
            SemanticBaseline(),
        )["answer"]
        == "110"
    )


def test_years_in_rows_are_supported():
    document = Document("transpose", (("", "Revenue"), ("2023", "120"), ("2022", "100")), ())
    result = answer(document, "Revenue change from 2022 to 2023?", SemanticBaseline())
    assert result["answer"] == "20"


def test_source_unit_exceptions_and_mixed_unknown_units(document):
    catalog = source_catalog(document, document.evidence())
    assert catalog["r1c1"]["scale"] == "million"
    assert catalog["r2c1"]["scale"] == ""
    assert catalog["r3c1"]["scale"] == "percent"
    with pytest.raises(ExecutionError, match="units"):
        verify_selection(
            Calculation(operation="difference", operands=["r1c1", "r2c1"], scale=""),
            catalog,
            "What is the difference?",
        )


def test_duplicate_operands_and_operator_mismatch_rejected(document):
    catalog = source_catalog(document, document.evidence())
    with pytest.raises(ExecutionError, match="Repeated"):
        verify_selection(
            Calculation(operation="difference", operands=["r1c1", "r1c1"], scale=""),
            catalog,
            "Revenue change?",
        )
    with pytest.raises(ExecutionError, match="operator"):
        verify_selection(
            Calculation(operation="percent_change", operands=["r3c1", "r3c2"], scale="percent"),
            catalog,
            "Difference in tax percent?",
        )


def test_model_cannot_invent_units_and_cache_is_reused(document, monkeypatch, tmp_path):
    calls = []

    def request(_, body):
        calls.append(json.loads(body))
        result = {"operation": "lookup", "operands": ["r2c1"], "scale": "billion"}
        return json.dumps({"choices": [{"message": {"content": json.dumps(result)}}]}).encode()

    monkeypatch.setattr(SelectiveLLM, "_request", request)
    engine = SelectiveLLM(cache_dir=tmp_path)
    for _ in range(2):
        result = answer(document, "What are diluted earnings per share in 2023?", engine)
        assert result["answer"] == "2.50"
        assert result["plan"]["scale"] == ""
    assert len(calls) == 1
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert "answer" not in calls[0]["messages"][-1]["content"]


def test_download_enforces_https_across_redirects(monkeypatch, tmp_path):
    captured = []

    def run(command, **_):
        captured.extend(command)
        Path(command[command.index("--output") + 1]).write_text("public data")

    monkeypatch.setattr("financial_qa.storage.subprocess.run", run)
    monkeypatch.setattr("financial_qa.storage.shutil.which", lambda _: "/usr/bin/curl")
    download("https://example.com/data.json", tmp_path / "data.json")
    assert captured[captured.index("--proto") + 1] == "=https"
    assert captured[captured.index("--proto-redir") + 1] == "=https"
    with pytest.raises(ValueError, match="HTTPS"):
        download("http://example.com/data.json", tmp_path / "unsafe.json")


def test_static_assets_are_served():
    client = TestClient(create_app())
    assert client.get("/static/index.html").status_code == 200
    assert client.get("/static/../api.py").status_code == 404


def test_year_valued_metric_is_not_mistaken_for_header():
    document = Document("year-like", (("", "2023", "2022"), ("Revenue", "2000", "1900")), ())
    catalog = source_catalog(document, document.evidence())
    assert catalog["r1c1"]["value"] == "2000"
    assert (
        answer(document, "Revenue change from 2022 to 2023?", SemanticBaseline())["answer"] == "100"
    )


def test_unit_labeled_year_header_and_percent_columns():
    document = Document(
        "units",
        (("", "Fiscal year", ""), ("(in millions)", "2023", "2022"), ("Revenue", "120", "100")),
        (),
    )
    catalog = source_catalog(document, document.evidence())
    assert "r1c1" not in catalog
    assert catalog["r2c1"]["scale"] == "million"


def test_weighted_average_metric_does_not_override_explicit_change():
    from financial_qa.selective import operation_family

    assert operation_family("Percentage change in weighted average shares?") == "percent_change"


def test_frozen_protocols_keep_documents_disjoint():
    root = Path(__file__).parents[1]
    training = json.loads((root / "benchmarks/train-v2.json").read_text())
    development = {row["document_id"] for row in training["development"]}
    validation = {row["document_id"] for row in training["validation"]}
    assert len(development) == 30
    assert len(validation) == 40
    assert development.isdisjoint(validation)
    prior = json.loads((root / "benchmarks/dev-holdout-v1.json").read_text())
    final = json.loads((root / "benchmarks/dev-holdout-v2.json").read_text())
    used = set(prior["excluded_document_ids"]) | {row["document_id"] for row in prior["examples"]}
    selected = {row["document_id"] for row in final["examples"]}
    assert len(selected) == final["size"] == 100
    assert selected.isdisjoint(used)
    assert used <= set(final["excluded_document_ids"])
