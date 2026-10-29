import json
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from financial_qa.api import create_app
from financial_qa.benchmark import freeze_manifest, validate_manifest
from financial_qa.context import table_context, variable_catalog
from financial_qa.data import Document, Example, parse_number
from financial_qa.executor import ExecutionError, execute
from financial_qa.retrieval import retrieve
from financial_qa.service import answer
from financial_qa.structured import (
    Calculation,
    HeaderAwareBaseline,
    StructuredLLM,
    compile_calculation,
    response_schema,
)


@pytest.fixture
def multiheader():
    return Document(
        "synthetic-multilevel",
        (
            ("", "Domestic", "", "International", ""),
            ("", "2019", "2018", "2019", "2018"),
            ("Discount rate", "4.00%", "3.75%", "1.90%", "2.80%"),
        ),
        (),
    )


def test_multirow_headers_preserve_parent_and_year(multiheader):
    context = table_context(multiheader)
    assert context.header_rows == (0, 1)
    assert context.columns[2] == "Domestic / 2018"
    assert context.columns[3] == "International / 2019"
    evidence = retrieve(multiheader, "domestic discount rate", 1, "coherent")
    assert {row.id for row in evidence} == {"R0", "R1", "R2"}
    catalog = variable_catalog(multiheader, {row.id for row in evidence})
    assert "r1c1" not in catalog  # Year headers are metadata, not selectable financial operands.
    assert catalog["r2c1"]["scale"] == "percent"


def test_multiheader_year_change_is_supported(multiheader):
    result = answer(
        multiheader,
        "What is the percentage change in domestic discount rate from 2018 to 2019?",
        HeaderAwareBaseline(),
    )
    assert Decimal(result["answer"]) == Decimal(100) / Decimal(15)
    assert result["plan"]["scale"] == "percent"


def test_compiler_derives_citations_and_prevents_unit_relabeling(multiheader):
    catalog = variable_catalog(multiheader, {"R0", "R1", "R2"})
    calculation = Calculation(operation="difference", operands=["r2c1", "r2c3"], scale="million")
    plan = compile_calculation(calculation, catalog)
    assert plan.citations == ["R2"]
    assert plan.scale == "percent"
    assert execute(plan, multiheader.evidence()) == Decimal("2.10")


@pytest.mark.parametrize("operation", ["difference", "percent_change", "percent_ratio", "divide"])
def test_mixed_source_units_rejected_before_every_operation(operation):
    catalog = {
        "r1c1": {"source": "R1", "scale": "million"},
        "r1c2": {"source": "R1", "scale": "thousand"},
    }
    with pytest.raises(ExecutionError, match="incompatible"):
        compile_calculation(
            Calculation(operation=operation, operands=list(catalog), scale=""), catalog
        )


def test_currency_and_accounting_negative_formats():
    assert parse_number("€ 1,234.5") == Decimal("1234.5")
    assert parse_number("(4.5)%") == Decimal("-4.5")


def test_schema_only_allows_existing_operands():
    schema = response_schema({"r1c1": {}, "r1c2": {}})
    properties = schema["json_schema"]["schema"]["properties"]
    assert properties["operands"]["items"]["enum"] == ["r1c1", "r1c2"]
    assert "citations" not in properties
    assert "reason" not in properties


def test_constrained_llm_request_and_deterministic_compilation(multiheader, monkeypatch):
    captured = {}

    def request(_, body):
        captured.update(json.loads(body))
        content = {"operation": "percent_change", "operands": ["r2c1", "r2c2"], "scale": "percent"}
        return json.dumps({"choices": [{"message": {"content": json.dumps(content)}}]}).encode()

    monkeypatch.setattr(StructuredLLM, "_request", request)
    result = answer(
        multiheader, "Domestic discount rate change from 2018 to 2019?", StructuredLLM()
    )
    assert not result["abstained"]
    assert captured["response_format"]["type"] == "json_schema"
    assert result["plan"]["citations"] == ["R2"]


def examples():
    return [
        Example(
            Document(f"doc{index}", (("Year", "2023"), ("Revenue", "100")), ()),
            f"q{index}",
            "What is revenue in 2023?",
            100,
            "",
            "arithmetic",
        )
        for index in range(180)
    ]


def test_manifest_is_disjoint_and_reuse_validates_protocol(tmp_path):
    source = examples()
    path = tmp_path / "manifest.json"
    manifest = freeze_manifest(source, path, size=50)
    selected = {row["document_id"] for row in manifest["examples"]}
    assert selected.isdisjoint(manifest["excluded_document_ids"])
    assert len(selected) == 50
    assert freeze_manifest(source, path, size=50) == manifest
    with pytest.raises(ValueError, match="seed"):
        freeze_manifest(source, path, size=50, seed=99)
    invalid = {**manifest, "examples": manifest["examples"][:-1]}
    with pytest.raises(ValueError, match="unique"):
        validate_manifest(source, invalid, 50, 71)


def test_real_context_endpoint_excludes_gold_labels(tmp_path):
    record = {
        "table": {"uid": "doc", "table": [["Year", "2023"], ["Revenue", "100"]]},
        "paragraphs": [{"text": "A source."}],
        "questions": [
            {
                "uid": "q",
                "question": "What was revenue in 2023?",
                "answer": "SECRET_GOLD",
                "derivation": "SECRET_DERIVATION",
                "answer_type": "arithmetic",
            }
        ],
    }
    path = tmp_path / "data.json"
    path.write_text(json.dumps([record]))
    client = TestClient(create_app(data_path=path))
    assert client.get("/examples").json()[0]["id"] == "q"
    response = client.get("/examples/q")
    assert "SECRET" not in response.text
    assert "source" in response.json()
    assert client.get("/examples/unknown").status_code == 404
