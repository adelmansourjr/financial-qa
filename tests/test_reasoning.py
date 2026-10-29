import json
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from financial_qa.api import create_app
from financial_qa.data import Document, load_examples, parse_number
from financial_qa.evaluation import numeric_match
from financial_qa.executor import ExecutionError, Plan, execute
from financial_qa.planner import LocalLLM, TableBaseline, build_messages
from financial_qa.retrieval import retrieve
from financial_qa.service import answer


@pytest.fixture
def document():
    return Document(
        "synthetic",
        (
            ("USD million", "2023", "2022"),
            ("Revenue", "120", "100"),
            ("Operating income", "30", "25"),
        ),
        (),
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("(1,234.50)", Decimal("-1234.50")),
        ("$12.5%", Decimal("12.5")),
        ("−3", Decimal(-3)),
        ("N/A", None),
        ("NaN", None),
        ("Revenue 20", None),
    ],
)
def test_financial_number_parsing(text, expected):
    assert parse_number(text) == expected


def test_end_to_end_calculation_and_citations(document):
    result = answer(
        document, "What was the percentage increase in revenue from 2022 to 2023?", TableBaseline()
    )
    assert Decimal(result["answer"]) == 20
    assert result["plan"]["scale"] == "percent"
    assert set(result["plan"]["citations"]) == {"R0", "R1"}


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "r1c1.__class__",
        "r1c1 ** 100",
        "sum([r1c1])",
        "r1c1 / 0",
        "r1c1 + 999",
        "True + r1c1",
        "r9c9",
        "100",
    ],
)
def test_untrusted_expressions_are_rejected(document, expression):
    with pytest.raises(ExecutionError):
        execute(Plan(expression=expression, citations=["R1"]), document.evidence())


def test_citation_must_support_every_operand(document):
    with pytest.raises(ExecutionError, match="no cited source"):
        execute(Plan(expression="r1c1-r2c1", citations=["R1"]), document.evidence())
    with pytest.raises(ExecutionError, match="does not refer"):
        execute(Plan(expression="r1c1", citations=["R999"]), document.evidence())


def test_prompt_receives_no_gold_annotations(tmp_path):
    record = {
        "table": {"uid": "doc", "table": [["Year", "2023"], ["Revenue", "100"]]},
        "paragraphs": [{"text": "Public evidence."}],
        "questions": [
            {
                "uid": "q",
                "question": "What was revenue in 2023?",
                "answer": "SECRET_GOLD",
                "derivation": "SECRET_DERIVATION",
            }
        ],
    }
    path = tmp_path / "data.json"
    path.write_text(json.dumps([record]))
    example = load_examples(path)[0]
    messages = json.dumps(build_messages(example.question, example.document.evidence()))
    assert "SECRET_GOLD" not in messages
    assert "SECRET_DERIVATION" not in messages
    assert not hasattr(example.document, "answer")


def test_scale_scoring_is_not_plain_string_matching():
    assert numeric_match("1", "million", 1000, "thousand")
    assert numeric_match("20", "percent", 0.2, "")
    assert not numeric_match("20", "", 20, "percent")


def test_api_validates_and_returns_an_answer(document):
    client = TestClient(create_app())
    assert client.get("/").status_code == 200
    response = client.post(
        "/answer", json={"question": "What was revenue in 2023?", "table": document.table}
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "120"
    assert client.post("/answer", json={"question": "x", "table": []}).status_code == 422
    assert client.get("/health").json()["backend"] == "SemanticBaseline"


def test_unknown_question_abstains(document):
    assert answer(document, "What will the stock price be tomorrow?", TableBaseline())["abstained"]
    with pytest.raises(ValueError):
        retrieve(document, "revenue", top_k=0)


def test_missing_requested_year_cannot_silently_become_a_lookup(document):
    result = answer(document, "What was the revenue increase from 2021 to 2023?", TableBaseline())
    assert result["abstained"]
    assert "absent" in result["reason"]


def test_api_rejects_ragged_documents():
    client = TestClient(create_app())
    response = client.post(
        "/answer", json={"question": "What is revenue?", "table": [["2023", "2022"], ["100"]]}
    )
    assert response.status_code == 422


def test_local_llm_protocol_and_validation(document, monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, _):
            plan = {"expression": "r1c1", "citations": ["R1"], "scale": "million"}
            return json.dumps({"choices": [{"message": {"content": json.dumps(plan)}}]}).encode()

    def request(value, timeout):
        captured.update(json.loads(value.data))
        assert timeout == 120
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", request)
    result = answer(document, "What was revenue in 2023?", LocalLLM())
    assert result["answer"] == "120"
    assert captured["temperature"] == 0
    assert captured["response_format"]["type"] == "json_object"
    with pytest.raises(ValueError, match="loopback"):
        LocalLLM("https://example.com/v1")


def test_llm_caches_the_actual_response(document, monkeypatch, tmp_path):
    calls = []
    raw = json.dumps(
        {
            "choices": [
                {"message": {"content": json.dumps({"expression": "r1c1", "citations": ["R1"]})}}
            ]
        }
    ).encode()

    def request(_, body):
        calls.append(body)
        return raw

    monkeypatch.setattr(LocalLLM, "_request", request)
    planner = LocalLLM(cache_dir=tmp_path)
    first = answer(document, "What was revenue in 2023?", planner)
    second = answer(document, "What was revenue in 2023?", planner)
    assert first["answer"] == second["answer"] == "120"
    assert len(calls) == 1
    assert json.loads(next(tmp_path.iterdir()).read_text())["response"] == raw.decode()


def test_malformed_llm_output_is_an_abstention(document, monkeypatch):
    monkeypatch.setattr(
        LocalLLM, "_request", lambda *_: b'{"choices":[{"message":{"content":null}}]}'
    )
    assert answer(document, "What was revenue in 2023?", LocalLLM())["abstained"]
