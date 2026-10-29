"""Regression checks for layout-preserving program proposals, not model accuracy."""

import json
from decimal import Decimal

import pytest

from financial_qa.cli import inference_metadata, make_planner, model_defaults
from financial_qa.data import Document
from financial_qa.executor import ExecutionError, execute
from financial_qa.reasoning_v3 import Program, RuntimeProfile, Step, financial_catalog
from financial_qa.reasoning_v4 import (
    LayoutProgramPlanner,
    compile_v4,
    explicit_operation,
    response_schema,
)


@pytest.fixture
def shares():
    return Document(
        "synthetic",
        (("Shares in millions", "2023", "2022"), ("Weighted-average diluted shares", "120", "100")),
        (),
    )


def test_metric_word_does_not_rewrite_proposed_operation(shares):
    evidence = shares.evidence()
    catalog = financial_catalog(shares, evidence)
    plan = compile_v4(
        Program(steps=[Step(operation="difference", operands=["r1c1", "r1c2"])], abstain=False),
        catalog,
        "How much did weighted-average diluted shares change from 2022 to 2023?",
    )
    assert execute(plan, evidence) == Decimal(20)


def test_reverse_time_comparison_is_not_silently_sorted(shares):
    program = Program(
        steps=[Step(operation="difference", operands=["r1c2", "r1c1"])], abstain=False
    )
    plan = compile_v4(
        program,
        financial_catalog(shares, shares.evidence()),
        "How many fewer shares were there in 2022 than 2023?",
    )
    assert execute(plan, shares.evidence()) == Decimal(-20)


def test_compound_ratio_retains_intermediate_program():
    document = Document(
        "synthetic",
        (
            ("USD million", "2023"),
            ("Short-term debt", "20"),
            ("Long-term debt", "30"),
            ("Assets", "100"),
        ),
        (),
    )
    program = Program(
        steps=[
            Step(operation="sum", operands=["r1c1", "r2c1"]),
            Step(operation="percent_ratio", operands=["s0", "r3c1"]),
        ],
        abstain=False,
    )
    plan = compile_v4(
        program,
        financial_catalog(document, document.evidence()),
        "What percentage of assets is debt including both short- and long-term?",
    )
    assert execute(plan, document.evidence()) == Decimal(50)
    assert plan.scale == "percent"


def test_grammar_has_correct_operator_specific_arities():
    schema = response_schema(["r1c1", "r1c2"])["json_schema"]["schema"]
    assert next(iter(schema["properties"])) == "analysis"
    variants = schema["$defs"]["Step"]["oneOf"]
    by_operator = {
        v["properties"]["operation"]["const"]: v["properties"]["operands"] for v in variants
    }
    assert by_operator["lookup"]["maxItems"] == 1
    assert by_operator["divide"]["minItems"] == by_operator["divide"]["maxItems"] == 2
    assert by_operator["sum"]["maxItems"] == 8
    assert set(by_operator["lookup"]["items"]["enum"]) == {"r1c1", "r1c2", "s0", "s1", "s2"}


def test_forward_reference_and_mixed_currency_still_rejected(shares):
    catalog = financial_catalog(shares, shares.evidence())
    program = Program(steps=[Step(operation="divide", operands=["s0", "r1c2"])], abstain=False)
    with pytest.raises(ExecutionError, match="future step"):
        compile_v4(program, catalog, "Ratio in 2023")
    catalog["r1c1"]["currency"], catalog["r1c2"]["currency"] = "USD", "EUR"
    program.steps[0] = Step(operation="difference", operands=["r1c1", "r1c2"])
    with pytest.raises(ExecutionError, match="currencies"):
        compile_v4(program, catalog, "Difference")


def test_layout_prompt_is_label_free_and_raw_response_is_cached(shares, tmp_path):
    planner = LayoutProgramPlanner(RuntimeProfile("test-model", "a" * 64), cache_dir=tmp_path)
    sent = []

    def request(body):
        sent.append(json.loads(body))
        return json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "analysis": "Compare the two explicit source periods.",
                                    "steps": [
                                        {"operation": "difference", "operands": ["r1c1", "r1c2"]}
                                    ],
                                    "abstain": False,
                                }
                            )
                        },
                    }
                ]
            }
        ).encode()

    planner.client.request = request
    first = planner.plan(shares, "Change from 2022 to 2023?", shares.evidence())
    second = planner.plan(shares, "Change from 2022 to 2023?", shares.evidence())
    assert len(sent) == 1 and first == second
    content = json.loads(sent[0]["messages"][1]["content"])
    assert "120 [r1c1]" in content["table"][1]
    assert not {"answer", "gold", "derivation"}.intersection(content)
    assert planner.state["certified"] is False
    cache = next(tmp_path.glob("*.json"))
    saved = json.loads(cache.read_text())
    saved["model_sha256"] = "b" * 64
    cache.write_text(json.dumps(saved))
    with pytest.raises(ExecutionError, match="identity"):
        planner.plan(shares, "Change from 2022 to 2023?", shares.evidence())


def test_truncated_output_is_rejected(shares):
    planner = LayoutProgramPlanner(RuntimeProfile("test-model", "a" * 64))
    planner.client.request = lambda _body: b'{"choices":[{"finish_reason":"length"}]}'
    with pytest.raises(ExecutionError, match="truncated"):
        planner.plan(shares, "Change from 2022 to 2023?", shares.evidence())


def test_v4_cli_selects_explicit_new_backend_without_claiming_calibration(tmp_path):
    runtime = {
        "build_info": "test",
        "n_ctx": 8192,
        "chat_template_sha256": "a" * 64,
        "default_generation_sha256": "b" * 64,
        "reasoning_format": "none",
    }
    signature = tmp_path / "runtime.json"
    signature.write_text(json.dumps(runtime))
    planner = make_planner("program-v4", None, None, runtime_profile=signature)
    assert isinstance(planner, LayoutProgramPlanner)
    assert model_defaults("program-v4", "benchmark")[0].endswith(":18081/v1")
    metadata = inference_metadata(planner, "program-v4")
    assert metadata["generation"]["version"] == 4
    assert metadata["generation"]["max_tokens"] == 640
    assert metadata["reasoning"]["certified"] is False
    with pytest.raises(ValueError, match="one experimental"):
        make_planner(
            "program-v4", None, None, runtime_profile=signature, program_policy="agreement"
        )


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the change in weighted-average shares?", "difference"),
        ("How many percent of organic change is from acquisitions?", "percent_ratio"),
        ("What is the ratio including both short-term and long-term debt?", None),
        ("What is the total income in 2023?", None),
        ("What is the average number of weighted-average shares from 2021-2023?", "average"),
    ],
)
def test_explicit_operator_respects_compounds_and_metric_names(question, expected):
    assert explicit_operation(question) == expected


def test_explicit_simple_grammar_prevents_duplicate_percentage_transforms():
    schema = response_schema(["r1c1", "r1c2"], "Percentage change from 2022 to 2023?")
    body = schema["json_schema"]["schema"]
    assert body["properties"]["steps"]["maxItems"] == 1
    operations = body["$defs"]["Step"]["oneOf"]
    assert len(operations) == 1
    assert operations[0]["properties"]["operation"]["const"] == "percent_change"


def test_deep_model_response_becomes_controlled_abstention(shares):
    planner = LayoutProgramPlanner(RuntimeProfile("test-model", "a" * 64))
    planner.client.request = lambda _body: b"[" * 3000 + b"]" * 3000
    with pytest.raises(ExecutionError, match="Layout program rejected"):
        planner.plan(shares, "Change from 2022 to 2023?", shares.evidence())
