"""Synthetic safety/behavior checks; these do not measure real-model accuracy."""

import json
from decimal import Decimal

import pytest

from financial_qa.data import Document
from financial_qa.executor import ExecutionError, execute
from financial_qa.grounding_v5 import layout_catalog
from financial_qa.reasoning_v3 import Program, RuntimeProfile, Step
from financial_qa.reasoning_v5 import (
    CandidateProgramPlanner,
    compile_v5,
    normalize_program,
    response_schema,
    source_payload,
    source_references,
)


@pytest.fixture(autouse=True)
def model_path_only(monkeypatch):
    """These tests isolate proposal/repair; exact rules have separate coverage."""
    monkeypatch.setattr("financial_qa.reasoning_v5.exact_plan", lambda *_args: None)


@pytest.fixture
def document():
    return Document(
        "synthetic-no-annotations",
        (("USD million", "2023", "2022", "2021"), ("Revenue", "150", "120", "90")),
        (),
    )


def program(operation, operands):
    return Program(steps=[Step(operation=operation, operands=operands)], abstain=False)


def issued_operands(body, operands):
    """Mocks must choose exactly the descriptive aliases issued by the request."""
    choices = json.loads(body)["response_format"]["json_schema"]["schema"]["$defs"]["Operand"][
        "enum"
    ]
    by_id = {choice.split(" ", 1)[0]: choice for choice in choices}
    return [by_id.get(operand, operand) for operand in operands]


def response(operation="difference", operands=None, abstain=False, body=None):
    proposed = {
        "analysis": "Use the explicitly requested source observations.",
        "steps": []
        if abstain
        else [
            {
                "operation": operation,
                "operands": issued_operands(body, operands or ["r1c1", "r1c2"])
                if body
                else operands or ["r1c1", "r1c2"],
            }
        ],
        "abstain": abstain,
    }
    return json.dumps(
        {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(proposed)}}]}
    ).encode()


def test_schema_retains_compound_operators_for_average_percentage():
    schema = response_schema(["r1c1"])["json_schema"]["schema"]
    assert schema["properties"]["steps"]["maxItems"] == 4
    operations = {v["properties"]["operation"]["const"] for v in schema["$defs"]["Step"]["oneOf"]}
    assert operations == {
        "lookup",
        "sum",
        "average",
        "difference",
        "divide",
        "multiply",
        "percent_ratio",
        "percent_change",
    }


@pytest.mark.parametrize(
    "question,operation",
    [
        ("What is the average effective tax rate, expressed as a percentage?", "average"),
        ("What is the sum of revenue for 2021 and 2022?", "sum"),
        ("What is the percentage change in weighted-average shares?", "percent_change"),
        ("What is the change in revenue?", "difference"),
    ],
)
def test_simple_question_grammar_preserves_main_operator(question, operation):
    schema = response_schema(["r1c1", "r1c2"], question)["json_schema"]["schema"]
    assert schema["properties"]["steps"]["maxItems"] == 1
    assert [v["properties"]["operation"]["const"] for v in schema["$defs"]["Step"]["oneOf"]] == [
        operation
    ]


@pytest.mark.parametrize(
    "question",
    [
        "What is the average percentage of revenue to total assets?",
        "What is the change in average revenue?",
        "What is the ratio of the sum of debt to assets?",
        "What is the difference between average sales and average costs?",
    ],
)
def test_compound_questions_keep_intermediate_operators(question):
    schema = response_schema(["r1c1", "r1c2"], question)["json_schema"]["schema"]
    assert schema["properties"]["steps"]["maxItems"] == 4
    assert len(schema["$defs"]["Step"]["oneOf"]) == 8


@pytest.mark.parametrize(
    "operation,operands",
    [("lookup", ["r1c2"]), ("sum", ["r1c2", "r1c3"]), ("difference", ["r1c1", "r1c2"])],
)
def test_single_year_change_rejects_lookup_sum_or_future_comparator(document, operation, operands):
    with pytest.raises(ExecutionError):
        compile_v5(
            program(operation, operands),
            layout_catalog(document, document.evidence()),
            "What was the change in revenue in 2022?",
        )


def test_natural_language_aggregation_range_includes_intermediate_year(document):
    catalog = layout_catalog(document, document.evidence())
    plan = compile_v5(
        program("average", ["r1c1", "r1c2", "r1c3"]),
        catalog,
        "What was average revenue from 2021 to 2023?",
    )
    assert execute(plan, document.evidence()) == Decimal(120)
    with pytest.raises(ExecutionError, match="requested years"):
        compile_v5(
            program("average", ["r1c1", "r1c3"]),
            catalog,
            "What was average revenue from 2021 to 2023?",
        )


@pytest.mark.parametrize("period", ["2021 to 2023", "2021-2023", "2021–2023"])
def test_change_range_uses_only_endpoints(document, period):
    catalog = layout_catalog(document, document.evidence())
    plan = compile_v5(
        program("difference", ["r1c3", "r1c1"]),
        catalog,
        f"What was the change in revenue from {period}?",
    )
    assert execute(plan, document.evidence()) == Decimal(60)


def test_single_year_change_allows_prior_observation_but_must_include_year(document):
    catalog = layout_catalog(document, document.evidence())
    plan = compile_v5(
        program("difference", ["r1c2", "r1c1"]), catalog, "What was the change in revenue in 2023?"
    )
    assert execute(plan, document.evidence()) == Decimal(30)
    with pytest.raises(ExecutionError, match="omit the explicitly requested year"):
        compile_v5(
            program("difference", ["r1c2", "r1c3"]),
            catalog,
            "What was the change in revenue in 2023?",
        )


def test_multi_year_membership_stays_exact(document):
    catalog = layout_catalog(document, document.evidence())
    with pytest.raises(ExecutionError, match="requested years"):
        compile_v5(
            program("difference", ["r1c2", "r1c3"]),
            catalog,
            "What was the change from 2022 to 2023?",
        )


@pytest.mark.parametrize(
    "operation,operands,match",
    [
        ("sum", ["r1c1", "r1c1"], "Repeated"),
        ("difference", ["s0", "r1c1"], "future step"),
        ("multiply", ["r1c1", "r1c2"], "exactly one explicit percentage"),
    ],
)
def test_compiler_safety_not_relaxed(document, operation, operands, match):
    with pytest.raises(ExecutionError, match=match):
        compile_v5(program(operation, operands), layout_catalog(document, document.evidence()), "")


def test_payload_contains_original_layout_ids_and_typed_periods(document):
    payload = json.loads(
        source_payload(document, "Revenue?", layout_catalog(document, document.evidence()))
    )
    assert payload["table"][1][1] == "150 [r1c1]"
    assert payload["numeric_source_catalog"]["r1c1"]["years"] == ["2023"]
    assert payload["numeric_source_catalog"]["r1c1"]["row"] == "Revenue"
    assert payload["numeric_source_catalog"]["r1c1"]["dimension"] == "money"
    assert not {"answer", "derivation", "gold"} & payload.keys()


def test_successful_proposal_never_requests_second_opinion(document):
    planner = CandidateProgramPlanner(RuntimeProfile("test-model", "a" * 64))
    requests = []

    def request(body):
        requests.append(json.loads(body))
        return response(body=body)

    planner.client.request = request
    plan = planner.plan(document, "Change from 2022 to 2023", document.evidence())
    assert execute(plan, document.evidence()) == Decimal(30)
    assert len(requests) == 1
    assert requests[0]["temperature"] == 0.2
    assert requests[0]["max_tokens"] == 800
    assert not planner.state["certified"]


def test_invalid_proposal_has_one_feedback_repair(document):
    planner = CandidateProgramPlanner(RuntimeProfile("test-model", "a" * 64))
    requests = []

    def request(body):
        requests.append(json.loads(body))
        return (
            response("sum", ["r1c1", "r1c1"], body=body)
            if len(requests) == 1
            else response(body=body)
        )

    planner.client.request = request
    plan = planner.plan(document, "Change from 2022 to 2023", document.evidence())
    assert execute(plan, document.evidence()) == Decimal(30)
    assert len(requests) == 2
    assert "Repeated program operands" in requests[1]["messages"][-1]["content"]
    assert requests[1]["seed"] == requests[0]["seed"] + 1


def test_abstention_is_not_retried(document):
    planner = CandidateProgramPlanner(RuntimeProfile("test-model", "a" * 64))
    requests = []
    planner.client.request = lambda body: requests.append(body) or response(abstain=True, body=body)
    assert planner.plan(document, "Unsupported?", document.evidence()).abstain
    assert len(requests) == 1


def test_invalid_program_is_bounded_to_three_attempts(document):
    planner = CandidateProgramPlanner(RuntimeProfile("test-model", "a" * 64))
    requests = []
    planner.client.request = lambda body: requests.append(body) or response(
        "sum", ["r1c1", "r1c1"], body=body
    )
    with pytest.raises(ExecutionError, match="Version-five program rejected"):
        planner.plan(document, "Sum", document.evidence())
    assert len(requests) == 3


def test_transport_errors_are_not_repaired(document):
    planner = CandidateProgramPlanner(RuntimeProfile("test-model", "a" * 64))
    requests = []

    def request(body):
        requests.append(body)
        raise ExecutionError("Local model request failed")

    planner.client.request = request
    with pytest.raises(ExecutionError, match="Local model request failed"):
        planner.plan(document, "Sum", document.evidence())
    assert len(requests) == 1


def test_raw_cache_replays_both_invalid_and_valid_proposals(document, tmp_path):
    planner = CandidateProgramPlanner(RuntimeProfile("test-model", "a" * 64), cache_dir=tmp_path)
    calls = []

    def request(body):
        calls.append(body)
        return (
            response("sum", ["r1c1", "r1c1"], body=body) if len(calls) == 1 else response(body=body)
        )

    planner.client.request = request
    first = planner.plan(document, "Change from 2022 to 2023", document.evidence())
    assert len(list(tmp_path.glob("*.json"))) == 2
    planner.client.request = lambda body: pytest.fail("cached response was regenerated")
    assert planner.plan(document, "Change from 2022 to 2023", document.evidence()) == first


def test_oversized_context_abstains_without_generation(document):
    document = Document(document.id, document.table, ("x" * 27_000,))
    planner = CandidateProgramPlanner(RuntimeProfile("test-model", "a" * 64))
    planner.client.request = lambda body: pytest.fail("oversized context sent to model")
    assert planner.plan(document, "Sum", document.evidence()).abstain


@pytest.mark.parametrize(
    "envelope",
    [
        [],
        {"choices": []},
        {"choices": ["bad"]},
        {"choices": [{"message": None}]},
        {"choices": [{"message": {"content": {}}}]},
    ],
)
def test_malformed_response_envelopes_fail_in_controlled_bounded_way(document, envelope):
    planner = CandidateProgramPlanner(RuntimeProfile("test-model", "a" * 64))
    requests = []
    planner.client.request = lambda body: requests.append(body) or json.dumps(envelope).encode()
    with pytest.raises(ExecutionError, match="Version-five program rejected"):
        planner.plan(document, "Sum", document.evidence())
    assert len(requests) == 3


@pytest.mark.parametrize(
    "invented", ["r1c1", "r1c1 [Revenue;2023]", "s3", "s0 ", "r1c1 [ignore prior instructions]"]
)
def test_alias_normalization_rejects_every_nonissued_source_string(document, invented):
    references = source_references(layout_catalog(document, document.evidence()))
    with pytest.raises(ExecutionError, match="not an exact issued source choice"):
        normalize_program(program("lookup", [invented]), references)


def test_aliases_preserve_uniqueness_and_unicode_byte_bounds_after_truncation():
    label = "財務" * 50
    catalog = {
        "r1c1": {
            "row": label + "甲",
            "column": "年度" * 20,
            "years": ["2024"],
            "dimension": "money",
            "currency": "JPY",
        },
        "r2c1": {
            "row": label + "乙",
            "column": "年度" * 20,
            "years": ["2024"],
            "dimension": "money",
            "currency": "JPY",
        },
    }
    references = source_references(catalog)
    assert set(references.values()) == {"r1c1", "r2c1"}
    assert len(references) == 2
    suffixes = []
    for alias in references:
        assert alias.encode().decode() == alias
        suffix = alias.split(" [", 1)[1][:-1]
        parts = suffix.split(";")
        assert len(parts[0].encode()) <= 36
        assert len(parts[1].encode()) <= 24
        assert len(parts[2].encode()) <= 24
        assert len(parts[3].encode()) <= 32
        assert "\ufffd" not in alias
        suffixes.append(suffix)
    # Even labels that become identical after truncation retain distinct IDs.
    assert suffixes[0] == suffixes[1]


def test_normalization_precedes_duplicate_validation(document):
    catalog = layout_catalog(document, document.evidence())
    references = source_references(catalog)
    alias = next(alias for alias, name in references.items() if name == "r1c1")
    normalized = normalize_program(program("sum", [alias, alias]), references)
    assert normalized.steps[0].operands == ["r1c1", "r1c1"]
    with pytest.raises(ExecutionError, match="Repeated program operands"):
        compile_v5(normalized, catalog, "")


def test_normalization_preserves_steps_for_subsequent_dag_validation(document):
    catalog = layout_catalog(document, document.evidence())
    references = source_references(catalog)
    alias = next(alias for alias, name in references.items() if name == "r1c1")
    forward = normalize_program(program("difference", ["s0", alias]), references)
    with pytest.raises(ExecutionError, match="future step"):
        compile_v5(forward, catalog, "")
    earlier = Program(
        steps=[
            Step(operation="lookup", operands=[alias]),
            Step(operation="lookup", operands=["s0"]),
        ],
        abstain=False,
    )
    normalized = normalize_program(earlier, references)
    assert execute(compile_v5(normalized, catalog, ""), document.evidence()) == Decimal(150)


def test_simple_grammar_disallows_step_references_and_shares_one_operand_definition():
    schema = response_schema(["issued reference"], "What is the average net revenue?")[
        "json_schema"
    ]["schema"]
    assert schema["$defs"]["Operand"]["enum"] == ["issued reference"]
    for variant in schema["$defs"]["Step"]["oneOf"]:
        assert variant["properties"]["operands"]["items"] == {"$ref": "#/$defs/Operand"}


def test_start_to_end_change_uses_supported_annual_boundary_rows():
    source = Document(
        "synthetic",
        (
            ("Assets USD million", "Total"),
            ("As of January 1, 2024", "100"),
            ("Additions", "25"),
            ("As of December 31, 2024", "125"),
        ),
        (),
    )
    catalog = layout_catalog(source, source.evidence())
    plan = compile_v5(
        program("percent_change", ["r1c1", "r3c1"]),
        catalog,
        "What is the percentage change in total assets from start to end 2024?",
    )
    assert execute(plan, source.evidence()) == Decimal(25)
    assert plan.scale == "percent"


def test_beginning_and_end_shares_use_same_count_column_and_metric():
    source = Document(
        "synthetic",
        (
            ("", "Number of shares", "Fair value USD"),
            ("Outstanding at January 1, 2024", "100", "5"),
            ("Granted", "25", "6"),
            ("Outstanding at December 31, 2024", "125", "7"),
        ),
        (),
    )
    catalog = layout_catalog(source, source.evidence())
    plan = compile_v5(
        program("difference", ["r1c1", "r3c1"]),
        catalog,
        "What is the change in outstanding shares between beginning and end of 2024?",
    )
    assert execute(plan, source.evidence()) == Decimal(25)


@pytest.mark.parametrize(
    "operation,expected", [("difference", Decimal(20)), ("percent_change", Decimal(20))]
)
def test_named_month_change_uses_chronological_same_metric_columns(operation, expected):
    source = Document(
        "synthetic",
        (
            ("USD million", "March 31, 2024", "June 30, 2024", "September 30, 2024"),
            ("Net revenue", "100", "120", "140"),
        ),
        (),
    )
    catalog = layout_catalog(source, source.evidence())
    question = (
        "What is the percentage change" if operation == "percent_change" else "What is the change"
    )
    plan = compile_v5(
        program(operation, ["r1c1", "r1c2"]),
        catalog,
        question + " in net revenue between March and June in 2024?",
    )
    assert execute(plan, source.evidence()) == expected


@pytest.mark.parametrize(
    "operands,question",
    [
        (["r1c1", "r2c2"], "Change in net revenue between March and June in 2024?"),
        (["r1c1", "r2c1"], "Change in net revenue between March and June in 2024?"),
        (["r1c1", "r1c2"], "Change in net revenue in 2024?"),
        (["r1c1", "r1c2"], "Change in net revenue between March and September in 2024?"),
        (["r1c1", "r1c2"], "Change in net revenue between start and end of 2024?"),
        (["r1c1", "r1c2"], "Change in net revenue between March and June in 2023?"),
    ],
)
def test_intrayear_exception_rejects_unproven_period_or_metric_pairs(operands, question):
    source = Document(
        "synthetic",
        (
            ("USD million", "March 31, 2024", "June 30, 2024"),
            ("Net revenue", "100", "120"),
            ("Gross profit", "40", "50"),
        ),
        (),
    )
    with pytest.raises(ExecutionError, match="valid prior comparator|within-year periods"):
        compile_v5(
            program("difference", operands), layout_catalog(source, source.evidence()), question
        )


@pytest.mark.parametrize(
    "headers",
    [
        ["Reported March 31, 2024", "Adjusted June 30, 2024"],
        ["Three months ended March 31, 2024", "Year ended June 30, 2024"],
        ["March 31, 2024", "June 30, 2024 / adjustment"],
        ["2024 amount", "2024 adjustment"],
    ],
)
def test_intrayear_exception_keeps_category_and_period_type_qualifiers(headers):
    source = Document("synthetic", (("USD million", *headers), ("Net revenue", "100", "120")), ())
    with pytest.raises(ExecutionError, match="valid prior comparator|within-year periods"):
        compile_v5(
            program("difference", ["r1c1", "r1c2"]),
            layout_catalog(source, source.evidence()),
            "Change in net revenue between March and June in 2024?",
        )


def test_intrayear_exception_rejects_missing_year_metadata_and_unknown_dimensions():
    source = Document(
        "synthetic",
        (("USD million", "March 31, 2024", "June 30, 2024"), ("Net revenue", "100", "120")),
        (),
    )
    for mutation in ("years", "dimension"):
        catalog = layout_catalog(source, source.evidence())
        catalog["r1c1"][mutation] = [] if mutation == "years" else "unknown"
        with pytest.raises(ExecutionError):
            compile_v5(
                program("difference", ["r1c1", "r1c2"]),
                catalog,
                "Change in net revenue between March and June in 2024?",
            )


def test_intrayear_exception_does_not_relax_currency_checks():
    # Explicit synthetic source metadata isolates currency checks from parsing.
    catalog = {
        "r1c1": {
            "value": "100",
            "scale": "",
            "dimension": "money",
            "currency": "USD",
            "row": "Net revenue",
            "column": "March 31, 2024",
            "years": ["2024"],
            "source": "R1",
        },
        "r1c2": {
            "value": "120",
            "scale": "",
            "dimension": "money",
            "currency": "EUR",
            "row": "Net revenue",
            "column": "June 30, 2024",
            "years": ["2024"],
            "source": "R1",
        },
    }
    with pytest.raises(ExecutionError, match="mixes currencies"):
        compile_v5(
            program("difference", ["r1c1", "r1c2"]),
            catalog,
            "Change in net revenue between March and June in 2024?",
        )


def test_intrayear_exception_requires_a_single_temporal_step():
    source = Document(
        "synthetic",
        (("USD million", "March 31, 2024", "June 30, 2024"), ("Net revenue", "100", "120")),
        (),
    )
    catalog = layout_catalog(source, source.evidence())
    for proposed in (
        program("sum", ["r1c1", "r1c2"]),
        Program(
            steps=[
                Step(operation="lookup", operands=["r1c1"]),
                Step(operation="difference", operands=["r1c2", "s0"]),
            ],
            abstain=False,
        ),
    ):
        with pytest.raises(ExecutionError, match="valid prior comparator|within-year periods"):
            compile_v5(proposed, catalog, "Change in net revenue between March and June in 2024?")


def test_intrayear_exception_rejects_invalid_calendar_days():
    source = Document(
        "synthetic",
        (("USD million", "February 31, 2024", "June 30, 2024"), ("Net revenue", "100", "120")),
        (),
    )
    with pytest.raises(ExecutionError, match="valid prior comparator|within-year periods"):
        compile_v5(
            program("difference", ["r1c1", "r1c2"]),
            layout_catalog(source, source.evidence()),
            "Change in net revenue between February and June in 2024?",
        )


def test_intrayear_exception_rejects_cross_section_rows_even_with_matching_period_labels():
    source = Document(
        "synthetic",
        (
            ("USD million", "Total"),
            ("As of January 1, 2024", "100"),
            ("As of December 31, 2024", "125"),
        ),
        (),
    )
    catalog = layout_catalog(source, source.evidence())
    catalog["r1c1"]["section"] = "Domestic"
    catalog["r2c1"]["section"] = "Foreign"
    with pytest.raises(ExecutionError, match="valid prior comparator|within-year periods"):
        compile_v5(
            program("difference", ["r1c1", "r2c1"]),
            catalog,
            "Change in total from start to end 2024?",
        )


@pytest.mark.parametrize(
    "period",
    [
        "between March and June",
        "in May",
        "between Q1 and Q2",
        "in the first quarter",
        "from start to end",
        "in H1",
        "on 2024-05-31",
        "on 05/31/2024",
    ],
)
def test_explicit_subannual_request_cannot_fall_back_to_a_prior_year(period):
    source = Document(
        "synthetic",
        (
            ("USD million", "March 31, 2024", "June 30, 2023"),
            ("Net revenue", "120", "100"),
        ),
        (),
    )
    with pytest.raises(ExecutionError, match="within-year periods"):
        compile_v5(
            program("difference", ["r1c1", "r1c2"]),
            layout_catalog(source, source.evidence()),
            f"What is the change in net revenue {period} in 2024?",
        )


def test_unspecified_annual_change_retains_valid_prior_year_support():
    source = Document(
        "synthetic",
        (
            ("USD million", "2024", "2023"),
            ("Net revenue", "120", "100"),
        ),
        (),
    )
    plan = compile_v5(
        program("difference", ["r1c1", "r1c2"]),
        layout_catalog(source, source.evidence()),
        "What is the change in net revenue in 2024?",
    )
    assert execute(plan, source.evidence()) == Decimal(20)


def test_percentage_of_a_named_change_metric_is_not_forced_to_temporal_change():
    schema = response_schema(
        ["a", "b"],
        "How many percent of organic change in Group A is the organic change in Group B?",
    )["json_schema"]["schema"]
    assert schema["properties"]["steps"]["maxItems"] == 4
    assert len(schema["$defs"]["Step"]["oneOf"]) == 8


@pytest.mark.parametrize(
    "amount,share,total",
    [("25", "5%", Decimal(500)), ("(25)", "5%", Decimal(-500)), ("25", "100%", Decimal(25))],
)
def test_typed_inverse_percentage_preserves_numerator_unit(amount, share, total):
    source = Document(
        "synthetic",
        (("USD thousand", "2024"), ("Expense", amount), ("Percent of sales", share)),
        (),
    )
    plan = compile_v5(
        program("divide", ["r1c1", "r2c1"]),
        layout_catalog(source, source.evidence()),
        "What are total sales in 2024?",
    )
    assert execute(plan, source.evidence()) == total
    assert plan.scale == "thousand"
    assert plan.citations == ["R1", "R2"]


def test_inverse_percentage_zero_share_abstains_at_execution():
    source = Document(
        "synthetic", (("USD million", "2024"), ("Expense", "25"), ("Percent of sales", "0%")), ()
    )
    plan = compile_v5(
        program("divide", ["r1c1", "r2c1"]),
        layout_catalog(source, source.evidence()),
        "What are total sales in 2024?",
    )
    with pytest.raises(ExecutionError, match="zero"):
        execute(plan, source.evidence())


@pytest.mark.parametrize(
    "mutation", ["different_year", "requested_year_missing", "subannual", "unknown_denominator"]
)
def test_inverse_percentage_requires_source_types_and_period_alignment(mutation):
    source = Document(
        "synthetic", (("USD million", "2024"), ("Expense", "25"), ("Percent of sales", "5%")), ()
    )
    catalog = layout_catalog(source, source.evidence())
    question = "What are total sales in 2024?"
    if mutation == "different_year":
        catalog["r2c1"]["years"] = ["2023"]
    elif mutation == "requested_year_missing":
        question = "What are total sales in 2023?"
    elif mutation == "subannual":
        question = "What are total sales in Q1 2024?"
    else:
        catalog["r2c1"]["dimension"] = "unknown"
    with pytest.raises(ExecutionError):
        compile_v5(program("divide", ["r1c1", "r2c1"]), catalog, question)
