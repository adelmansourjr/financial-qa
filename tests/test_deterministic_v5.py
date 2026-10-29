"""Exact rules must prove source selection and decline ambiguous calculations."""

from decimal import Decimal

import pytest

from financial_qa.data import Document
from financial_qa.deterministic_v5 import exact_plan
from financial_qa.executor import execute


def doc(rows):
    return Document("synthetic", tuple(map(tuple, rows)), ())


def test_exact_metric_change_uses_correct_row_and_period_without_model():
    source = doc(
        [
            ["USD million", "2023", "2022"],
            ["Aggregate purchase price", "120", "100"],
            ["Total cash used", "122", "102"],
        ]
    )
    plan = exact_plan(
        source, "What is the change in aggregate purchase price from2022 to2023?", source.evidence()
    )
    # Spaced source years are required; arbitrary numeric substrings are not dates.
    assert plan is None
    plan = exact_plan(
        source,
        "What is the change in aggregate purchase price from 2022 to 2023?",
        source.evidence(),
    )
    assert execute(plan, source.evidence()) == Decimal(20)
    assert plan.citations == ["R1"]


def test_named_percentage_share_does_not_become_a_lookup():
    source = doc([["", "2023"], ["Debt", "20"], ["Total assets", "100"]])
    assert (
        exact_plan(source, "What percentage of total assets is debt in 2023?", source.evidence())
        is None
    )


def test_percentage_metric_is_not_an_amount_with_the_same_words():
    source = doc([["", "2023"], ["Expenses", "20"], ["% of net sales", "5%"]])
    assert exact_plan(source, "What is the amount of net sales in 2023?", source.evidence()) is None


def test_duplicate_metric_requires_disambiguation():
    source = doc([["", "2023", "2022"], ["Tax credits", "20", "10"], ["Tax credits", "40", "30"]])
    assert exact_plan(source, "Change in tax credits from 2022 to 2023?", source.evidence()) is None


def test_prior_year_cannot_be_replaced_by_a_future_year():
    source = doc([["USD million", "2024", "2023"], ["Net revenue", "150", "120"]])
    assert exact_plan(source, "Change in net revenue in 2023?", source.evidence()) is None


def test_explicit_average_range_includes_all_periods():
    source = doc([["USD million", "2023", "2022", "2021"], ["Net revenue", "150", "120", "90"]])
    plan = exact_plan(source, "What is average net revenue from 2021 to 2023?", source.evidence())
    assert execute(plan, source.evidence()) == Decimal(120)


def test_explicit_sum_range_includes_intermediate_periods():
    source = doc([["USD million", "2023", "2022", "2021"], ["Net revenue", "150", "120", "90"]])
    plan = exact_plan(
        source, "What is the sum of net revenue from 2021 to 2023?", source.evidence()
    )
    assert execute(plan, source.evidence()) == Decimal(360)


@pytest.mark.parametrize(
    "question",
    [
        "What is the weighted average net revenue in 2023?",
        "What is average percentage change in net revenue from 2021 to 2023?",
        "What is annualized growth in net revenue from 2021 to 2023?",
        "What is the decline in net revenue from 2021 to 2023?",
        "What is the change in net revenue from 2023 to 2021?",
        "What is net revenue compared with net income in 2023?",
        "How many times net income is net revenue in 2023?",
    ],
)
def test_unsupported_compound_or_directional_questions_delegate(question):
    source = doc(
        [
            ["USD million", "2023", "2022", "2021"],
            ["Net revenue", "150", "120", "90"],
            ["Net income", "30", "25", "20"],
        ]
    )
    assert exact_plan(source, question, source.evidence()) is None


def test_numeric_metric_categories_are_preserved():
    source = doc([["USD million", "2023"], ["Level 1 assets", "120"], ["Level 2 assets", "90"]])
    plan = exact_plan(source, "What are level 2 assets in 2023?", source.evidence())
    assert execute(plan, source.evidence()) == Decimal(90)


@pytest.mark.parametrize("modifier", ["adjusted", "gross", "domestic", "noncontrolling"])
def test_shorter_metric_does_not_establish_an_unobserved_category(modifier):
    source = doc([["USD million", "2023"], ["Net revenue", "150"]])
    assert exact_plan(source, f"What is {modifier} net revenue in 2023?", source.evidence()) is None


def test_explicit_ratio_uses_distinct_compatible_observations():
    source = doc([["USD million", "2023"], ["Net debt", "20"], ["Total assets", "100"]])
    plan = exact_plan(
        source, "What is the ratio of net debt to total assets in 2023?", source.evidence()
    )
    assert execute(plan, source.evidence()) == Decimal("0.2")
    assert (
        exact_plan(source, "What is the ratio of net debt to net debt in 2023?", source.evidence())
        is None
    )


def test_ratio_does_not_establish_an_unobserved_adjusted_numerator():
    source = doc([["USD million", "2023"], ["Net debt", "20"], ["Total assets", "100"]])
    assert (
        exact_plan(
            source,
            "What is the ratio of adjusted net debt to total assets in 2023?",
            source.evidence(),
        )
        is None
    )


def test_ratio_rejects_mixed_currencies():
    source = doc([["", "2023"], ["Net debt USD", "20"], ["Total assets EUR", "100"]])
    assert (
        exact_plan(
            source,
            "What is the ratio of net debt USD to total assets EUR in 2023?",
            source.evidence(),
        )
        is None
    )


def test_declared_numeric_label_footnotes_do_not_obscure_a_ratio():
    source = Document(
        "synthetic",
        (
            ("USD million", "2024"),
            ("Borrowings less current portion(2)", "80"),
            ("Total debt(3)", "100"),
        ),
        (
            "(2) Borrowings exclude the current portion.",
            "(3) Total debt includes the current portion.",
        ),
    )
    plan = exact_plan(
        source,
        "What is the percentage of borrowings less current portion to total debt in 2024?",
        source.evidence(),
    )
    assert execute(plan, source.evidence()) == Decimal(80)
    assert plan.scale == "percent"


def test_undeclared_parenthesized_metric_category_is_not_a_footnote():
    source = doc([["USD million", "2024"], ["Level (2) assets", "80"]])
    assert exact_plan(source, "What are level assets in 2024?", source.evidence()) is None
    plan = exact_plan(source, "What are level 2 assets in 2024?", source.evidence())
    assert execute(plan, source.evidence()) == Decimal(80)


def test_declared_footnotes_preserve_unparenthesized_category_digits():
    source = Document(
        "synthetic",
        (("USD million", "2024"), ("Level 1 assets(2)", "80"), ("Level 2 assets(2)", "100")),
        ("(2) Measured at fair value.",),
    )
    plan = exact_plan(source, "What are level 2 assets in 2024?", source.evidence())
    assert execute(plan, source.evidence()) == Decimal(100)


def test_outer_average_of_a_literal_weighted_average_metric_uses_unique_years():
    source = doc(
        [
            ["Shares thousand", "2024", "2023", "2022"],
            ["Weighted-average shares outstanding basic", "42", "54", "66"],
        ]
    )
    plan = exact_plan(
        source,
        "What is the average number of basic weighted-average shares outstanding from 2022-2024?",
        source.evidence(),
    )
    assert execute(plan, source.evidence()) == Decimal(54)


def test_a_weighted_mean_request_is_not_an_outer_arithmetic_mean():
    source = doc([["USD million", "2024", "2023"], ["Net revenue", "42", "54"]])
    assert (
        exact_plan(
            source, "What is the weighted average net revenue from 2023 to 2024?", source.evidence()
        )
        is None
    )


def test_outer_mean_requires_the_complete_literal_weighted_metric():
    source = doc(
        [
            ["Shares thousand", "2024", "2023"],
            ["Weighted-average shares outstanding basic", "42", "54"],
        ]
    )
    assert (
        exact_plan(
            source,
            "What is the average number of diluted weighted-average shares outstanding "
            "from 2023 to 2024?",
            source.evidence(),
        )
        is None
    )


def test_period_additions_do_not_match_an_asset_closing_balance():
    source = doc(
        [
            ["USD thousand", "2024", "2023"],
            ["Additions equipment", "40", "60"],
            ["Equipment", "400", "500"],
        ]
    )
    plan = exact_plan(
        source,
        "What is the average value of additional equipment added in 2023 and 2024?",
        source.evidence(),
    )
    assert execute(plan, source.evidence()) == Decimal(50)
    assert plan.citations == ["R1"]


@pytest.mark.parametrize(
    "period", ["in May", "between March and June", "between Q1 and Q2", "from beginning to end"]
)
def test_annual_shortcut_delegates_explicit_subannual_requests(period):
    source = doc([["USD million", "2024", "2023"], ["Net revenue", "120", "100"]])
    assert exact_plan(source, f"Change in net revenue {period} in 2024?", source.evidence()) is None
