"""Synthetic declarations isolate legitimate metadata repairs from benchmark labels."""

from financial_qa.data import Document
from financial_qa.grounding_v5 import layout_catalog


def catalog(rows, paragraphs=()):
    document = Document("synthetic", tuple(map(tuple, rows)), tuple(paragraphs))
    evidence = document.evidence()
    result = layout_catalog(document, evidence)
    original = {name: str(value) for row in evidence for name, value in row.variables.items()}
    assert all(original[name] == item["value"] for name, item in result.items())
    return result


def test_currency_shorthand_row_can_follow_descriptive_header():
    result = catalog([["", "2023", "2022"], ["Credit rating", "£m", "£m"], ["AA", "120", "100"]])
    assert result["r2c1"]["scale"] == "million"
    assert result["r2c1"]["currency"] == "GBP"


def test_explicit_common_size_statement_applies_to_unmarked_cells():
    result = catalog(
        [["", "2023", "2022"], ["Sales", "100.0%", "100.0%"], ["Costs", "12", "14"]],
        ["The following table reports, as a percentage of total revenue, our results."],
    )
    assert result["r2c1"]["dimension"] == "percent"
    assert result["r2c2"]["scale"] == "percent"


def test_count_declaration_does_not_scale_per_share_amounts():
    result = catalog(
        [["", "Number of shares", "Price per share"], ["Outstanding", "120", "$10"]],
        ["The following table lists amounts in thousands except per share amounts."],
    )
    assert result["r1c1"]["scale"] == "thousand"
    assert result["r1c2"]["scale"] == ""


def test_comparative_narrative_count_has_local_years_and_dimension():
    result = catalog(
        [["", "2023", "2022"], ["Revenue", "120", "100"]],
        ["In the year, 250 units opened (2022: 260 stores)."],
    )
    assert result["p0n0"]["years"] == ["2023"]
    assert result["p0n2"]["years"] == ["2022"]
    assert result["p0n2"]["dimension"] == "count"
    assert "p0n1" not in result or result["p0n1"]["role"] == "date"


def test_ordinary_tables_do_not_invent_percent_or_count_scale():
    result = catalog([["", "2023", "2022"], ["Measure", "120", "100"]])
    assert result["r1c1"]["scale"] == ""
    assert result["r1c1"]["dimension"] == "unknown"
