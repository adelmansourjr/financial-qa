"""Regression cases for source-layout mistakes, without benchmark annotations."""

from financial_qa.data import Document
from financial_qa.grounding_v4 import layout_catalog


def catalog(rows, paragraphs=()):
    document = Document("synthetic", tuple(tuple(row) for row in rows), tuple(paragraphs))
    return layout_catalog(document, document.evidence())


def test_labeled_year_header_retains_period_without_treating_entity_year_as_period():
    sources = catalog(
        [
            ("", "", "Year Ended December 31,"),
            ("Stock-based compensation expense data:", "2019", "2018"),
            ("2022 Notes", "2000", "1900"),
        ]
    )
    assert sources["r2c1"]["years"] == ["2019"]
    assert sources["r2c2"]["years"] == ["2018"]
    assert sources["r2c1"]["value"] == "2000"
    assert "r1c1" not in sources


def test_percent_neighbors_and_textual_body_rows_do_not_contaminate_counts():
    sources = catalog(
        [
            ("", "2018 Grant", "2019 Grant"),
            ("Grant date", "1st October", "1st October"),
            ("Exercise price", "7240.0p", "7835.0p"),
            ("Number of employees", "1294", "1318"),
            ("Expected volatility", "19%", "21%"),
            ("Risk free interest rate", "0.4%", "0.5%"),
        ]
    )
    assert sources["r3c2"]["dimension"] == "count"
    assert sources["r3c2"]["scale"] == ""
    assert sources["r3c2"]["years"] == ["2019"]
    assert sources["r4c2"]["scale"] == "percent"
    assert "Exercise price" not in sources["r3c2"]["column"]


def test_merged_percent_group_stops_before_amount_year_column():
    sources = catalog(
        [
            ("", "", "", "Year Ended May 31,", ""),
            ("", "", "", "Percent Change", ""),
            ("Dollars in millions", "2019", "Actual", "Constant", "2018"),
            ("Employee costs", "$49", "3%", "4%", "$48"),
        ]
    )
    assert sources["r3c1"]["scale"] == sources["r3c4"]["scale"] == "million"
    assert sources["r3c4"]["dimension"] == "money"
    assert sources["r3c3"]["dimension"] == "percent"


def test_source_attributes_durations_and_unit_declarations_are_preserved():
    sources = catalog(
        [
            ("Name", "Age", "Position"),
            ("Alice", "65", "Senior Vice President"),
            ("Bob", "69", "Chief Executive Officer"),
        ]
    )
    assert "Senior Vice President" in sources["r1c1"]["row_attributes"]
    assert sources["r1c1"]["dimension"] == "duration"
    monetary = catalog(
        [
            ("", "2019", "2018"),
            ("All figures in USD ‘000", "", ""),
            ("Additions Vessels", "2531", "169446"),
        ]
    )
    assert monetary["r2c1"]["scale"] == "thousand"
    assert monetary["r2c1"]["currency"] == "USD"


def test_right_aligned_year_groups_and_explicit_balance_dates():
    sources = catalog(
        [
            ("", "", "2019", "", "2018"),
            ("", "Shares million", "£m", "Shares million", "£m"),
            ("Disposals", "-1.1", "-3.5", "-1.0", "-3.0"),
        ]
    )
    assert sources["r2c1"]["years"] == sources["r2c2"]["years"] == ["2019"]
    assert sources["r2c3"]["years"] == sources["r2c4"]["years"] == ["2018"]
    assert sources["r2c1"]["dimension"] == "count"
    assert sources["r2c2"]["dimension"] == "money"
    balances = catalog(
        [
            ("", "Total"),
            ("At January 1, 2019", "214371"),
            ("At December 31, 2019", "206495"),
        ]
    )
    assert balances["r1c1"]["years"] == ["2019"]
    assert balances["r1c1"]["row"] == "At January 1, 2019"


def test_paragraph_number_units_and_date_roles_are_local():
    sources = catalog(
        [],
        [
            "In 2019, 256 units opened versus 262 units in 2018; rent was £125 million.",
            "Income was $2.2 million and $16.9 million during fiscal 2018 and 2017, respectively.",
            "The cash balance was 2000.",
            "Two balances were 3000 and 2000. In 2019, dividends were $55.0 million.",
        ],
    )
    assert sources["p0n0"]["role"] == "date"
    assert sources["p0n1"]["dimension"] == "count"
    assert sources["p0n1"]["currency"] == ""
    assert sources["p0n4"]["scale"] == "million"
    assert sources["p0n4"]["currency"] == "GBP"
    assert sources["p1n1"]["scale"] == "million"
    assert sources["p1n3"]["role"] == "date"
    assert sources["p2n0"]["role"] == "amount"
    assert sources["p3n1"]["role"] == "amount"
    assert sources["p3n3"]["years"] == ["2019"]


def test_per_share_and_duration_do_not_inherit_table_money_multiplier():
    sources = catalog(
        [
            ("Dollars in millions", "2019", "2018"),
            ("Earnings per share", "$2.5", "$2.1"),
            ("Average term of debt in years", "3.9", "4.7"),
        ]
    )
    assert sources["r1c1"]["dimension"] == "per_share"
    assert sources["r1c1"]["scale"] == ""
    assert sources["r2c1"]["dimension"] == "duration"
    assert sources["r2c1"]["currency"] == ""


def test_values_and_identifiers_remain_executor_evidence_and_subset_is_respected():
    document = Document("source", (("", "2019"), ("Revenue", "$(12.5)")), ("Tax $3.",))
    evidence = document.evidence()
    sources = layout_catalog(document, evidence)
    original = {name: str(value) for row in evidence for name, value in row.variables.items()}
    assert all(original[name] == item["value"] for name, item in sources.items())
    assert sources["r1c1"]["value"] == "-12.5"
    assert not any(name.startswith("r") for name in layout_catalog(document, evidence[-1:]))


def test_number_at_paragraph_start_has_no_currency_prefix():
    document = Document(
        "synthetic", (("Year", "2023"), ("Revenue", "1")), ("3000 employees worked in 2023.",)
    )
    catalog = layout_catalog(document, document.evidence())
    assert catalog["p0n0"]["value"] == "3000"
    assert catalog["p0n0"]["currency"] == ""


def test_table_declaration_and_sparse_currency_symbols_are_shared():
    sources = catalog(
        [("", "2023", "2022"), ("Revenue", "$120", "$100"), ("Other revenue", "12", "10")],
        ("Revenue by location (in millions):",),
    )
    assert sources["r2c1"]["scale"] == "million"
    assert sources["r2c1"]["dimension"] == "money"
    assert sources["r2c1"]["currency"] == sources["r1c1"]["currency"] == "DOLLAR"
    awards = catalog([("Officer", "Restricted Share Units", "Total"), ("Person A", "$20", "$40")])
    assert awards["r1c1"]["dimension"] == "money"


def test_explicit_table_count_declaration_retains_multiplier():
    sources = catalog(
        [("", "2023", "2022"), ("Shares outstanding", "20", "10")],
        ("The following table reports shares (in thousands):",),
    )
    assert sources["r1c1"]["dimension"] == "count"
    assert sources["r1c1"]["scale"] == "thousand"
    mixed = catalog([("USD million", "2023"), ("Revenue", "20"), ("Number of employees", "1000")])
    assert mixed["r2c1"]["dimension"] == "count"
    assert mixed["r2c1"]["scale"] == ""
