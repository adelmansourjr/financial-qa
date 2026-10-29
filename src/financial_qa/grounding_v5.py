"""Source-only refinements for units and narrative counts in v5.

The frozen v4 catalog and original Decimal observations remain untouched. Each
correction requires an explicit declaration or local lexical evidence; unknown
units and ambiguous periods remain unresolved. Evaluation labels are never read.
"""

from __future__ import annotations

import re

from .data import Document, Evidence
from .grounding_v4 import NUMBER, YEAR
from .grounding_v4 import layout_catalog as legacy_catalog

SHORT_UNIT = re.compile(r"^\s*([£€$])\s*(m|bn|000)\s*$", re.I)
COUNT_WORD = re.compile(
    r"^\s*(?:million\s+|thousand\s+)?(?:units?|stores?|employees?|shares?)\b", re.I
)


def layout_catalog(document: Document, evidence: list[Evidence]) -> dict:
    """Preserve every numeric ID/value while improving explicit source attributes."""
    catalog = legacy_catalog(document, evidence)
    # Unit-only header rows can follow a descriptive first column and therefore
    # sit outside v4's more conservative header recognizer.
    units = {}
    for index, row in enumerate(document.table[:5]):
        matches = [(column, SHORT_UNIT.fullmatch(cell)) for column, cell in enumerate(row)]
        for column, match in matches:
            if match and column > 0:
                units[column] = (index, match.group(1), match.group(2).lower())
    common_size = any(
        re.search(r"\b(?:following|this) table\b.*\bas a percentage of\b", text, re.I)
        for text in document.paragraphs
    )
    count_declaration = any(
        re.search(
            r"\b(?:following|this) table\b.*amounts in thousands except per.share", text, re.I
        )
        for text in document.paragraphs
    )
    header_years = {
        year for row in document.table[:3] for cell in row[1:] for year in YEAR.findall(cell)
    }
    current_year = max(header_years) if header_years else None
    for name, item in catalog.items():
        if name.startswith("r"):
            row, column = map(int, re.fullmatch(r"r(\d+)c(\d+)", name).groups())
            if common_size and item["dimension"] == "unknown":
                item.update(scale="percent", dimension="percent", currency="")
            if count_declaration and item["dimension"] == "count" and not item["scale"]:
                item["scale"] = "thousand"
            if column in units and row > units[column][0] and item["dimension"] == "unknown":
                _, symbol, multiplier = units[column]
                item.update(
                    scale={"m": "million", "bn": "billion", "000": "thousand"}[multiplier],
                    dimension="money",
                    currency={"£": "GBP", "€": "EUR", "$": "DOLLAR"}[symbol],
                )
            continue
        paragraph, position = map(int, re.fullmatch(r"p(\d+)n(\d+)", name).groups())
        text = document.paragraphs[paragraph]
        matches = list(NUMBER.finditer(text))
        match = matches[position]
        before, after = text[: match.start()], text[match.end() :]
        if COUNT_WORD.match(after):
            item.update(dimension="count", currency="")
            # A parenthetical comparison '2018: 262 stores' gives a direct year
            # for262; it must not relabel the preceding current-year256 units.
            direct = re.search(r"\b((?:19|20)\d{2})\s*:\s*$", before)
            if direct:
                item.update(years=[direct.group(1)], period=direct.group(1))
            elif current_year and re.search(r"\bin the year\s*,?\s*$", before, re.I):
                item.update(years=[current_year], period=current_year)
        if YEAR.fullmatch(match.group().strip()) and (
            re.match(r"\s*:", after) or re.search(r"(?:,|and)\s*$", before)
        ):
            item.update(role="date", years=[], period="")
    return catalog
