"""Recover multirow headers and source units without looking at annotations."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .data import Document, parse_number


@dataclass(frozen=True)
class TableContext:
    header_rows: tuple[int, ...]
    columns: tuple[str, ...]
    scale: str


def infer_scale(text: str) -> str:
    lower = text.lower()
    patterns = {
        "billion": r"\bbillions?\b|\bbn\b",
        "million": r"\bmillions?\b|[€£$]\s*m\b|\b(?:usd|eur|gbp)\s*m\b",
        "thousand": r"\bthousands?\b|[€£$]\s*['’]?000\b",
    }
    found = [scale for scale, pattern in patterns.items() if re.search(pattern, lower)]
    return found[0] if len(found) == 1 else ""


def table_context(document: Document) -> TableContext:
    header_rows = []
    for index, row in enumerate(document.table):
        if index >= 10:
            break
        # A numeric labeled row starts the table body; unlabeled years remain headers.
        if (
            index
            and row
            and row[0].strip()
            and any(parse_number(cell) is not None for cell in row[1:])
        ):
            break
        header_rows.append(index)
    width = max((len(row) for row in document.table), default=0)
    columns = [[] for _ in range(width)]
    for index in header_rows:
        previous = ""
        for column, cell in enumerate(document.table[index]):
            if column == 0:
                continue
            if cell.strip():
                previous = cell.strip()
            if previous and previous not in columns[column]:
                columns[column].append(previous)
    unit_text = " ".join(" ".join(document.table[index]) for index in header_rows)
    for paragraph in document.paragraphs:
        if re.search(r"\b(?:in|of) (?:thousands?|millions?|billions?)\b", paragraph.lower()):
            unit_text += " " + paragraph
    return TableContext(
        tuple(header_rows), tuple(" / ".join(parts) for parts in columns), infer_scale(unit_text)
    )


def variable_catalog(document: Document, allowed_ids: set[str]) -> dict[str, dict[str, str]]:
    context = table_context(document)
    catalog = {}
    section = ""
    for index, row in enumerate(document.table):
        if index in context.header_rows:
            continue
        if row and row[0].strip() and not any(parse_number(cell) is not None for cell in row[1:]):
            section = row[0].strip()
        if f"R{index}" not in allowed_ids:
            continue
        for column, cell in enumerate(row[1:], 1):
            value = parse_number(cell)
            if value is None:
                continue
            header = context.columns[column] if column < len(context.columns) else ""
            unit = "percent" if "%" in cell or "%" in header else context.scale
            catalog[f"r{index}c{column}"] = {
                "value": str(value),
                "row": row[0] if row else "",
                "column": header,
                "section": section,
                "scale": unit,
                "source": f"R{index}",
            }
    for item in document.evidence():
        if item.id.startswith("P") and item.id in allowed_ids:
            for key, value in item.variables.items():
                catalog[key] = {
                    "value": str(value),
                    "source": item.id,
                    "text": item.text,
                    "scale": "",
                }
    return catalog
