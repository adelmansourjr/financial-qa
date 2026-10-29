"""Layout-aware, label-free source metadata for the version-four planner.

Numeric values and identifiers come from the existing Evidence objects. This module
only describes them: it never repairs a number, supplies a missing value, or reads
an evaluation annotation. Ambiguous units and periods remain unresolved.
"""

from __future__ import annotations

import re

from .context import infer_scale
from .data import Document, Evidence, parse_number

YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
NUMBER = re.compile(r"(?<!\w)\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?")
DATE_LABEL = re.compile(
    r"\bas (?:at|of)\b|\bat december\b|\byears? (?:ended|ending)\b|"
    r"\bfiscal years?\b|\bbalance\b|\bbeginning\b|\bopening\b|\bending\b|"
    r"\bjanuary\b|\bfebruary\b|\bmarch\b|\bapril\b|\bmay\b|\bjune\b|"
    r"\bjuly\b|\baugust\b|\bseptember\b|\boctober\b|\bnovember\b|\bdecember\b",
    re.I,
)
COUNT = re.compile(r"\bnumber\b|\bshares\b|\bemployees\b|\bheadcount\b|\bunits\b", re.I)


def _scale(text: str) -> str:
    # Currency-code notation is common in source statements but absent from the
    # older generic parser. A declaration applies only within its table scope.
    if re.search(r"\b(?:USD|EUR|GBP|AUD|CAD|JPY|CHF)\s*['’‘]000\b", text, re.I):
        return "thousand"
    return infer_scale(text)


def _currency(text: str) -> str:
    found = set(re.findall(r"\b(?:USD|EUR|GBP|AUD|CAD|JPY|CHF)\b", text.upper()))
    for symbol, code in (("€", "EUR"), ("£", "GBP"), ("¥", "JPY")):
        if symbol in text:
            found.add(code)
    if "$" in text and not found & {"USD", "AUD", "CAD"}:
        found.add("DOLLAR")
    return next(iter(found)) if len(found) == 1 else "conflict" if found else ""


def _headers(table: tuple[tuple[str, ...], ...]) -> list[int]:
    headers = []
    for index, row in enumerate(table[:10]):
        if not row:
            continue
        label = row[0].strip()
        numbers = [parse_number(cell) for cell in row[1:] if parse_number(cell) is not None]
        year_row = bool(numbers) and all(1900 <= value <= 2099 for value in numbers)
        header_label = not label or bool(
            DATE_LABEL.search(label)
            or _scale(label)
            or re.search(r"\b(?:year|years|data|figures)\b", label, re.I)
        )
        if numbers and not (year_row and header_label):
            break
        # A textual body row (e.g. Grant date / 1st October) must not become
        # another header just because its cells are not plain decimal numbers.
        if index and label and not header_label:
            break
        headers.append(index)
    return headers


def _column_context(table: tuple[tuple[str, ...], ...], headers: list[int]) -> list[dict]:
    width = max((len(row) for row in table), default=0)
    columns = [{"labels": [], "years": [], "direct": []} for _ in range(width)]
    for index in headers:
        row = table[index]
        anchors = [c for c in range(1, len(row)) if YEAR.fullmatch(row[c].strip())]
        # Some merged year labels are right aligned over repeated two-column
        # groups (Shares / £m). Recognize only a complete, regular partition.
        stride = anchors[0] if anchors else 0
        right_aligned = (
            stride > 1
            and len(anchors) >= 2
            and anchors == list(range(stride, width, stride))
            and anchors[-1] == width - 1
        )
        previous = ""
        for column in range(1, width):
            cell = row[column].strip() if column < len(row) else ""
            if cell:
                previous = cell
                columns[column]["direct"].append(cell)
            text = previous
            if right_aligned:
                anchor = ((column - 1) // stride + 1) * stride
                text = row[anchor]
            if text:
                columns[column]["labels"].append(text)
            # A comparison heading (2019 vs 2018) describes a rate column,
            # rather than two separately observed accounting periods.
            years = YEAR.findall(text)
            if len(years) == 1:
                columns[column]["years"].extend(years)
    for column in columns:
        column["years"] = list(dict.fromkeys(column["years"]))
    return columns


def _dimension(scope: str, scale: str, currency: str) -> str:
    if scale == "percent":
        return "percent"
    if re.search(r"per\s+(?:share|unit)", scope, re.I):
        return "per_share"
    if re.search(r"\bage\b|\bin years\b|\b(?:remaining|contractual) life\b", scope, re.I):
        return "duration"
    if COUNT.search(scope):
        return "count"
    return "money" if currency or scale in {"thousand", "million", "billion"} else "unknown"


def layout_catalog(document: Document, evidence: list[Evidence]) -> dict:
    """Describe available numeric sources with local units and observation periods.

    ``years`` describes observation periods, not arbitrary years in entity names.
    ``role=date`` is assigned only to paragraph tokens with clear date context;
    a bare financial amount equal to 2000 therefore remains eligible evidence.
    """
    available = {name: str(value) for item in evidence for name, value in item.variables.items()}
    headers = _headers(document.table)
    columns = _column_context(document.table, headers)
    declaration = " ".join(" ".join(document.table[index]) for index in headers)
    table_scale, table_currency = _scale(declaration), _currency(declaration)
    # Only explicit columnar/table declarations may provide paragraph fallback.
    declarations = [
        text
        for text in document.paragraphs
        if re.search(
            r"columnar|amounts (?:shown|expressed)|all figures|dollars in|"
            r"\bin (?:thousands?|millions?|billions?)\b",
            text,
            re.I,
        )
    ]
    if not table_scale:
        table_scale = _scale(" ".join(declarations))
    if not table_currency:
        # Accounting statements commonly print the symbol only on the first and
        # total cells. A single consistent body currency supplies table context;
        # distinct currencies remain unresolved rather than being converted.
        body_currency = {
            _currency(cell)
            for row in document.table
            for cell in row[1:]
            if parse_number(cell) is not None and _currency(cell)
        }
        if len(body_currency) == 1:
            table_currency = next(iter(body_currency))
    catalog = {}
    section = ""
    for index, row in enumerate(document.table):
        if index in headers or not row:
            continue
        numeric = any(parse_number(cell) is not None for cell in row[1:])
        if not numeric and row[0].strip() and all(not cell.strip() for cell in row[1:]):
            section = row[0].strip()
        attributes = " | ".join(cell for cell in row if cell and parse_number(cell) is None)
        for column, cell in enumerate(row[1:], 1):
            name = f"r{index}c{column}"
            if name not in available:
                continue
            context = columns[column]
            labels = context["labels"]
            # Do not extend an upper percent-change group through a later year
            # amount column. Explicit percent cells/direct headings remain valid.
            column_text = " / ".join(
                label
                for label in labels
                if not (
                    re.search(r"percent|%", label, re.I)
                    and label not in context["direct"]
                    and context["years"]
                )
            )
            scope = " ".join((row[0], column_text, section))
            direct_percent = any(
                re.search(r"^%|percent change|percentage$", x, re.I) for x in context["direct"]
            )
            row_percent = bool(re.search(r"%|\bpercent(?:age)?\b", row[0], re.I))
            scale = _scale(scope) or table_scale
            if "%" in cell or row_percent or direct_percent:
                scale = "percent"
            currency = _currency(cell + " " + scope) or table_currency
            dimension = _dimension(scope, scale, currency)
            if _currency(cell) and dimension == "count":
                # Explicit cell symbols identify award value even when a column
                # is named restricted-share units. Table-wide currency does not
                # override a separately labeled share-count column.
                dimension = "money"
            if dimension in {"count", "duration", "per_share"}:
                scale = (
                    _scale(row[0] + " " + column_text)
                    or (table_scale if not table_currency else "")
                    if dimension == "count"
                    else ""
                )
            if dimension in {"percent", "count", "duration"}:
                currency = ""
            years = list(context["years"])
            if DATE_LABEL.search(row[0]):
                years = list(dict.fromkeys(YEAR.findall(row[0]) + years))
            catalog[name] = {
                "value": available[name],
                "row": row[0],
                "column": column_text,
                "section": section,
                "row_attributes": attributes,
                "scale": scale,
                "currency": currency,
                "dimension": dimension,
                "source": f"R{index}",
                "years": years,
                "period": " / ".join(years),
                "role": "amount",
            }
    for item in evidence:
        if not item.id.startswith("P"):
            continue
        matches = list(NUMBER.finditer(item.text))
        for name in item.variables:
            position = int(name.rsplit("n", 1)[1])
            if position >= len(matches):
                continue
            match = matches[position]
            # Attach a multiplier/symbol to its own number, never every number
            # in a paragraph containing another monetary amount or percentage.
            suffix = item.text[match.end() : match.end() + 24]
            token = match.group()
            local = item.text[max(0, match.start() - 40) : match.end() + 65]
            adjacent = token + re.split(r"[,;.]|\band\b|\bof\b", suffix, maxsplit=1)[0]
            scale = (
                "percent"
                if "%" in token or re.match(r"\s*(?:per cent|percent)\b", suffix)
                else _scale(adjacent)
            )
            prefix = item.text[max(0, match.start() - 32) : match.start()]
            currency = _currency(token + (prefix[-1] if prefix and prefix[-1:] in "€£¥" else ""))
            is_date = bool(
                YEAR.fullmatch(available[name])
                and (
                    re.search(r"\b(?:in|fiscal|during|year|years)\s*$", prefix, re.I)
                    or (
                        re.search(r"\band\s*$", prefix, re.I)
                        and re.search(r"\b(?:fiscal|years?)\b", prefix, re.I)
                        and YEAR.search(prefix)
                    )
                    or re.match(r"\s*:", suffix)
                    or re.search(
                        r"\b(?:January|February|March|April|May|June|July|August|"
                        r"September|October|November|December)\s+\d{1,2},?\s*$",
                        prefix,
                        re.I,
                    )
                )
            )
            dimension = _dimension(adjacent, scale, currency)
            if dimension in {"percent", "count", "duration"}:
                currency = ""
            # A sole year in the same sentence can safely scope several facts.
            # Multiple years (including 'respectively' constructions) need a
            # semantic alignment decision and remain unresolved here.
            sentence_start = 0
            sentence_end = len(item.text)
            for boundary in re.finditer(r"[.!?]\s+(?=[A-Z])", item.text):
                if boundary.start() < match.start():
                    sentence_start = boundary.end()
                elif boundary.start() >= match.end():
                    sentence_end = boundary.start()
                    break
            sentence_years = list(
                dict.fromkeys(YEAR.findall(item.text[sentence_start:sentence_end]))
            )
            years = sentence_years if len(sentence_years) == 1 and not is_date else []
            catalog[name] = {
                "value": available[name],
                "text": item.text,
                "local_text": local,
                "scale": scale,
                "currency": currency,
                "dimension": dimension,
                "source": item.id,
                "years": years,
                "period": " / ".join(years),
                "role": "date" if is_date else "amount",
            }
    return catalog
