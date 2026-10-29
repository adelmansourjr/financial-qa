"""Exact label/period resolution before probabilistic program construction.

This rule deliberately requires a unique complete row label and unambiguous
source observations. It never uses benchmark IDs, answers, fuzzy thresholds or
learned error exceptions. Ambiguity delegates to the model rather than guessing.
"""

from __future__ import annotations

import re

from .data import Document, Evidence
from .executor import ExecutionError, Plan
from .grounding_v5 import layout_catalog
from .periods_v5 import has_subannual_period
from .reasoning_v3 import Program, Step, compile_program

IGNORED = {"the", "of", "and", "in", "at", "for", "a", "an"}
# These change the financial measure rather than merely the question's wording.
# A shorter row label cannot establish an adjusted/net/category-specific value.
MEASURE_QUALIFIERS = {
    "adjusted",
    "unadjusted",
    "net",
    "gross",
    "gaap",
    "continuing",
    "discontinued",
    "controlling",
    "noncontrolling",
    "attributable",
    "diluted",
    "basic",
    "domestic",
    "foreign",
    "international",
    "current",
    "noncurrent",
    "consolidated",
    "additional",
}


def words(value: str) -> set[str]:
    # Treat common inflections of an addition consistently. This distinguishes
    # additions during a period from a closing balance of the same asset.
    value = re.sub(r"\b(?:additions?|added)\b", "additional", value, flags=re.I)
    return {
        word[:-1] if word.endswith("s") and not word.endswith("ss") and len(word) > 3 else word
        for word in re.findall(r"[a-z]+|\d+(?:\.\d+)?", value.lower())
        if word not in IGNORED
    }


def _label_words(value: str, footnotes: set[str]) -> set[str]:
    """Ignore only trailing source-declared footnotes, retaining category digits."""
    while match := re.search(r"\s*\(([a-z]|[ivxlcdm]+|\d{1,2})\)\s*$", value, re.I):
        if match.group(1).lower() not in footnotes:
            break
        value = value[: match.start()]
    return words(value)


def exact_plan(document: Document, question: str, evidence: list[Evidence]) -> Plan | None:
    """Resolve a single literal metric; return None for any unsupported ambiguity."""
    lower = question.lower()
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", question))
    footnotes = {
        match.group(1).lower()
        for paragraph in document.paragraphs
        if (match := re.match(r"\s*\(([a-z]|[ivxlcdm]+|\d{1,2})\)(?:\s|$)", paragraph, re.I))
    }
    weighted_metric_terms = None
    if re.search(r"weighted[ -]average", lower):
        # Mask the lexical metric qualifier without moving character offsets.
        # Only a separate outer arithmetic average over an exactly named row
        # can be resolved; constructing a weighted mean still needs weights.
        masked = re.sub(r"weighted[ -]average", lambda match: "_" * len(match.group()), lower)
        outer = re.search(
            r"\baverage\s+(?:(?:number|amount|value)\s+of\s+)?(?:of\s+)?(?:the\s+)?"
            r"(.+?)\s+(?:from|between|for|in)\s+(?:19|20)\d{2}\b",
            masked,
        )
        if not outer or len(years) < 2:
            return None
        metric = lower[outer.start(1) : outer.end(1)]
        if not re.search(r"weighted[ -]average", metric):
            return None
        weighted_metric_terms = words(metric)
        if not any(
            row and _label_words(row[0], footnotes) == weighted_metric_terms
            for row in document.table
        ):
            return None
    # Compound statistics still require a program, not a single-row rule.
    temporal = bool(re.search(r"\b(?:change|increase|decrease|growth|decline)\b", lower))
    if temporal and len(years) == 1 and has_subannual_period(question):
        # The annual shortcut cannot establish month/quarter or opening/closing
        # comparability. The model/compiler must resolve explicit source periods.
        return None
    if re.search(r"annualized|\bcagr\b|compound annual", lower) or (
        temporal and re.search(r"\baverage\b|\bsum of\b", lower)
    ):
        return None
    if re.search(r"\b(?:divided|multiplied|times|versus|compared|excluding)\b", lower):
        return None
    # The symbolic rule always uses newer minus older. An explicitly reversed
    # comparison or a requested magnitude of decrease needs model interpretation.
    if temporal and re.search(r"\b(?:decrease|decline|reduction)\b", lower):
        return None
    for first, last in re.findall(r"\bfrom\s+((?:19|20)\d{2})\s+to\s+((?:19|20)\d{2})\b", lower):
        if int(first) >= int(last):
            return None
    if len(years) == 1 and not re.search(r"\baverage\b|including both|\bchange\b", lower):
        relation = re.search(r"(?:ratio of|percentage of)\s+(.+?)\s+to\s+(.+)", lower)
        if relation is None:
            relation = re.search(
                r"(?:what is|what was)\s+(?:the\s+)?(.+?)\s+expressed as a ratio of\s+(.+)", lower
            )
        if relation:
            catalog = layout_catalog(document, evidence)
            operands = []
            for phrase in relation.groups():
                matching = {}
                for name, item in catalog.items():
                    terms = _label_words(item.get("row", ""), footnotes)
                    context_terms = words(
                        " ".join(item.get(key, "") for key in ("row", "column", "section"))
                    )
                    if (
                        name.startswith("r")
                        and terms
                        and terms <= words(phrase)
                        and not ((words(phrase) & MEASURE_QUALIFIERS) - context_terms)
                        and item.get("years") == list(years)
                    ):
                        matching[name] = len(terms)
                if not matching:
                    return None
                best = max(matching.values())
                chosen = [name for name in matching if matching[name] == best]
                if len(chosen) != 1:
                    return None
                operands.extend(chosen)
            try:
                operation = "percent_ratio" if "percentage" in lower else "divide"
                return compile_program(
                    Program(steps=[Step(operation=operation, operands=operands)], abstain=False),
                    catalog,
                    "",
                )
            except ExecutionError:
                return None
    if not years or re.search(
        r"\bratio\b|\bproportion\b|percent(?:age)? of|how many percent", lower
    ):
        return None
    if temporal:
        operation = "percent_change" if re.search(r"percent|growth", lower) else "difference"
        if len(years) == 1:
            years.add(str(int(next(iter(years))) - 1))
    elif re.search(r"\baverage\b", lower) and (
        not re.search(r"weighted[ -]average", lower) or weighted_metric_terms is not None
    ):
        operation = "average"
    elif "sum of" in lower:
        operation = "sum"
    elif len(years) == 1 and not re.search(r"\bdifference\b|\btotal amount\b", lower):
        if re.search(r"percent|share of|fraction|portion", lower):
            return None
        operation = "lookup"
    else:
        return None
    if operation in {"sum", "average"}:
        for first, last in re.findall(
            r"((?:19|20)\d{2})\s*(?:to|through|[-–])\s*((?:19|20)\d{2})", lower
        ):
            if 0 < int(last) - int(first) <= 10:
                years.update(str(y) for y in range(int(first), int(last) + 1))
            else:
                return None
    catalog = layout_catalog(document, evidence)
    query = words(question)
    candidates = {}
    for name, item in catalog.items():
        if not name.startswith("r"):
            continue
        label = item.get("row", "")
        if ("%" in label or "percent" in label.lower()) and not re.search(r"percent|rate", lower):
            continue
        terms = _label_words(label, footnotes)
        if weighted_metric_terms is not None and terms != weighted_metric_terms:
            continue
        if not terms or terms <= {"total", "other", "amount", "net", "ratio"} or not terms <= query:
            continue
        context_terms = words(" ".join((label, item.get("column", ""), item.get("section", ""))))
        if (query & MEASURE_QUALIFIERS) - context_terms:
            continue
        candidates.setdefault(int(re.match(r"r(\d+)", name).group(1)), (terms, []))[1].append(name)
    if not candidates:
        return None
    highest = max(len(value[0]) for value in candidates.values())
    rows = [value for value in candidates.values() if len(value[0]) == highest]
    if len(rows) != 1:
        return None
    names = rows[0][1]
    chosen = []
    for year in sorted(years, reverse=True):
        observed = [name for name in names if catalog[name].get("years") == [year]]
        # A change/share column cannot replace its underlying monetary amount.
        amounts = [
            name
            for name in observed
            if catalog[name]["dimension"] != "percent"
            and not re.search(r"change|increase|decrease", catalog[name].get("column", ""), re.I)
        ]
        observed = amounts or observed
        if len(observed) != 1:
            return None
        chosen.extend(observed)
    if (operation == "lookup" and len(chosen) != 1) or (
        operation in {"difference", "percent_change"} and len(chosen) != 2
    ):
        return None
    try:
        return compile_program(
            Program(steps=[Step(operation=operation, operands=chosen)], abstain=False), catalog, ""
        )
    except ExecutionError:
        return None
