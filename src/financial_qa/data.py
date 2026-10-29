"""A label-free document view separated from TAT-QA evaluation annotations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .storage import download

REVISION = "870accc41953dcde885aabeb963d94aabdc0fbc3"
ROOT_URL = f"https://raw.githubusercontent.com/NExTplusplus/TAT-QA/{REVISION}/dataset_raw"


@dataclass(frozen=True)
class Evidence:
    id: str
    text: str
    variables: dict[str, Decimal]


@dataclass(frozen=True)
class Document:
    id: str
    table: tuple[tuple[str, ...], ...]
    paragraphs: tuple[str, ...]

    def evidence(self) -> list[Evidence]:
        items = []
        for row_index, row in enumerate(self.table):
            values = {}
            for column, cell in enumerate(row):
                value = parse_number(cell)
                if value is not None:
                    values[f"r{row_index}c{column}"] = value
            items.append(Evidence(f"R{row_index}", " | ".join(row), values))
        for index, paragraph in enumerate(self.paragraphs):
            values = {}
            for position, match in enumerate(
                re.finditer(r"(?<!\w)\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?", paragraph)
            ):
                value = parse_number(match.group())
                if value is not None:
                    values[f"p{index}n{position}"] = value
            items.append(Evidence(f"P{index}", paragraph, values))
        return items


@dataclass(frozen=True)
class Example:
    document: Document
    id: str
    question: str
    answer: str | float | list[str]
    scale: str
    answer_type: str


def parse_number(text: str) -> Decimal | None:
    normalized = text.strip().replace(",", "").replace("%", "")
    normalized = re.sub(r"[$€£¥]", "", normalized)
    normalized = re.sub(r"\(([^()]*)\)\s*%?$", r"(\1)", normalized)
    normalized = normalized.replace("−", "-").strip()
    if normalized.startswith("(") and normalized.endswith(")"):
        normalized = "-" + normalized[1:-1].strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", normalized):
        return None
    try:
        value = Decimal(normalized)
        return value if value.is_finite() and abs(value) < Decimal("1e18") else None
    except InvalidOperation:
        return None


def fetch_dataset(directory: Path, split: str = "dev") -> Path:
    if split not in {"train", "dev", "test"}:
        raise ValueError("Split must be train, dev, or test")
    path = directory / f"tatqa_{split}.json"
    download(f"{ROOT_URL}/tatqa_dataset_{split}.json", path)
    return path


def load_examples(path: Path) -> list[Example]:
    records = json.loads(path.read_text())
    examples = []
    for record in records:
        document = Document(
            id=record["table"]["uid"],
            table=tuple(tuple(str(cell) for cell in row) for row in record["table"]["table"]),
            paragraphs=tuple(paragraph["text"] for paragraph in record["paragraphs"]),
        )
        for question in record["questions"]:
            if "answer" not in question:
                continue
            examples.append(
                Example(
                    document,
                    question["uid"],
                    question["question"],
                    question["answer"],
                    question.get("scale", ""),
                    question.get("answer_type", "unknown"),
                )
            )
    return examples
