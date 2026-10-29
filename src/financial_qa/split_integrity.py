"""Conservative source groups for future QA experiments and retrospective audits.

Context IDs are identifiers, not evidence of independent filings. Group sources
that repeat a complete table or a sufficiently long paragraph, including indirect
chains. This deliberately over-excludes some boilerplate; it cannot recognize
every paraphrased filing or establish independence from LLM pretraining data.
Historical manifests must remain unchanged when an audit discovers overlap.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from itertools import combinations

from .data import Document, Example, parse_number

GROUPING_VERSION = "exact-table-or-informative-paragraph-components-v1"
MIN_PARAGRAPH_CHARACTERS = 80
MIN_PARAGRAPH_WORDS = 12


def normalize_source(value: str) -> str:
    """Normalize typography/case/spacing, retaining financial numbers and signs."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def source_keys(document: Document) -> set[str]:
    """Build source-only keys; questions, gold values and annotation IDs are absent."""
    table = [[normalize_source(cell) for cell in row] for row in document.table]
    keys = {"table:" + _digest(table)} if any(any(row) for row in table) else set()
    for paragraph in document.paragraphs:
        text = normalize_source(paragraph)
        if (
            len(text) >= MIN_PARAGRAPH_CHARACTERS
            and len(re.findall(r"\w+", text)) >= MIN_PARAGRAPH_WORDS
        ):
            keys.add("paragraph:" + _digest(text))
    return keys


def document_index(documents: list[Document]) -> dict[str, Document]:
    """Reject ambiguous IDs rather than letting a dict silently replace a source."""
    result = {}
    for document in documents:
        if not isinstance(document.id, str) or not document.id.strip() or document.id in result:
            raise ValueError("Source contexts must have unique nonempty IDs")
        result[document.id] = document
    if not result:
        raise ValueError("Need source contexts for an integrity audit")
    return result


def content_groups(documents: list[Document]) -> dict[str, str]:
    """Find transitive source components with deterministic minimum-ID labels."""
    indexed = document_index(documents)
    parent = {identifier: identifier for identifier in indexed}

    def find(identifier: str) -> str:
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    seen = {}
    for identifier in sorted(indexed):
        for key in sorted(source_keys(indexed[identifier])):
            if key in seen:
                left, right = find(identifier), find(seen[key])
                parent[max(left, right)] = min(left, right)
            else:
                seen[key] = identifier
    return {identifier: find(identifier) for identifier in sorted(indexed)}


def overlap_audit(documents: list[Document], stages: dict[str, list[str]]) -> dict:
    """Describe direct and indirect overlap without reading any outcome labels."""
    indexed = document_index(documents)
    groups = content_groups(documents)
    indexes = {}
    for stage, identifiers in stages.items():
        if len(set(identifiers)) != len(identifiers) or not set(identifiers) <= indexed.keys():
            raise ValueError(f"Unknown or duplicated contexts in stage {stage}")
        keys: dict[str, set[str]] = defaultdict(set)
        for identifier in identifiers:
            for key in source_keys(indexed[identifier]):
                keys[key].add(identifier)
        indexes[stage] = keys
    pairs = []
    for left, right in combinations(stages, 2):
        shared = sorted(indexes[left].keys() & indexes[right].keys())
        direct = []
        for key in shared:
            direct.append(
                {
                    "kind": key.split(":", 1)[0],
                    "source_sha256": key.split(":", 1)[1],
                    "left_contexts": sorted(indexes[left][key]),
                    "right_contexts": sorted(indexes[right][key]),
                }
            )
        common_groups = {groups[i] for i in stages[left]} & {groups[i] for i in stages[right]}
        pairs.append(
            {
                "left": left,
                "right": right,
                "shared_context_ids": sorted(set(stages[left]) & set(stages[right])),
                "shared_table_keys": sum(item["kind"] == "table" for item in direct),
                "shared_paragraph_keys": sum(item["kind"] == "paragraph" for item in direct),
                "shared_source_groups": len(common_groups),
                "left_contexts_in_shared_groups": sorted(
                    i for i in stages[left] if groups[i] in common_groups
                ),
                "right_contexts_in_shared_groups": sorted(
                    i for i in stages[right] if groups[i] in common_groups
                ),
                "direct_shared_sources": direct,
            }
        )
    return {
        "grouping_version": GROUPING_VERSION,
        "minimum_paragraph_characters": MIN_PARAGRAPH_CHARACTERS,
        "minimum_paragraph_words": MIN_PARAGRAPH_WORDS,
        "source_contexts": len(indexed),
        "source_groups": len(set(groups.values())),
        "stage_counts": {stage: len(ids) for stage, ids in stages.items()},
        "source_group_disjoint": all(not row["shared_source_groups"] for row in pairs),
        "pairs": pairs,
    }


def assert_group_disjoint(groups: dict[str, str], stages: dict[str, list[str]]) -> None:
    """Enforce stage isolation before a future fitting/calibration/evaluation run."""
    seen = {}
    for stage, identifiers in stages.items():
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"Duplicated source context in stage {stage}")
        local = set()
        for identifier in identifiers:
            if identifier not in groups:
                raise ValueError(f"Unknown source context in stage {stage}")
            group = groups[identifier]
            if group in seen:
                raise ValueError(f"Source group crosses stages {seen[group]} and {stage}")
            local.add(group)
        seen.update({group: stage for group in local})


def guarded_sample(
    examples: list[Example], documents: list[Document], consumed_ids: set[str], size: int, seed: int
) -> dict:
    """Choose at most one question per unused source group, without gold ranking.

    Numeric answer availability/type defines the task, as in earlier manifests;
    actual answer values, scales, predictions and error outcomes never determine
    sample order. A consumed source blocks its whole connected component.
    """
    if type(size) is not int or size < 1 or type(seed) is not int:
        raise ValueError("Need a positive integer size and an integer seed")
    indexed = document_index(documents)
    if not consumed_ids <= indexed.keys():
        raise ValueError("Consumed sources include unknown contexts")
    groups = content_groups(documents)
    blocked = {groups[identifier] for identifier in consumed_ids}
    candidates: dict[str, list[Example]] = defaultdict(list)
    seen_questions = set()
    for example in examples:
        if example.id in seen_questions or indexed.get(example.document.id) != example.document:
            raise ValueError("Question identity is duplicated or does not match its source")
        seen_questions.add(example.id)
        if (
            groups[example.document.id] not in blocked
            and example.answer_type == "arithmetic"
            and parse_number(str(example.answer)) is not None
        ):
            candidates[groups[example.document.id]].append(example)
    if len(candidates) < size:
        raise ValueError("Insufficient unused source groups; do not recycle inspected evaluation")
    rng = random.Random(seed)
    available = sorted(candidates)
    rng.shuffle(available)
    selected = [
        rng.choice(sorted(candidates[group], key=lambda e: (e.document.id, e.id)))
        for group in available[:size]
    ]
    result = {
        "grouping_version": GROUPING_VERSION,
        "size": size,
        "seed": seed,
        "consumed_context_ids": sorted(consumed_ids),
        "available_unconsumed_source_groups": len(candidates),
        "examples": [
            {"id": e.id, "document_id": e.document.id, "source_group": groups[e.document.id]}
            for e in selected
        ],
    }
    assert_group_disjoint(
        groups,
        {"consumed": sorted(consumed_ids), "future": [e.document.id for e in selected]},
    )
    return result


def validate_guarded_sample(
    manifest: dict,
    examples: list[Example],
    documents: list[Document],
    consumed_ids: set[str],
    size: int,
    seed: int,
) -> None:
    """Reconstruct selection; edited exclusions/membership cannot pass by rehashing."""
    if (
        not isinstance(manifest, dict)
        or any(
            type(manifest.get(field)) is not int
            for field in ("size", "seed", "available_unconsumed_source_groups")
        )
        or manifest != guarded_sample(examples, documents, consumed_ids, size, seed)
    ):
        raise ValueError("Manifest differs from the complete source-group sampling contract")
