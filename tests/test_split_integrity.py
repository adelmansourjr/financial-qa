"""Adversarial split mutations and annotation poisoning must not bypass isolation."""

import copy
import json
from dataclasses import replace

import pytest

from financial_qa.data import Document, Example, load_examples
from financial_qa.reasoning_v3 import RuntimeProfile
from financial_qa.reasoning_v4 import LayoutProgramPlanner
from financial_qa.split_integrity import (
    assert_group_disjoint,
    content_groups,
    guarded_sample,
    overlap_audit,
    source_keys,
    validate_guarded_sample,
)

PARAGRAPH = (
    "This deliberately long source paragraph describes the same company disclosure "
    "and payment arrangements across different table contexts during the stated financial year."
)


def document(identifier, value, paragraphs=()):
    return Document(
        identifier, (("USD million", "2023", "2022"), ("Revenue", value, "100")), paragraphs
    )


def example(source, identifier=None):
    return Example(
        source, identifier or f"q-{source.id}", "Change in revenue?", 20, "million", "arithmetic"
    )


def test_identical_tables_join_despite_different_ids_and_paragraphs():
    first = document("a", "120", ("One paragraph.",))
    second = document("b", "120", ("Different paragraph.",))
    groups = content_groups([first, second])
    assert groups["a"] == groups["b"]
    audit = overlap_audit([first, second], {"development": ["a"], "calibration": ["b"]})
    assert not audit["source_group_disjoint"]
    assert audit["pairs"][0]["shared_table_keys"] == 1
    with pytest.raises(ValueError, match="crosses stages"):
        assert_group_disjoint(groups, {"development": ["a"], "calibration": ["b"]})


def test_shared_paragraphs_join_transitively_and_preserve_numeric_differences():
    first = document("a", "120")
    bridge = document("b", "120", (PARAGRAPH,))
    last = document("c", "130", (PARAGRAPH,))
    assert set(content_groups([last, bridge, first]).values()) == {"a"}
    changed = replace(last, paragraphs=(PARAGRAPH + " Amount 123.",))
    assert content_groups([first, changed]) == {"a": "a", "c": "c"}
    assert content_groups([last, bridge, first]) == content_groups([first, bridge, last])


def test_unstaged_source_bridge_still_blocks_future_sampling():
    consumed = document("a", "120")
    bridge = document("b", "120", (PARAGRAPH,))
    candidate = document("c", "130", (PARAGRAPH,))
    # The intermediate source need not be selected in any manifest for its
    # shared-content evidence to connect development and a proposed future item.
    with pytest.raises(ValueError, match="Insufficient"):
        guarded_sample([example(candidate)], [consumed, bridge, candidate], {"a"}, 1, 42)


def test_unicode_case_spacing_normalization_and_short_boilerplate():
    first = document("a", "120", (PARAGRAPH,))
    second = replace(first, id="b", paragraphs=("  " + PARAGRAPH.upper().replace(" ", "\n"),))
    assert source_keys(first) == source_keys(second)
    short_first = document("a", "120", ("Annual report",))
    short_second = document("b", "130", ("Annual report",))
    assert len(set(content_groups([short_first, short_second]).values())) == 2


@pytest.mark.parametrize("bad", ["unknown", "duplicate_id", "duplicate_stage"])
def test_ambiguous_membership_is_rejected(bad):
    first = document("a", "120")
    if bad == "duplicate_id":
        with pytest.raises(ValueError, match="unique"):
            content_groups([first, replace(first, table=(("Different",),))])
    else:
        with pytest.raises(ValueError, match="Unknown or duplicated"):
            overlap_audit([first], {"stage": ["absent"] if bad == "unknown" else ["a", "a"]})


def test_sampling_excludes_whole_consumed_component_and_one_per_remaining_group():
    docs = [
        document("old", "120"),
        document("alias", "120"),
        document("c", "130"),
        document("d", "130"),
        document("e", "140"),
    ]
    examples = [example(source) for source in docs]
    manifest = guarded_sample(examples, docs, {"old"}, 2, 42)
    chosen = {row["document_id"] for row in manifest["examples"]}
    assert not {"old", "alias"} & chosen
    assert len(chosen & {"c", "d"}) == 1
    assert "e" in chosen
    with pytest.raises(ValueError, match="Insufficient"):
        guarded_sample(examples, docs, {"old"}, 3, 42)


@pytest.mark.parametrize("mutation", ["excluded", "question", "group", "size", "seed"])
def test_full_contract_reconstruction_rejects_edited_manifests(mutation):
    docs = [document("a", "120"), document("b", "130"), document("c", "140")]
    examples = [example(source) for source in docs]
    manifest = guarded_sample(examples, docs, {"a"}, 1, 42)
    if mutation == "excluded":
        manifest["consumed_context_ids"] = []
    elif mutation in {"question", "group"}:
        manifest["examples"][0]["id" if mutation == "question" else "source_group"] = "forged"
    else:
        manifest[mutation] += 1
    with pytest.raises(ValueError, match="complete"):
        validate_guarded_sample(manifest, examples, docs, {"a"}, 1, 42)


def test_sampling_does_not_rank_by_gold_values_or_annotation_scale():
    docs = [document("a", "120"), document("b", "130"), document("c", "140")]
    examples = [example(source) for source in docs]
    poisoned = [replace(e, answer=-987654, scale="billion") for e in examples]
    assert guarded_sample(examples, docs, set(), 2, 42) == guarded_sample(
        poisoned, docs, set(), 2, 42
    )


def test_boolean_manifest_counts_do_not_pass_as_integer_one():
    docs = [document("a", "120")]
    examples = [example(docs[0])]
    manifest = guarded_sample(examples, docs, set(), 1, 1)
    manifest["seed"] = True
    with pytest.raises(ValueError, match="complete"):
        validate_guarded_sample(manifest, examples, docs, set(), 1, 1)


@pytest.mark.parametrize("size,seed", [(True, 42), (1, True), (0, 42)])
def test_sampling_rejects_invalid_protocol_parameters(size, seed):
    docs = [document("a", "120")]
    with pytest.raises(ValueError, match="integer"):
        guarded_sample([example(docs[0])], docs, set(), size, seed)


def test_gold_derivation_and_question_metadata_poisoning_do_not_change_llm_input(tmp_path):
    record = {
        "table": {
            "uid": "context",
            "table": [["USD million", "2023", "2022"], ["Revenue", "120", "100"]],
        },
        "paragraphs": [{"text": "Synthetic public source."}],
        "questions": [
            {
                "uid": "q",
                "question": "Change in revenue from 2022 to 2023?",
                "answer": 20,
                "scale": "million",
                "answer_type": "arithmetic",
                "derivation": "120-100",
                "mapping": {"table": [[1, 1]]},
            }
        ],
    }
    poisoned = copy.deepcopy(record)
    poisoned["questions"][0].update(
        answer="SECRET_GOLD",
        scale="SECRET_SCALE",
        answer_type="SECRET_TYPE",
        derivation="SECRET_DERIVATION",
        mapping={"secret": "SECRET_MAPPING"},
    )
    sent = []
    for index, source in enumerate((record, poisoned)):
        path = tmp_path / f"input-{index}.json"
        path.write_text(json.dumps([source]))
        item = load_examples(path)[0]
        planner = LayoutProgramPlanner(RuntimeProfile("synthetic", "a" * 64))

        def request(body):
            sent.append(json.loads(body))
            return json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(
                                    {
                                        "analysis": "Two explicit periods.",
                                        "steps": [
                                            {
                                                "operation": "difference",
                                                "operands": ["r1c1", "r1c2"],
                                            }
                                        ],
                                        "abstain": False,
                                    }
                                )
                            },
                        }
                    ]
                }
            ).encode()

        planner.client.request = request
        assert not planner.plan(item.document, item.question, item.document.evidence()).abstain
    assert len(sent) == 2 and sent[0] == sent[1]
    assert "SECRET_" not in json.dumps(sent)
