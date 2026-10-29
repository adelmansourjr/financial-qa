"""Reservations must survive concurrency, publication failure and metadata edits."""

import importlib.util
import json
from pathlib import Path

import pytest

from financial_qa import source_reservations as reservations
from financial_qa.calibration import digest
from financial_qa.data import Document, Example
from financial_qa.split_integrity import content_groups
from financial_qa.storage import write_json


@pytest.fixture
def pool(tmp_path):
    documents = [Document(str(i), (("Revenue", str(100 + i)),), ()) for i in range(7)]
    examples = [Example(d, f"q-{d.id}", "Revenue?", 20, "", "arithmetic") for d in documents]
    ledger = tmp_path / "benchmarks/source-reservations.json"
    write_json(ledger, {"schema": reservations.SCHEMA, "entries": []})
    return ledger, {"pinned-source": "synthetic"}, examples, documents, {"0"}


def freeze(pool, name, size=2):
    ledger, identity, examples, documents, historical = pool
    return reservations.freeze_reservation(
        ledger, ledger.parent / "future" / name, identity, examples, documents, historical, size, 42
    )


def selected(entry):
    return {row["document_id"] for row in entry["sampling"]["examples"]}


def test_two_separate_freezes_reserve_disjoint_groups_and_old_manifest_still_validates(pool):
    first = freeze(pool, "calibration.json")
    second = freeze(pool, "test.json")
    assert not selected(first) & selected(second)
    ledger, identity, examples, documents, historical = pool
    groups = content_groups(documents)
    assert not {groups[i] for i in selected(first)} & {groups[i] for i in selected(second)}
    assert (
        reservations.validate_reservation(
            ledger,
            ledger.parent / "future/calibration.json",
            identity,
            examples,
            documents,
            historical,
            2,
            42,
        )
        == first
    )


def test_publication_crash_consumes_sources_before_next_reservation(pool, monkeypatch):
    ledger = pool[0]
    destination = ledger.parent / "future/crashed.json"
    real_write = reservations.write_json

    def fail_publication(path, value):
        if path == destination:
            raise OSError("Synthetic publication failure")
        return real_write(path, value)

    monkeypatch.setattr(reservations, "write_json", fail_publication)
    with pytest.raises(OSError, match="Synthetic"):
        freeze(pool, "crashed.json")
    assert not destination.exists()
    first = json.loads(ledger.read_text())["entries"][0]
    second = freeze(pool, "after-crash.json")
    assert not selected(first) & selected(second)
    assert not ledger.with_suffix(".json.lock").exists()


def test_writer_lock_prevents_second_reservation_without_mutation(pool):
    ledger = pool[0]
    before = ledger.read_bytes()
    with reservations.exclusive_writer(ledger):
        with pytest.raises(ValueError, match="writer lock"):
            freeze(pool, "overlap.json")
        assert ledger.read_bytes() == before
    assert freeze(pool, "after-unlock.json")["index"] == 1


@pytest.mark.parametrize("mutation", ["overlap", "chain", "identity", "rollback", "published_bool"])
def test_metadata_mutation_cannot_hide_source_reuse_or_tampering(pool, mutation):
    first = freeze(pool, "calibration.json")
    freeze(pool, "test.json")
    ledger, identity, examples, documents, historical = pool
    value = json.loads(ledger.read_text())
    if mutation == "overlap":
        value["entries"][1]["sampling"] = first["sampling"]
        value["entries"][1]["entry_sha256"] = digest(
            {k: v for k, v in value["entries"][1].items() if k != "entry_sha256"}
        )
    elif mutation == "chain":
        value["entries"][1]["previous_sha256"] = "0" * 64
    elif mutation == "identity":
        identity = {"pinned-source": "changed"}
    elif mutation == "rollback":
        value["entries"] = value["entries"][:1]
    else:
        first["index"] = True
        write_json(ledger.parent / "future/calibration.json", first)
    write_json(ledger, value)
    with pytest.raises(ValueError):
        reservations.read_reservations(ledger, identity, examples, documents, historical)


def test_missing_ledger_is_not_silently_reset_and_destination_cannot_escape(pool):
    ledger, identity, examples, documents, historical = pool
    with pytest.raises(ValueError, match="benchmarks/future"):
        reservations.freeze_reservation(
            ledger, ledger.parent / "outside.json", identity, examples, documents, historical, 1, 42
        )
    ledger.unlink()
    with pytest.raises(ValueError, match="missing"):
        freeze(pool, "missing.json")


def test_capacity_failure_preserves_existing_ledger(pool, monkeypatch):
    ledger = pool[0]
    before = ledger.read_bytes()
    monkeypatch.setattr(reservations, "MAX_LEDGER_BYTES", len(before) + 10)
    with pytest.raises(ValueError, match="capacity"):
        freeze(pool, "too-large.json")
    assert ledger.read_bytes() == before


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_leakage.py"
    spec = importlib.util.spec_from_file_location("qa_leakage_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_cannot_overwrite_protected_evidence(tmp_path, monkeypatch, runner):
    evidence = tmp_path / "historical.json"
    evidence.write_text("historical bytes")
    monkeypatch.setattr("sys.argv", ["audit_leakage.py", "audit", "--output", str(evidence)])
    monkeypatch.setattr(runner, "audit", lambda: pytest.fail("Output preflight must run first"))
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert evidence.read_text() == "historical bytes"


def test_new_reserved_prediction_report_does_not_block_next_cli_freeze(pool, monkeypatch, runner):
    ledger, identity, examples, documents, historical = pool
    monkeypatch.setattr(runner, "LEDGER", ledger)
    monkeypatch.setattr(
        runner, "sources", lambda: (documents, {"historical": list(historical)}, identity)
    )
    monkeypatch.setattr(runner, "load_examples", lambda _path: examples)
    first = freeze(pool, "completed-study.json")
    monkeypatch.setattr(runner, "observed_training_contexts", lambda: selected(first))
    second_output = ledger.parent / "future/next-study.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_leakage.py",
            "freeze",
            "--output",
            str(second_output),
            "--size",
            "2",
            "--seed",
            "42",
        ],
    )
    runner.main()
    assert not selected(first) & selected(json.loads(second_output.read_text()))


def test_unregistered_prediction_report_blocks_future_sampling(pool, monkeypatch, runner):
    ledger, identity, examples, documents, historical = pool
    monkeypatch.setattr(runner, "LEDGER", ledger)
    monkeypatch.setattr(
        runner, "sources", lambda: (documents, {"historical": list(historical)}, identity)
    )
    monkeypatch.setattr(runner, "load_examples", lambda _path: examples)
    monkeypatch.setattr(runner, "observed_training_contexts", lambda: {"6"})
    output = ledger.parent / "future/unsafe.json"
    monkeypatch.setattr(
        "sys.argv",
        ["audit_leakage.py", "freeze", "--output", str(output), "--size", "2", "--seed", "42"],
    )
    with pytest.raises(ValueError, match="additional consumed"):
        runner.main()
    assert not output.exists()


def test_nested_reports_and_interrupted_checkpoints_are_consumed(tmp_path, monkeypatch, runner):
    monkeypatch.chdir(tmp_path)
    docs = [Document(str(i), (("Metric", str(i)),), ()) for i in range(3)]
    examples = [Example(d, f"q-{d.id}", "Metric?", 1, "", "arithmetic") for d in docs]
    monkeypatch.setattr(runner, "load_examples", lambda _path: examples)
    write_json(Path("reports/nested/completed.json"), {"predictions": [{"document_id": "1"}]})
    write_json(Path("artifacts/pilot/checkpoint.json"), {"predictions": [{"question_id": "q-2"}]})
    assert runner.observed_training_contexts() == {"1", "2"}


def test_modified_dev_and_matching_receipt_cannot_change_retired_sources(
    tmp_path, monkeypatch, runner
):
    monkeypatch.chdir(tmp_path)
    train = [
        {
            "table": {"uid": "train", "table": [["Revenue", "100"]]},
            "paragraphs": [],
            "questions": [],
        }
    ]
    dev = [
        {"table": {"uid": "dev", "table": [["Revenue", "200"]]}, "paragraphs": [], "questions": []}
    ]
    for path, value in [(runner.TRAIN, train), (runner.DEV, dev)]:
        write_json(path, value)
        split = "train" if path == runner.TRAIN else "dev"
        write_json(
            path.with_suffix(".json.receipt.json"),
            {
                "sha256": runner.file_sha256(path),
                "bytes": path.stat().st_size,
                "url": f"{runner.ROOT_URL}/tatqa_dataset_{split}.json",
            },
        )
    monkeypatch.setattr(
        runner,
        "EXPECTED_SOURCE_SHA256",
        {path: runner.file_sha256(path) for path in (runner.TRAIN, runner.DEV)},
    )
    manifests = [
        {"previously_used_document_ids": [], "development": [], "calibration": []},
        {"data_sha256": runner.file_sha256(runner.TRAIN), "examples": []},
        {"data_sha256": runner.file_sha256(runner.TRAIN), "examples": []},
    ]
    for path, value in zip(runner.HISTORY, manifests, strict=True):
        write_json(path, value)
    monkeypatch.setattr(
        runner,
        "EXPECTED_HISTORY_SHA256",
        tuple(runner.file_sha256(path) for path in runner.HISTORY),
    )
    assert runner.sources()[1]["retired_public_development"] == ["dev"]
    write_json(runner.DEV, [])
    write_json(
        runner.DEV.with_suffix(".json.receipt.json"),
        {
            "sha256": runner.file_sha256(runner.DEV),
            "bytes": runner.DEV.stat().st_size,
            "url": f"{runner.ROOT_URL}/tatqa_dataset_dev.json",
        },
    )
    with pytest.raises(ValueError, match="Pinned source"):
        runner.sources()
