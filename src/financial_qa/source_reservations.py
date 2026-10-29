"""Append-only local reservations for future source-group QA samples.

One exclusive writer reserves groups before publishing a manifest. A crash can
leave a reservation without a manifest, which conservatively consumes its sources.
Checksums detect ordinary changes; they are not signatures against malicious
rewriting/deletion of the entire local ledger. Retain ledger history in version
control, and never delete reservations to reuse inspected examples.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

from .calibration import digest
from .data import Document, Example
from .split_integrity import guarded_sample, validate_guarded_sample
from .storage import write_json

MAX_LEDGER_BYTES = 4_000_000
SCHEMA = "qa-source-reservations-v1"


def _destination(ledger: Path, output: Path) -> str:
    """Keep all manifests in one discoverable tree, with portable canonical paths."""
    root = ledger.parent.resolve()
    resolved = output.resolve()
    if not resolved.is_relative_to(root / "future") or resolved.suffix != ".json":
        raise ValueError("Future manifests must be JSON files inside benchmarks/future")
    return resolved.relative_to(root).as_posix()


@contextmanager
def exclusive_writer(path: Path):
    """Fail promptly on concurrent/stale writers; never steal a potentially live lock."""
    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError(
            "Reservation writer lock exists; check its owner before recovery"
        ) from error
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()) + "\n")
        yield
    finally:
        lock.unlink(missing_ok=True)


def read_reservations(
    ledger: Path,
    identity: dict,
    examples: list[Example],
    documents: list[Document],
    historical_ids: set[str],
) -> list[dict]:
    """Reconstruct every prefix so later reservations cannot reuse earlier groups."""
    if not ledger.exists():
        raise ValueError("Reservation ledger is missing; restore it rather than resetting history")
    try:
        with ledger.open("rb") as stream:
            raw = stream.read(MAX_LEDGER_BYTES + 1)
        if len(raw) > MAX_LEDGER_BYTES:
            raise ValueError("Reservation ledger exceeds metadata budget")
        saved = json.loads(raw)
        if not isinstance(saved, dict) or set(saved) != {"schema", "entries"}:
            raise ValueError("Malformed reservation ledger")
        if saved["schema"] != SCHEMA or not isinstance(saved["entries"], list):
            raise ValueError("Unsupported reservation ledger schema")
        consumed = set(historical_ids)
        previous = "0" * 64
        destinations = set()
        for index, entry in enumerate(saved["entries"], 1):
            if (
                not isinstance(entry, dict)
                or set(entry)
                != {"index", "output", "identity", "sampling", "previous_sha256", "entry_sha256"}
                or type(entry["index"]) is not int
                or entry["index"] != index
                or entry["identity"] != identity
                or entry["previous_sha256"] != previous
                or not isinstance(entry["output"], str)
                or entry["output"] in destinations
                or entry["entry_sha256"]
                != digest({key: value for key, value in entry.items() if key != "entry_sha256"})
            ):
                raise ValueError("Reservation chain, identity or unique destination changed")
            if _destination(ledger, ledger.parent / entry["output"]) != entry["output"]:
                raise ValueError("Reservation output is not a canonical future-manifest path")
            sampling = entry["sampling"]
            validate_guarded_sample(
                sampling, examples, documents, consumed, sampling["size"], sampling["seed"]
            )
            consumed.update(row["document_id"] for row in sampling["examples"])
            destinations.add(entry["output"])
            previous = entry["entry_sha256"]
        # Catch accidental ledger rollback/deletion of entries while manifests
        # still exist. A missing publication is allowed after a write-ahead crash.
        by_output = {entry["output"]: entry for entry in saved["entries"]}
        for output in sorted((ledger.parent / "future").rglob("*.json")):
            destination = _destination(ledger, output)
            if destination not in by_output:
                raise ValueError("Unregistered future manifest; restore the complete ledger")
            with output.open("rb") as stream:
                raw_manifest = stream.read(MAX_LEDGER_BYTES + 1)
            if len(raw_manifest) > MAX_LEDGER_BYTES or digest(json.loads(raw_manifest)) != digest(
                by_output[destination]
            ):
                raise ValueError("Published manifest differs from its registered reservation")
        return saved["entries"]
    except (KeyError, TypeError, UnicodeError, RecursionError) as error:
        raise ValueError("Malformed reservation metadata") from error


def freeze_reservation(
    ledger: Path,
    output: Path,
    identity: dict,
    examples: list[Example],
    documents: list[Document],
    historical_ids: set[str],
    size: int,
    seed: int,
) -> dict:
    """Append before publishing; publication failure never reopens reserved sources."""
    with exclusive_writer(ledger):
        entries = read_reservations(ledger, identity, examples, documents, historical_ids)
        destination = _destination(ledger, output)
        if output.exists() or any(entry["output"] == destination for entry in entries):
            raise ValueError("Preserve existing/reserved output; refusing replacement")
        consumed = set(historical_ids)
        for entry in entries:
            consumed.update(row["document_id"] for row in entry["sampling"]["examples"])
        sampling = guarded_sample(examples, documents, consumed, size, seed)
        entry = {
            "index": len(entries) + 1,
            "output": destination,
            "identity": identity,
            "sampling": sampling,
            "previous_sha256": entries[-1]["entry_sha256"] if entries else "0" * 64,
        }
        entry["entry_sha256"] = digest(entry)
        saved = {"schema": SCHEMA, "entries": [*entries, entry]}
        if len((json.dumps(saved, indent=2, allow_nan=False) + "\n").encode()) > MAX_LEDGER_BYTES:
            raise ValueError("Reservation ledger capacity exceeded; preserve existing history")
        # Write-ahead order intentionally wastes capacity rather than risk overlap.
        write_json(ledger, saved)
        write_json(output, entry)
        return entry


def validate_reservation(
    ledger: Path,
    output: Path,
    identity: dict,
    examples: list[Example],
    documents: list[Document],
    historical_ids: set[str],
    size: int,
    seed: int,
) -> dict:
    """Require the published manifest to match its validated reservation exactly."""
    entries = read_reservations(ledger, identity, examples, documents, historical_ids)
    selected = [entry for entry in entries if entry["output"] == _destination(ledger, output)]
    if len(selected) != 1:
        raise ValueError("Future manifest has no unique registered reservation")
    entry = selected[0]
    if (
        entry["sampling"]["size"] != size
        or entry["sampling"]["seed"] != seed
        or type(size) is not int
        or type(seed) is not int
    ):
        raise ValueError("External sample size/seed differs from the reservation")
    with output.open("rb") as stream:
        raw = stream.read(MAX_LEDGER_BYTES + 1)
    if len(raw) > MAX_LEDGER_BYTES or digest(json.loads(raw)) != digest(entry):
        raise ValueError("Published future manifest differs from its reservation")
    return entry
