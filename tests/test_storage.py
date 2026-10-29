"""Cached downloads retain their original size and provenance requirements."""

import hashlib
import json

import pytest

from financial_qa.storage import download

URL = "https://example.com/dataset.json"


def cached_file(tmp_path, payload=b"cached financial table"):
    path = tmp_path / "dataset.json"
    path.write_bytes(payload)
    receipt = {"url": URL, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    path.with_suffix(".json.receipt.json").write_text(json.dumps(receipt))
    return path, receipt


def test_cached_download_cannot_bypass_a_smaller_size_budget(tmp_path):
    path, _ = cached_file(tmp_path)
    with pytest.raises(ValueError, match="size"):
        download(URL, path, max_bytes=5)
    assert path.read_bytes() == b"cached financial table"


@pytest.mark.parametrize("receipt", [None, [], {"url": URL}, {"url": URL, "bytes": True}])
def test_malformed_cached_receipt_fails_with_a_validation_error(tmp_path, receipt):
    path, _ = cached_file(tmp_path)
    path.with_suffix(".json.receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt|provenance"):
        download(URL, path)


def test_cached_receipt_size_must_match_the_verified_bytes(tmp_path):
    path, receipt = cached_file(tmp_path)
    receipt["bytes"] += 1
    path.with_suffix(".json.receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="provenance"):
        download(URL, path)


def test_valid_cache_does_not_use_an_unbounded_read(tmp_path, monkeypatch):
    path, receipt = cached_file(tmp_path)

    def unbounded_read_forbidden(_):
        raise AssertionError("Cached archive verification must stream its hash")

    monkeypatch.setattr(type(path), "read_bytes", unbounded_read_forbidden)
    assert download(URL, path) == receipt
