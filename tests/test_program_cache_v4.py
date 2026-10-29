"""Hostile cache files must not bypass the serving parser's resource limits."""

import io

import pytest

from financial_qa.executor import ExecutionError
from financial_qa.model_client import MAX_RESPONSE_BYTES
from financial_qa.program_cache_v4 import read_program_cache


@pytest.mark.parametrize("raw", [b"[" * 3000 + b"]" * 3000, b'{"value":NaN}', b"\xff", b"[]"])
def test_malformed_deep_nonfinite_and_nonobject_cache_is_controlled(tmp_path, raw):
    path = tmp_path / "cache.json"
    path.write_bytes(raw)
    with pytest.raises(ExecutionError, match="cache"):
        read_program_cache(path)


def test_actual_read_is_bounded_without_trusting_cached_stat():
    counts = []

    class GrowingStream(io.BytesIO):
        def read(self, count):
            counts.append(count)
            return super().read(count)

    class CachePath:
        def open(self, _mode):
            return GrowingStream(b" " * (MAX_RESPONSE_BYTES + 100))

    with pytest.raises(ExecutionError):
        read_program_cache(CachePath())
    assert counts == [MAX_RESPONSE_BYTES + 1]


def test_missing_cache_is_a_controlled_failure(tmp_path):
    with pytest.raises(ExecutionError):
        read_program_cache(tmp_path / "removed.json")
