"""Bound cached model responses by actual bytes read, independent of file size."""

from __future__ import annotations

import json
from pathlib import Path

from .executor import ExecutionError
from .model_client import MAX_RESPONSE_BYTES


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def read_program_cache(path: Path) -> dict:
    """Read an object with a hard byte budget, including files growing after stat.

    The caller separately verifies request/model identities. Deeply nested or
    invalid local artifacts become controlled abstentions, not parser exceptions.
    """
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("Response cache exceeds the byte budget")
        result = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
        if not isinstance(result, dict):
            raise ValueError("Response cache must contain an object")
        return result
    except (OSError, ValueError, RecursionError) as error:
        raise ExecutionError(
            "Layout program cache is unreadable, malformed or oversized"
        ) from error
