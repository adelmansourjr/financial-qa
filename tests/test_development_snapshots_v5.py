"""Preserve reproducible source identities for completed development experiments."""

import hashlib
import json
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
SNAPSHOTS = json.loads((PROJECT / "benchmarks/development-snapshots-v5.json").read_text())


@pytest.mark.parametrize("version", SNAPSHOTS["versions"], ids=lambda item: item["revision"])
def test_development_source_snapshot_matches_the_measured_report(version):
    report_path = PROJECT / version["report"]
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == version["report_sha256"]
    report = json.loads(report_path.read_text())
    assert report["stage"] == "development"
    assert report["evaluated"] == len(report["predictions"]) == 100
    assert report["correct"] == sum(row["correct"] for row in report["predictions"])
    for path, saved in version["files"].items():
        digest = hashlib.sha256(saved["source"].encode()).hexdigest()
        assert digest == saved["sha256"]
        if path.startswith("scripts/"):
            assert digest == report["identity"]["runner_sha256"]
        else:
            key = Path(path).stem + "_sha256"
            assert digest == report["identity"]["methods"]["candidate"][key]
