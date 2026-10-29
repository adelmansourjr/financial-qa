"""Resume the predeclared v3 calibration/final stages without changing inference.

This orchestration helper can wait for the live model calibration already running.
It skips completed immutable reports and invokes the same checked experiment CLI.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from financial_qa.calibration import POLICIES


def command(stage: str, policy: str | None = None) -> None:
    arguments = [sys.executable, "scripts/experiment_v3.py", stage]
    if policy:
        arguments.extend(["--policy", policy])
    print(json.dumps({"starting_stage": stage, "policy": policy}), flush=True)
    subprocess.run(arguments, check=True)


def main() -> None:
    deadline = time.monotonic() + 7200
    while not Path("reports/v3-calibration-model.json").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Waiting for live400-document calibration exceeded two hours")
        time.sleep(3)
    for policy in POLICIES:
        if not Path(f"reports/v3-calibration-{policy}.json").exists():
            command("calibration", policy)
    if not Path("reports/v3-risk-certificate.json").exists():
        command("certify")
    # Live model generation happens once; the remaining policies replay exactly
    # the same source-grounded programs after deterministic/rule gating.
    for policy in ("model", "semantic", "hybrid", "agreement", "strict_agreement"):
        if not Path(f"reports/v3-final-{policy}.json").exists():
            command("final", policy)
    print(
        json.dumps(
            {
                "completed": True,
                "calibration_documents": 400,
                "final_documents": 86,
                "policies": list(POLICIES),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
