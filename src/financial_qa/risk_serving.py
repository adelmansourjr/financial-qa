"""Serving-only model identity guard around the frozen risk-controlled planner.

This adds an availability restriction, not a new inference policy. The experiment's
frozen prediction components remain unchanged. Metadata catches ordinary server
restarts with a different model path/alias; it cannot attest what an actively
malicious server computed or detect weights replaced behind an unchanged path.
"""

from __future__ import annotations

from pathlib import Path

from .data import Document, Evidence
from .executor import Plan
from .reasoning_v3 import RiskControlledPlanner


class ServingIdentityGuard:
    """Permanently fail closed when the model identity changes after startup."""

    retrieval_mode = "complete"

    def __init__(self, planner: RiskControlledPlanner, model_path: Path):
        self.planner = planner
        self.model_path = model_path.resolve()
        self.server_root = Path(__file__).resolve().parents[2]

    @property
    def state(self) -> dict:
        return self.planner.state

    def plan(self, document: Document, question: str, evidence: list[Evidence]) -> Plan:
        engine = self.planner.engine
        if engine is not None:
            try:
                properties = engine.client.properties()
                path = Path(properties["model_path"])
                if not path.is_absolute():
                    path = self.server_root / path
                aliases = properties.get("model_alias", properties.get("model_aliases", []))
                if isinstance(aliases, str):
                    aliases = [aliases]
                if path.resolve() != self.model_path or engine.profile.model not in aliases:
                    raise ValueError("Local server model alias/path changed after startup")
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.planner.engine = None
                self.planner.state = {
                    "certified": False,
                    "state": "model_identity_changed_fail_closed",
                    "reason": str(error),
                }
                return Plan(
                    abstain=True, reason="Risk-controlled model identity verification failed"
                )
        return self.planner.plan(document, question, evidence)


class UnavailableRiskPlanner:
    """Expose an explicit unavailable state when startup configuration is missing."""

    retrieval_mode = "complete"

    def __init__(self, reason: str):
        self.state = {
            "version": 3,
            "certified": False,
            "state": "configuration_unavailable_fail_closed",
            "reason": reason,
        }

    def plan(self, document: Document, question: str, evidence: list[Evidence]) -> Plan:
        return Plan(
            abstain=True, reason="Risk-controlled mode unavailable: " + self.state["reason"]
        )
