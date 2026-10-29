"""One predeclared temporal-agreement gate with independent risk calibration.

The policy was developed only on the original training-development contexts.
Its future conditional-error interpretation requires IID reference contexts;
context IDs do not verify independent source filings or arbitrary-user safety.
"""

from __future__ import annotations

import json
import platform
import re
from importlib.metadata import version
from pathlib import Path

from .calibration import digest, file_sha256, upper_error_bound
from .data import Document, Evidence
from .executor import ExecutionError, Plan, execute
from .grounding_v4 import layout_catalog
from .program_cache_v4 import read_program_cache
from .reasoning_v3 import runtime_signature
from .reasoning_v4 import LayoutProgramPlanner, explicit_operation
from .selective import YEARS, SemanticBaseline

POLICY = "same_row_temporal_agreement"
TARGET_ERROR = 0.10
ALPHA = 0.05
MINIMUM_ACCEPTED = 20
CALIBRATION_CONTEXTS = 300


def eligible_question(question: str) -> bool:
    """Short-circuit necessary gate conditions before expensive generation."""
    return (
        explicit_operation(question) in {"difference", "percent_change"}
        and len(set(YEARS.findall(question))) == 2
    )


def acceptance_plan(
    document: Document, question: str, evidence: list[Evidence], plan: Plan
) -> Plan:
    """Accept only two-period same-row agreement with known money/count units."""
    if plan.abstain:
        return plan
    declined = Plan(abstain=True, reason="Version4 temporal-agreement requirements are not met")
    if not eligible_question(question):
        return declined
    requested = set(YEARS.findall(question))
    names = list(dict.fromkeys(re.findall(r"\br\d+c\d+\b", plan.expression)))
    if (
        len(requested) != 2
        or len(names) != 2
        or re.search(r"\bp\d+n\d+\b", plan.expression)
        or len({name.split("c")[0] for name in names}) != 1
    ):
        return declined
    catalog = layout_catalog(document, evidence)
    if any(
        name not in catalog
        or catalog[name]["dimension"] not in {"money", "count"}
        or len(catalog[name]["years"]) != 1
        for name in names
    ):
        return declined
    if {catalog[name]["years"][0] for name in names} != requested:
        return declined
    rule = SemanticBaseline().plan(document, question, evidence)
    try:
        if (
            rule.abstain
            or rule.scale != plan.scale
            or execute(rule, evidence) != execute(plan, evidence)
        ):
            return declined
    except ExecutionError:
        return declined
    return plan


def policy_identity(engine: LayoutProgramPlanner) -> dict:
    """Bind the single acceptance rule to the full frozen base inference identity."""
    return {
        "base_inference": engine.identity(),
        "policy": POLICY,
        "acceptance_code_sha256": file_sha256(Path(__file__)),
        "environment": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pydantic": version("pydantic"),
        },
    }


def certificate_from_evidence(evidence: list[dict], identity: dict, membership: list[dict]) -> dict:
    """Recompute support/error bounds; no threshold or policy search occurs here."""
    expected = [(row["id"], row["document_id"]) for row in membership]
    observed = [(row["id"], row["document_id"]) for row in evidence]
    if (
        len(expected) != CALIBRATION_CONTEXTS
        or observed != expected
        or len(set(expected)) != len(expected)
        or len({doc for _, doc in expected}) != len(expected)
    ):
        raise ValueError("Calibration needs exact ordered unique context membership")
    if any(not isinstance(value, str) or not value for row in expected for value in row):
        raise ValueError("Calibration IDs must be nonempty strings")
    if any(
        set(row) != {"id", "document_id", "accepted", "correct"}
        or type(row["accepted"]) is not bool
        or type(row["correct"]) is not bool
        or not row["accepted"]
        and row["correct"]
        for row in evidence
    ):
        raise ValueError("Calibration outcomes must be strict boolean indicators")
    accepted = sum(row["accepted"] for row in evidence)
    errors = sum(row["accepted"] and not row["correct"] for row in evidence)
    bound = upper_error_bound(errors, accepted, ALPHA)
    qualifies = accepted >= MINIMUM_ACCEPTED and bound <= TARGET_ERROR
    payload = {
        "version": 4,
        "policy": POLICY,
        "policy_count": 1,
        "method": "One-sided exact Clopper-Pearson; one policy fixed before fresh calibration",
        "target_error": TARGET_ERROR,
        "familywise_alpha": ALPHA,
        "minimum_accepted": MINIMUM_ACCEPTED,
        "evaluated": len(evidence),
        "accepted": accepted,
        "errors": errors,
        "coverage": accepted / len(evidence),
        "empirical_accepted_accuracy": 1 - errors / accepted if accepted else None,
        "error_upper_bound": bound,
        "certified": qualifies,
        "state": "certified_under_iid_assumptions" if qualifies else "uncertified_fail_closed",
        "identity": identity,
        "membership": membership,
        "evidence": evidence,
        "assumptions": (
            "One arithmetic question per IID reference context; future accepted "
            "contexts follow that distribution. Context disjointness does not prove "
            "independent filings. No guarantee under shift or inference drift."
        ),
    }
    return {**payload, "artifact_sha256": digest(payload)}


def validate_certificate(path: Path, identity: dict, manifest: Path, data: Path) -> dict:
    """Reject truncated, rehashed, relabeled, or incompatible local certificates."""
    # Certificates contain300 small evidence rows, comfortably within this limit.
    with path.open("rb") as stream:
        raw = stream.read(256_001)
    if len(raw) > 256_000:
        raise ValueError("Version4 certificate exceeds metadata budget")
    try:
        saved = json.loads(raw)
        membership = read_program_cache(manifest)["examples"]
        expected_identity = {
            **identity,
            "manifest_sha256": file_sha256(manifest),
            "data_sha256": file_sha256(data),
        }
        if saved["identity"] != expected_identity:
            raise ValueError("Version4 calibration identity differs from runtime/data")
        expected = certificate_from_evidence(saved["evidence"], expected_identity, membership)
        if saved != expected:
            raise ValueError("Version4 certificate differs from recomputed evidence")
        return saved
    except (KeyError, TypeError, RecursionError) as error:
        raise ValueError("Version4 certificate is malformed") from error


class TemporalRiskPlanner:
    """Fail closed on unavailable evidence, failed calibration or runtime drift."""

    retrieval_mode = "complete"

    def __init__(
        self,
        engine: LayoutProgramPlanner,
        certificate: Path,
        manifest: Path,
        data: Path,
        model_path: Path,
    ):
        self.profile = engine.profile
        self.engine = None
        self.model_path = model_path.resolve()
        self.server_root = Path(__file__).resolve().parents[2]
        self.expected = policy_identity(engine)
        self.state = {"version": 4, "certified": False, "state": "uncertified_fail_closed"}
        try:
            saved = validate_certificate(certificate, self.expected, manifest, data)
            if not saved["certified"]:
                raise ValueError("Version4 temporal policy failed independent calibration")
            if file_sha256(model_path) != engine.profile.model_sha256:
                raise ValueError("Version4 model bytes differ from calibrated model")
            self._check_runtime(engine)
            self.engine = engine
            self.state = {
                key: saved[key]
                for key in (
                    "version",
                    "policy",
                    "certified",
                    "state",
                    "target_error",
                    "familywise_alpha",
                    "coverage",
                    "error_upper_bound",
                    "artifact_sha256",
                    "assumptions",
                )
            }
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.state["reason"] = str(error)[:400]

    def _check_runtime(self, engine: LayoutProgramPlanner) -> None:
        props = engine.client.properties()
        aliases = props.get("model_alias", props.get("model_aliases", []))
        if isinstance(aliases, str):
            aliases = [aliases]
        path = Path(props["model_path"])
        if not path.is_absolute():
            path = self.server_root / path
        if (
            path.resolve() != self.model_path
            or engine.profile.model not in aliases
            or runtime_signature(props) != engine.profile.runtime
            or policy_identity(engine) != self.expected
        ):
            raise ValueError("Version4 model/runtime/code identity changed")

    def plan(self, document: Document, question: str, evidence: list[Evidence]) -> Plan:
        if self.engine is None:
            return Plan(
                abstain=True,
                reason="Version4 risk-controlled mode unavailable: "
                + self.state.get("reason", "Independent risk target not qualified"),
            )
        try:
            self._check_runtime(self.engine)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.engine = None
            self.state = {
                "version": 4,
                "certified": False,
                "state": "runtime_changed_fail_closed",
                "reason": str(error)[:400],
            }
            return Plan(abstain=True, reason="Version4 runtime verification failed")
        if not eligible_question(question):
            return Plan(abstain=True, reason="Version4 temporal-agreement requirements are not met")
        proposed = self.engine.plan(document, question, evidence)
        return acceptance_plan(document, question, evidence, proposed)
