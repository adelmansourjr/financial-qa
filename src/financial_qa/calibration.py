"""Finite-policy selective-risk calibration with an exact binomial bound.

A policy is frozen before independent calibration labels are observed. For each
policy we bound error probability conditional on acceptance. Bonferroni correction
makes the bounds simultaneous across the finite grid, allowing coverage-based
selection without pretending that a selected empirical accuracy is a certificate.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .storage import write_json

POLICIES = ("semantic", "model", "hybrid", "agreement", "strict_agreement")


def binomial_cdf(errors: int, count: int, probability: float) -> float:
    """Evaluate P[Binomial(count,p)<=errors] stably without optional SciPy."""
    if probability == 0:
        return 1.0
    if probability == 1:
        return float(errors == count)
    logs = [
        math.lgamma(count + 1)
        - math.lgamma(index + 1)
        - math.lgamma(count - index + 1)
        + index * math.log(probability)
        + (count - index) * math.log1p(-probability)
        for index in range(errors + 1)
    ]
    maximum = max(logs)
    return min(1.0, math.exp(maximum) * sum(math.exp(value - maximum) for value in logs))


def upper_error_bound(errors: int, accepted: int, alpha: float) -> float:
    """One-sided Clopper-Pearson upper bound, including zero-acceptance cases."""
    if (
        type(errors) is not int
        or type(accepted) is not int
        or isinstance(alpha, bool)
        or not isinstance(alpha, (float, int))
        or not math.isfinite(alpha)
        or not 0 < alpha < 1
        or not 0 <= errors <= accepted
    ):
        raise ValueError("Invalid binomial calibration counts or tail probability")
    if accepted == 0 or errors == accepted:
        return 1.0
    lower, upper = 0.0, 1.0
    for _ in range(70):
        middle = (lower + upper) / 2
        if binomial_cdf(errors, accepted, middle) > alpha:
            lower = middle
        else:
            upper = middle
    return upper


def digest(value: Any) -> str:
    """Canonical JSON identity for protocols, manifests and calibration outputs."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash model bytes with bounded memory, including multi-gigabyte weights."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def code_fingerprint() -> dict[str, str]:
    """Bind acceptance to all prediction components, not just a model nickname."""
    directory = Path(__file__).parent
    names = (
        "reasoning_v3.py",
        "selective.py",
        "structured.py",
        "context.py",
        "data.py",
        "retrieval.py",
        "executor.py",
        "service.py",
        "calibration.py",
        "model_client.py",
    )
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}


def calibrate(
    reports: dict[str, dict],
    identity: dict,
    *,
    target_error: float = 0.10,
    alpha: float = 0.05,
    minimum_accepted: int = 20,
) -> dict:
    """Certify a predeclared grid and select maximum coverage among valid policies.

    The independent unit is a document, with exactly one sampled question each.
    Calibration records must use the identical nonempty question/document set for
    every policy. The reported confidence is conditional on IID calibration/future
    documents and fixed inference; it is not a distribution-shift guarantee.
    """
    if set(reports) != set(POLICIES):
        raise ValueError("Calibration must include every predeclared policy exactly once")
    if (
        type(minimum_accepted) is not int
        or minimum_accepted < 1
        or isinstance(target_error, bool)
        or isinstance(alpha, bool)
        or not isinstance(target_error, (float, int))
        or not isinstance(alpha, (float, int))
        or not math.isfinite(target_error)
        or not math.isfinite(alpha)
        or not 0 < target_error < 1
        or not 0 < alpha < 1
    ):
        raise ValueError("Invalid calibration target, alpha or support threshold")
    manifest_rows = identity.get("calibration_membership")
    manifest_membership = (
        {(row["id"], row["document_id"]) for row in manifest_rows} if manifest_rows else None
    )
    if manifest_rows is not None and (
        not manifest_rows or len(manifest_membership) != len(manifest_rows)
    ):
        raise ValueError("Calibration manifest membership is empty or duplicated")
    expected = None
    summary = {}
    for policy in POLICIES:
        records = reports[policy]["predictions"]
        keys = [(row["id"], row["document_id"]) for row in records]
        if not keys or len(set(keys)) != len(keys) or len({doc for _, doc in keys}) != len(keys):
            raise ValueError("Calibration requires one question per unique document")
        if manifest_membership is not None and set(keys) != manifest_membership:
            raise ValueError(
                "Calibration outcomes differ from the frozen manifest membership/count"
            )
        if expected is None:
            expected = set(keys)
        elif expected != set(keys):
            raise ValueError("All policies must use identical calibration documents/questions")
        if any(
            type(row["abstained"]) is not bool
            or type(row["correct"]) is not bool
            or row["abstained"]
            and row["correct"]
            for row in records
        ):
            raise ValueError("Calibration outcomes must be exact boolean indicators")
        accepted = sum(not row["abstained"] for row in records)
        errors = sum(not row["correct"] and not row["abstained"] for row in records)
        bound = upper_error_bound(errors, accepted, alpha / len(POLICIES))
        summary[policy] = {
            "accepted": accepted,
            "errors": errors,
            "evaluated": len(records),
            "coverage": accepted / len(records),
            "empirical_accepted_accuracy": 1 - errors / accepted if accepted else None,
            "error_upper_bound": bound,
            "certified": accepted >= minimum_accepted and bound <= target_error,
            "prediction_sha256": digest(
                [
                    {key: row[key] for key in ("id", "document_id", "abstained", "correct")}
                    for row in records
                ]
            ),
        }
    qualifying = [name for name in POLICIES if summary[name]["certified"]]
    selected = (
        max(
            qualifying,
            key=lambda name: (summary[name]["coverage"], -summary[name]["error_upper_bound"]),
        )
        if qualifying
        else None
    )
    payload = {
        "version": 3,
        "method": "one-sided Clopper-Pearson with finite-grid Bonferroni",
        "target_error": target_error,
        "familywise_alpha": alpha,
        "policy_count": len(POLICIES),
        "minimum_accepted": minimum_accepted,
        "selected_policy": selected,
        "certified": selected is not None,
        "state": "certified_under_iid_assumptions" if selected else "uncertified_fail_closed",
        "assumptions": (
            "Fixed policies; one independently sampled question per IID document; "
            "future documents follow the calibration distribution. No guarantee "
            "under distribution shift or model/prompt drift."
        ),
        "identity": identity,
        "evidence": {
            policy: [
                {key: row[key] for key in ("id", "document_id", "abstained", "correct")}
                for row in reports[policy]["predictions"]
            ]
            for policy in POLICIES
        },
        "policies": summary,
    }
    return {**payload, "artifact_sha256": digest(payload)}


def save_certificate(path: Path, certificate: dict) -> None:
    if path.exists():
        raise ValueError("Refusing to overwrite a calibration certificate")
    write_json(path, certificate)


def validate_certificate(path: Path, expected_identity: dict) -> dict:
    """Fail closed on tampering, code/model/prompt/protocol mismatch or no evidence."""
    if path.stat().st_size > 2_000_000:
        raise ValueError("Calibration artifact exceeds the size budget")
    certificate = json.loads(path.read_text())
    if not isinstance(certificate, dict):
        raise ValueError("Calibration artifact must be an object")
    payload = {key: value for key, value in certificate.items() if key != "artifact_sha256"}
    if certificate.get("artifact_sha256") != digest(payload):
        raise ValueError("Calibration artifact checksum mismatch")
    if certificate.get("identity") != expected_identity:
        raise ValueError("Calibration identity does not match code/model/prompt/data protocol")
    if (
        certificate.get("version") != 3
        or certificate.get("policy_count") != len(POLICIES)
        or certificate.get("target_error") != 0.10
        or certificate.get("familywise_alpha") != 0.05
        or certificate.get("minimum_accepted") != 20
    ):
        raise ValueError("Calibration artifact changes the predeclared risk target/protocol")
    recomputed = calibrate(
        {policy: {"predictions": certificate["evidence"][policy]} for policy in POLICIES},
        expected_identity,
        target_error=certificate["target_error"],
        alpha=certificate["familywise_alpha"],
        minimum_accepted=certificate["minimum_accepted"],
    )
    if recomputed != certificate:
        raise ValueError("Calibration artifact disagrees with recomputed evidence and bounds")
    if not recomputed["certified"]:
        raise ValueError("No policy met the independent calibration risk target")
    return recomputed
