"""Reproducible data, evaluation, and local serving commands."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .data import REVISION, fetch_dataset, load_examples
from .evaluation import evaluate
from .planner import LocalLLM, TableBaseline
from .storage import write_json
from .structured import HeaderAwareBaseline, StructuredLLM

V3_BACKENDS = {"program-v3", "risk-controlled"}
PROGRAM_BACKENDS = V3_BACKENDS | {"program-v4", "risk-controlled-v4", "program-v5"}
MODEL_BACKENDS = {"llm", "structured", "selective", "agreement", "hybrid"} | PROGRAM_BACKENDS


def model_defaults(backend: str, command: str) -> tuple[str, str]:
    """Select defaults by backend; an explicit endpoint is never rewritten."""
    if backend in PROGRAM_BACKENDS:
        return "http://127.0.0.1:18081/v1", "qwen3-4b-q4-nonthinking"
    if command == "benchmark":
        return "http://127.0.0.1:18080/v1", "qwen2.5-1.5b-q4"
    return "http://127.0.0.1:8080/v1", "local"


def load_runtime_profile(path: Path) -> dict:
    """Read only a bounded, complete runtime signature before creating a planner."""
    with path.open("rb") as stream:
        raw = stream.read(16_385)
    if len(raw) > 16_384:
        raise ValueError("Runtime profile exceeds the metadata size budget")
    try:
        runtime = json.loads(raw.decode("utf-8"))
    except (ValueError, RecursionError) as error:
        raise ValueError("Runtime profile is malformed JSON") from error
    fields = {
        "build_info",
        "n_ctx",
        "chat_template_sha256",
        "default_generation_sha256",
        "reasoning_format",
    }
    if not isinstance(runtime, dict) or set(runtime) != fields:
        raise ValueError("Runtime profile must contain exactly the runtime-signature fields")
    if type(runtime["n_ctx"]) is not int or not 1 <= runtime["n_ctx"] <= 2**31 - 1:
        raise ValueError("Runtime context size must be a positive integer")
    for field, limit in (("build_info", 256), ("reasoning_format", 64)):
        value = runtime[field]
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise ValueError("Runtime build and reasoning format must be bounded nonempty strings")
    for field in ("chat_template_sha256", "default_generation_sha256"):
        if not isinstance(runtime[field], str) or not re.fullmatch(r"[a-f0-9]{64}", runtime[field]):
            raise ValueError("Runtime signature hashes must be SHA-256 hex strings")
    return runtime


def inference_metadata(planner, backend: str) -> dict:
    """Describe the actual configured generator, including guarded unavailability."""
    if backend in {"program-v4", "risk-controlled-v4", "program-v5"}:
        engine = getattr(planner, "engine", None) if backend == "risk-controlled-v4" else planner
        return {
            "model": planner.profile.model,
            "generation": {**engine.identity(), "response_format": "json_schema"}
            if engine
            else None,
            "reasoning": planner.state,
            "cache": "Raw responses keyed by model hash, endpoint and complete request",
        }
    if backend in V3_BACKENDS:
        profile = getattr(planner, "profile", None)
        state = getattr(planner, "state", None)
        return {
            "model": profile.model if profile else None,
            "generation": {**profile.identity(), "response_format": "json_schema"}
            if profile
            else None,
            "cache": "Raw response replay keyed by model weight hash, endpoint and complete request"
            if profile
            else None,
            "reasoning": state,
        }
    return {
        "model": planner.model if backend in MODEL_BACKENDS else None,
        "generation": {
            "temperature": 0,
            "max_tokens": 220 if backend == "llm" else 150,
            "response_format": "json_object" if backend == "llm" else "json_schema",
        }
        if backend in MODEL_BACKENDS
        else None,
        "cache": "Raw response replay keyed by endpoint, model, prompt and settings"
        if backend in MODEL_BACKENDS
        else None,
    }


def make_planner(
    backend: str,
    url: str | None,
    model: str | None,
    cache: Path | None = None,
    *,
    program_policy="model",
    risk_certificate=Path("reports/v3-risk-certificate.json"),
    model_path=Path("local-runtime/qwen3-4b-q4.gguf"),
    runtime_profile=Path("benchmarks/runtime-v3.json"),
    calibration_manifest=Path("benchmarks/train-v3-400.json"),
    calibration_data=Path("data/tatqa_train.json"),
    v4_certificate=Path("reports/v4-temporal-certificate.json"),
    v4_calibration_manifest=Path("benchmarks/calibration-v4.json"),
):
    default_url, default_model = model_defaults(backend, "serve")
    url, model = url or default_url, model or default_model
    if backend == "program-v5":
        from .reasoning_v3 import RuntimeProfile
        from .reasoning_v5 import CandidateProgramPlanner

        if program_policy != "model":
            raise ValueError("Version5 exposes one source-exact/model candidate; use model policy")
        runtime = load_runtime_profile(runtime_profile)
        return CandidateProgramPlanner(
            RuntimeProfile(
                model,
                "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
                runtime=runtime,
            ),
            url,
            cache,
        )
    if backend in {"program-v4", "risk-controlled-v4"}:
        from .reasoning_v3 import RuntimeProfile
        from .reasoning_v4 import LayoutProgramPlanner

        if program_policy != "model":
            raise ValueError(
                "Version4 has one experimental layout policy; use --program-policy model"
            )
        try:
            runtime = load_runtime_profile(runtime_profile)
        except (OSError, ValueError):
            if backend != "risk-controlled-v4":
                raise
            from .risk_serving import UnavailableRiskPlanner

            unavailable = UnavailableRiskPlanner("Version4 runtime profile unavailable or invalid")
            unavailable.state["version"] = 4
            unavailable.profile = RuntimeProfile(
                model, "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"
            )
            return unavailable
        profile = RuntimeProfile(
            model,
            "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
            runtime=runtime,
        )
        engine = LayoutProgramPlanner(profile, url, cache)
        if backend == "program-v4":
            return engine
        from .selective_v4 import TemporalRiskPlanner

        return TemporalRiskPlanner(
            engine, v4_certificate, v4_calibration_manifest, calibration_data, model_path
        )
    if backend in V3_BACKENDS:
        from .reasoning_v3 import ProgramPlanner, RiskControlledPlanner, RuntimeProfile
        from .risk_serving import ServingIdentityGuard, UnavailableRiskPlanner

        try:
            runtime = load_runtime_profile(runtime_profile)
        except (OSError, ValueError) as error:
            message = "Runtime profile is unavailable or invalid; provide --runtime-profile"
            if backend == "risk-controlled":
                return UnavailableRiskPlanner(message)
            raise ValueError(message) from error
        profile = RuntimeProfile(
            model,
            "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
            runtime=runtime,
        )
        if backend == "program-v3":
            return ProgramPlanner(profile, url, cache, program_policy)
        controlled = RiskControlledPlanner(
            risk_certificate, profile, model_path, calibration_manifest, calibration_data, url
        )
        guard = ServingIdentityGuard(controlled, model_path)
        guard.profile = profile
        return guard

    if backend in {"selective", "agreement", "hybrid", "semantic"}:
        from .selective import SelectiveLLM, SemanticBaseline

        return (
            SemanticBaseline()
            if backend == "semantic"
            else SelectiveLLM(
                url,
                model,
                cache,
                policy=backend if backend in {"agreement", "hybrid"} else "verified",
            )
        )
    if backend == "baseline":
        return TableBaseline()
    if backend == "header":
        return HeaderAwareBaseline()
    if backend == "structured":
        return StructuredLLM(url, model, cache)
    return LocalLLM(url, model, cache)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download")
    fetch.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    fetch.add_argument("--data-dir", type=Path, default=Path("data"))
    run = commands.add_parser("evaluate")
    run.add_argument("--data", type=Path, default=Path("data/tatqa_dev.json"))
    run.add_argument("--limit", type=int, default=100)
    run.add_argument(
        "--backend",
        choices=[
            "baseline",
            "header",
            "llm",
            "structured",
            "selective",
            "agreement",
            "hybrid",
            "semantic",
            "program-v3",
            "program-v4",
            "program-v5",
            "risk-controlled-v4",
            "risk-controlled",
        ],
        default="baseline",
    )
    run.add_argument("--llm-url")
    run.add_argument("--model")
    run.add_argument("--cache-dir", type=Path, default=Path("artifacts/llm-cache"))
    run.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8001)
    serve.add_argument(
        "--backend",
        choices=[
            "baseline",
            "header",
            "llm",
            "structured",
            "selective",
            "agreement",
            "hybrid",
            "semantic",
            "program-v3",
            "program-v4",
            "program-v5",
            "risk-controlled-v4",
            "risk-controlled",
        ],
        default="semantic",
    )
    serve.add_argument("--llm-url")
    serve.add_argument("--model")
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--data", type=Path, default=Path("data/tatqa_dev.json"))
    benchmark.add_argument("--manifest", type=Path, default=Path("benchmarks/dev-holdout-v1.json"))
    benchmark.add_argument(
        "--backend",
        choices=[
            "baseline",
            "header",
            "llm",
            "structured",
            "selective",
            "agreement",
            "hybrid",
            "semantic",
            "program-v3",
            "program-v4",
            "program-v5",
            "risk-controlled-v4",
            "risk-controlled",
        ],
        default="header",
    )
    benchmark.add_argument("--llm-url")
    benchmark.add_argument("--model")
    benchmark.add_argument("--cache-dir", type=Path, default=Path("artifacts/holdout-cache"))
    benchmark.add_argument("--output", type=Path, default=Path("reports/holdout.json"))
    for target in (run, serve, benchmark):
        target.add_argument(
            "--program-policy",
            choices=["semantic", "model", "hybrid", "agreement", "strict_agreement"],
            default="model",
        )
        target.add_argument(
            "--risk-certificate", type=Path, default=Path("reports/v3-risk-certificate.json")
        )
        target.add_argument(
            "--model-path", type=Path, default=Path("local-runtime/qwen3-4b-q4.gguf")
        )
        target.add_argument(
            "--runtime-profile", type=Path, default=Path("benchmarks/runtime-v3.json")
        )
        target.add_argument(
            "--calibration-manifest", type=Path, default=Path("benchmarks/train-v3-400.json")
        )
        target.add_argument("--calibration-data", type=Path, default=Path("data/tatqa_train.json"))
        target.add_argument(
            "--v4-certificate", type=Path, default=Path("reports/v4-temporal-certificate.json")
        )
        target.add_argument(
            "--v4-calibration-manifest", type=Path, default=Path("benchmarks/calibration-v4.json")
        )
    args = parser.parse_args()
    if args.command != "download":
        default_url, default_model = model_defaults(args.backend, args.command)
        args.llm_url = args.llm_url or default_url
        args.model = args.model or default_model
    v3_options = {
        name: getattr(args, name)
        for name in (
            "program_policy",
            "risk_certificate",
            "model_path",
            "runtime_profile",
            "calibration_manifest",
            "calibration_data",
            "v4_certificate",
            "v4_calibration_manifest",
        )
        if hasattr(args, name)
    }

    def configured_planner():
        try:
            return make_planner(
                args.backend,
                args.llm_url,
                args.model,
                getattr(args, "cache_dir", None),
                **v3_options,
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))

    if args.command == "download":
        print(fetch_dataset(args.data_dir, args.split))
    elif args.command == "evaluate":
        planner = configured_planner()
        report = evaluate(load_examples(args.data), planner, args.limit)
        report.update(
            {
                "backend": args.backend,
                **inference_metadata(planner, args.backend),
                "dataset_revision": REVISION,
                "data_receipt": json.loads(args.data.with_suffix(".json.receipt.json").read_text())
                if args.data.with_suffix(".json.receipt.json").exists()
                else None,
            }
        )
        write_json(args.output, report)
        print(
            json.dumps(
                {key: value for key, value in report.items() if key != "predictions"}, indent=2
            )
        )
    elif args.command == "benchmark":
        from .benchmark import evaluate_manifest

        examples = load_examples(args.data)
        manifest = json.loads(args.manifest.read_text())
        planner = configured_planner()
        report = evaluate_manifest(examples, planner, manifest)
        report.update(
            {
                **inference_metadata(planner, args.backend),
                "dataset_revision": REVISION,
                "protocol": {1: "structured-v1", 2: "selective-v2", 3: "program-v3"}[
                    manifest.get("version", 1)
                ],
            }
        )
        write_json(args.output, report)
        print(
            json.dumps(
                {key: value for key, value in report.items() if key != "predictions"}, indent=2
            )
        )
    else:
        import logging

        import uvicorn

        from .api import create_app

        planner = configured_planner()
        logging.basicConfig(level=logging.INFO)
        uvicorn.run(
            create_app(planner),
            host="127.0.0.1",
            port=args.port,
            limit_concurrency=16,
            access_log=False,
        )


if __name__ == "__main__":
    main()
