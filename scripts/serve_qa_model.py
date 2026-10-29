"""Start the verified optional QA model without downloading weights or a runtime.

The foreground process stays on loopback and uses one bounded inference slot.
Run the financial-qa application separately; this endpoint only proposes programs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"
MODEL_BYTES = 2_497_280_256
ALIAS = "qwen3-4b-q4-nonthinking"


def verify_model(path: Path) -> None:
    """Stream a multi-gigabyte file instead of reading another copy into memory."""
    if not path.is_file() or path.stat().st_size != MODEL_BYTES:
        raise ValueError("Expected the pinned Qwen3-4B Q4_K_M model; see the QA README")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    if hasher.hexdigest() != MODEL_SHA256:
        raise ValueError("Model SHA-256 mismatch; refusing to start inference")


def command(runtime: Path, model: Path, port: int, gpu_layers: int) -> list[str]:
    """Keep inference settings aligned with the frozen research runtime profile."""
    return [
        str(runtime),
        "-m",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        "8192",
        "-t",
        "4",
        "-tb",
        "6",
        "-ngl",
        str(gpu_layers),
        "--parallel",
        "1",
        "--cache-ram",
        "0",
        "--alias",
        ALIAS,
        "--reasoning",
        "off",
        "--cors-origins",
        "localhost",
        "--no-cors-credentials",
        "--no-webui",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        type=Path,
        default=PROJECT / "local-runtime/llama-b11669/llama-server",
    )
    parser.add_argument("--model", type=Path, default=PROJECT / "local-runtime/qwen3-4b-q4.gguf")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument(
        "--gpu-layers", type=int, default=99 if platform.system() == "Darwin" else 0
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 0 <= args.gpu_layers <= 999:
        parser.error("Port must be 1–65535; GPU layers must be 0–999")
    runtime, model = args.runtime.resolve(), args.model.resolve()
    invocation = command(runtime, model, args.port, args.gpu_layers)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "command": invocation,
                    "model_sha256": MODEL_SHA256,
                    "downloads": False,
                },
                indent=2,
            )
        )
        return
    if not runtime.is_file() or not os.access(runtime, os.X_OK):
        parser.error("Install the trusted llama.cpp b11669 runtime first; see the QA README")
    try:
        verify_model(model)
        # Fail before loading weights if another process owns the desired endpoint.
        with socket.socket() as probe:
            # A recently stopped server can leave TIME_WAIT connections. Match
            # normal server rebinding without permitting a second live listener.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", args.port))
        version = subprocess.run(
            [str(runtime), "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        if "11669" not in version.stdout + version.stderr:
            raise ValueError("This reproduction requires the recorded llama.cpp b11669 runtime")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    # exec keeps Ctrl+C and exit status attached to the foreground server, and
    # avoids a detached process that could keep model memory after the task ends.
    os.chdir(PROJECT)
    os.execv(str(runtime), invocation)


if __name__ == "__main__":
    main()
