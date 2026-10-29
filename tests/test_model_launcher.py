"""The standalone launcher rejects corrupt weights and supports offline inspection."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "standalone_model_launcher", Path(__file__).parents[1] / "scripts/serve_qa_model.py"
)
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


def test_model_launcher_rejects_same_size_corruption(tmp_path, monkeypatch):
    model = tmp_path / "weights.gguf"
    original = b"verified fixture"
    model.write_bytes(original)
    monkeypatch.setattr(LAUNCHER, "MODEL_BYTES", len(original))
    monkeypatch.setattr(LAUNCHER, "MODEL_SHA256", hashlib.sha256(original).hexdigest())
    LAUNCHER.verify_model(model)
    model.write_bytes(b"modified fixture")
    with pytest.raises(ValueError, match="SHA-256"):
        LAUNCHER.verify_model(model)


def test_dry_run_uses_repository_paths_without_loading_weights(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["serve_qa_model.py", "--dry-run", "--gpu-layers", "0"])
    monkeypatch.setattr(
        LAUNCHER,
        "verify_model",
        lambda _: pytest.fail("A dry run must not load model bytes"),
    )
    LAUNCHER.main()
    result = json.loads(capsys.readouterr().out)
    assert result["downloads"] is False
    command = result["command"]
    project = Path(__file__).resolve().parents[1]
    assert command[0] == str(project / "local-runtime/llama-b11669/llama-server")
    assert command[command.index("-m") + 1] == str(project / "local-runtime/qwen3-4b-q4.gguf")
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--parallel") + 1] == "1"
    assert "--no-webui" in command
    assert "--no-cors-credentials" in command
