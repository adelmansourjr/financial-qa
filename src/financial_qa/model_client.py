"""Bounded loopback-only HTTP transport for private document inference."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .executor import ExecutionError

MAX_RESPONSE_BYTES = 100_000


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward source documents to a redirect target, including localhost."""

    def redirect_request(self, request, file, code, message, headers, new_url):
        raise ExecutionError("Model endpoint redirects are forbidden")


def validate_endpoint(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/v1"
    ):
        raise ValueError(
            "Model endpoint requires credential-free loopback HTTP /v1 without query/fragment"
        )
    try:
        if parsed.port is None or not 1 <= parsed.port <= 65535:
            raise ValueError("Model endpoint requires an explicit valid port")
    except ValueError as error:
        raise ValueError("Model endpoint requires an explicit valid port") from error
    return base_url.rstrip("/")


def read_bounded_json(path: Path) -> dict:
    if path.stat().st_size > MAX_RESPONSE_BYTES:
        raise ExecutionError("Model cache exceeds the response size budget")
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("Expected an object")
        return value
    except (ValueError, UnicodeError) as error:
        raise ExecutionError("Model cache is malformed") from error


class SafeModelClient:
    """Direct loopback transport with no proxy, redirect, or unbounded response."""

    def __init__(self, base_url: str):
        self.base_url = validate_endpoint(base_url)
        self.api_key = os.environ.get(
            "FINANCIAL_QA_LLM_API_KEY", os.environ.get("LLAMA_API_KEY", "")
        )
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, body: bytes) -> bytes:
        if len(body) > 90_000:
            raise ExecutionError("Model request exceeds the context size budget")
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": "Bearer " + self.api_key} if self.api_key else {}),
            },
            method="POST",
        )
        try:
            with self.opener.open(request, timeout=90) as response:
                if response.status != 200:
                    raise ExecutionError("Model server returned a non-success status")
                payload = response.read(MAX_RESPONSE_BYTES + 1)
                if len(payload) > MAX_RESPONSE_BYTES:
                    raise ExecutionError("Model response exceeds the response size budget")
                return payload
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ExecutionError("Local model request failed or timed out") from error

    def properties(self) -> dict:
        """Read trusted-launch metadata without transmitting a document."""
        url = self.base_url.removesuffix("/v1") + "/props"
        try:
            request = urllib.request.Request(
                url, headers={"Authorization": "Bearer " + self.api_key} if self.api_key else {}
            )
            with self.opener.open(request, timeout=5) as response:
                payload = response.read(MAX_RESPONSE_BYTES + 1)
                if len(payload) > MAX_RESPONSE_BYTES:
                    raise ValueError("Model metadata response exceeds the size budget")
                return json.loads(payload)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            raise ValueError("Cannot verify local model server metadata") from error
