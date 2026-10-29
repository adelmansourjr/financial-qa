# Local serving and CLI contracts

The default `serve` command uses deterministic semantic rules and needs no model,
runtime profile, dataset download or external service. If the pinned development
file is absent, the browser's real-example catalog is empty; synthetic/manual
contexts and the API still work. Installed wheels include the HTML, CSS and JavaScript.

Optional model modes need a separately running compatible local server and a runtime
profile file. Runtime profiles, manifests, datasets and weights are repository/local
artifacts rather than wheel package data. Pass their explicit paths when launching
outside the repository directory.

## Backend configuration

Without explicit overrides, `program-v3`, `program-v4`, `risk-controlled` and `risk-controlled-v4` use
`http://127.0.0.1:18081/v1` and alias `qwen3-4b-q4-nonthinking` for every CLI command.
Historical benchmark backends retain their 1.5B/18080 defaults; historical serving
and evaluation backends retain their 8080 defaults. Explicit `--llm-url` and `--model`
values are preserved. The safe v3 client rejects non-loopback endpoints and never
follows redirects or sends documents through an environment proxy.

`evaluate` and `benchmark` report the actual configured v3 generation profile,
including model weight identity and token budget. The generic benchmark command
accepts v1/v2/v3 manifests while checking unique question/context membership and
exclusion sets. Use a new output path when replaying a historical experiment;
a replay is not a new untouched evaluation or fresh inference-latency measurement.

Runtime signatures are bounded to 16 KiB and must contain exactly the build string,
positive integer context size, two SHA-256 hashes and reasoning-format string.
Booleans, incomplete profiles and extra fields are rejected.

Missing or malformed runtime configuration leaves `risk-controlled` mode unavailable
with state `configuration_unavailable_fail_closed`. Its API starts, health explains
that state, and answers abstain. Experimental `program-v3` requires a readable runtime
profile; CLI configuration errors produce a usage error instead of starting an
incorrectly configured server. The checked-in valid calibration artifact separately
fails its risk target, so it also keeps guarded mode unavailable.

## Admission and failures

Each app process admits one `/answer` inference at a time. Overlapping requests receive
HTTP 503 with `Retry-After: 1`; they are not queued into the single-slot local model
server. `/health`, the example catalog and static assets remain responsive. The slot
is released after both a successful response and a planner exception. This admission
rule protects the shared planner's fail-closed state from concurrent updates.

An accepted request still has source/context/model response budgets and transport
timeouts. Budget and model-protocol failures abstain. Unexpected server exceptions
use the shared HTTP guard's generic correlated error response. Request IDs identify
failures without logging document contents or raw query strings. See `/docs` for the
HTTP schema and the research guide for the model/runtime trust limitations.

These controls support a local research app. They do not provide authentication,
multiple-worker coordination or a production public-service deployment architecture.

`program-v4` is a separate layout-aware experimental backend. It accepts only the
model policy, reports its own prompt/code/generation identity, and does not use or
inherit the v3 risk certificate. Its health/answer state is
`experimental_v4_not_risk_calibrated`. See [the v4 guide](RESEARCH_V4.md).

`risk-controlled-v4` uses `--v4-certificate` and
`--v4-calibration-manifest` for the separate 300-context temporal-agreement
experiment. It never interprets the v3 certificate as v4 evidence. The same
`--calibration-data` and `--model-path` flags bind source data and local model
bytes. Missing runtime or certificate files leave the app explicitly unavailable.
A qualifying artifact, if produced, applies only to the stated reference-context
assumptions and narrower acceptance policy.

The completed v4 certificate accepts 95/300 calibration contexts, with 5 errors
and a 10.75% upper accepted-error bound. This exceeds the fixed 10% target, so
its actual serving state is `uncertified_fail_closed`. The service starts and
explains the failed calibration; every answer abstains without invoking the model.
See [the results and real browser check](RESULTS_V4.md).
