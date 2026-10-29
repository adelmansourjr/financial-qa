# Contributing

Use Python 3.11 or newer. Run all commands from this repository directory.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m pytest -q
```

Tests are offline and use synthetic documents and mocked model responses. New tests
should cover a real failure boundary: financial scope, units, ambiguous evidence,
unsafe expressions, request validation, cache identity, or artifact provenance. A
schema-valid response is not itself an accuracy test.

## Working on reasoning

Read [the architecture](docs/ARCHITECTURE.md) and the experiment protocols before
changing inference. The planner receives a `Document`, question and `Evidence`,
never an `Example`, gold answer or derivation. Keep this boundary intact.

Use training documents to diagnose issues. Keep separate document groups for
training development, policy selection, and final evaluation. Freeze the final
manifest before scoring and exclude previously inspected documents. Once results
are visible, further changes require another declared evaluation, rather than
silently reusing the same set as an untouched holdout. Public financial benchmarks
may have appeared in model pretraining even when local document splits are clean.

Preserve historical reports. Put experiment outputs under new filenames. Retain
abstentions in the denominator; report unconditional accuracy, answer coverage,
accuracy among accepted answers, latency and uncertainty together. Document the
local scale-aware scorer separately from official benchmark metrics.

## Interface changes

The CLI and API share `service.answer`. Add a new backend through `cli.make_planner`
and make its source/operation limitations visible in documentation. Keep browser
assets as readable HTML, JavaScript and CSS. Serve them under `/static` and include
all three extensions in wheel package data. Do not expose answer annotations through
the real-context API.

## Packaging and data

```sh
.venv/bin/python -m pip wheel --no-deps --no-build-isolation . -w /tmp/financial-qa-wheel
```

Inspect the wheel after asset or packaging changes. It should include Python source
and browser assets, and exclude datasets, model weights, raw caches and local build
artifacts. Use the pinned dataset download command and its hash receipt; keep
financial data/model licensing separate from the MIT code license in `DATA.md`.
