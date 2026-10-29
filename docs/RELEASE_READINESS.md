# Release scope and verification

This repository is a standalone local research application. It contains source,
offline tests, browser assets, experiment protocols and historical evaluation
reports. Data downloads, weights, runtimes, response caches and build outputs
are deliberately excluded. The code is MIT licensed; preserve the separate
[TAT-QA and model attribution](../DATA.md).

## Verify a checkout

Run from the repository root with Python 3.11 or newer:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m pytest -q
.venv/bin/python -m pip wheel --no-deps . -w dist
.venv/bin/financial-qa serve --port 8001
```

The default deterministic backend and offline tests do not require weights,
a model endpoint, paid services or a dataset download. Download the pinned data
before loading real benchmark contexts through the browser. For optional model
reasoning, install the independently verified local runtime and Qwen3-4B weights
as described in [the runtime guide](RESEARCH_V3.md#reproduce-on-a-laptop), then use
`scripts/serve_qa_model.py`. The launcher verifies model size and SHA-256 and keeps
inference on loopback. Use `--dry-run` to inspect the command without loading data.

CI checks installation, style, offline tests and wheel construction on Linux with
Python 3.11 and 3.12. A successful local check does not establish that a remote CI
run has passed; inspect the actual GitHub Actions run after publishing.

## Accuracy qualification

The completed v5 comparison measured 99/150 correct (66.0%) versus v4's 97/150
(64.7%). V5 answered 133 questions, with 74.4% accuracy among those answers. Its
paired gain was inconclusive and the declared 80% overall target was missed.
See [the complete evidence](RESULTS_V5.md) for uncertainty, latency and scoring
scope. Passing software checks is separate from meeting a scientific target.
This release is suitable for demonstrating research and engineering methods; it
does not claim production financial reliability or independent certification.

Historical reports and frozen inference sources are preserved. Reproducing a
consumed study does not create an untouched holdout. Do not overwrite reports,
reset reservations, or tune on their errors and present the same comparison as
new evidence.
