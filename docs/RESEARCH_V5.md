# Source-exact financial reasoning: v5 experiment

V5 is an experimental accuracy candidate, using the existing local Qwen3-4B
Q4_K_M model. It adds no model downloads or paid services. Completed v3/v4
predictions, source modules, calibration decisions and limitations remain intact.

## Why this change

An analysis of the original 100 reused training-development contexts found that
12 of v4's 28 misses primarily involved operand selection, three involved units,
and two involved period validation. See the
[development analysis](DEVELOPMENT_ERROR_ANALYSIS_V5.md). These are development
findings, not an estimate of performance on unseen financial reports.

The approach follows the separation of evidence selection and symbolic reasoning
described in the original [TAT-QA/TAGOP paper](https://aclanthology.org/2021.acl-long.254/).
That paper's full-dataset F1 scores are not comparable to this project's arithmetic
subset and scale-aware numeric accuracy.

## Inference design

1. A deterministic resolver handles literal, unique metric labels and explicit
   periods. It checks financial qualifiers, numeric metric categories, units,
   currencies and duplicate observations. Ambiguous categories and compound
   questions delegate to the model instead of choosing the nearest label.
2. The model sees the unchanged table layout with variable IDs and a flat source
   catalog containing values, full labels, periods, units and paragraph context.
   Issued source choices include a bounded metric, column, year and unit summary.
   An exact allowlist maps these choices back to the original variable IDs;
   arbitrary model strings cannot become executable references.
   No reference answer, derivation or correctness annotation enters the request.
3. The response contains a bounded explanation and at most four symbolic steps.
   Simple explicit operations have a narrower grammar. Compound questions keep
   intermediate operations available.
4. The existing compiler and Decimal executor validate source references, types,
   currencies, periods and arithmetic. At most two repairs are allowed after local
   schema/compiler/executor rejection. A successfully executed proposal or valid
   abstention never gets extra attempts based on its correctness.

The source catalog adds explicit currency multipliers such as `£m`, documented
common-size percentages, and locally supported narrative counts. It does not
infer arbitrary conversions or missing financial values.

Requests use fixed seeds, temperature 0.2, a maximum of 800 generated tokens, and
non-thinking mode. These are experimental settings, not an assertion of globally
optimal decoding. [Qwen's documentation](https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html)
describes the thinking switch and recommends different sampling settings for
general use. Raw responses are cached against the complete candidate identity,
endpoint and request. Changes to the candidate invalidate checkpoint identity.

## Evaluation discipline

The first complete raw-reference candidate scored 76/100 with 91 answers on the
reused development set, versus v4's 72/100 on the same set. These outcomes are
preserved in `reports/v5-development-raw-reference-candidate.json`. They prompted
a development-only revision using descriptive source references, source-declared
footnote normalization and a distinction between a named weighted-average metric
and an outer arithmetic average over its annual observations. The earlier rule
diagnostic (39/40 supported) is version-bound and is not the new policy's score.

The complete descriptive-reference revision scored **83/100**, with **95 answered**
and **87.4% accuracy among answers**, versus v4's 72/100 on the same reused set.
Its report is `reports/v5-development-descriptive-candidate.json`. This prompted
one additional source-based compiler correction: explicitly supported named-month
or beginning/end comparisons within the requested year are now distinguished
from ordinary comparisons against a prior year. The exception requires matching
metric/category labels on one table axis, explicit periods, known dimensions and
unchanged unit/currency validation. It does not allow arbitrary same-year cells.
The final development report must bind this corrected candidate and hardened
prospective runner before fresh membership can be reserved.

The final source-bound development pass completed with **83/100 correct**,
**95/100 answered**, and **87.4% accuracy among answers**. The complete prospective
policy and its 150-context membership are now frozen. This development result is
not a substitute for the frozen comparison result.

The fresh comparison is complete: v5 99/150 (66.0%) versus v4 97/150 (64.7%),
paired gain +1.33 points [-4.67, +6.67]. Both predeclared targets failed. The full
[results report](RESULTS_V5.md) includes coverage, secondary official scoring and
latency. These inspected source groups cannot support another untouched test.

Development source snapshots are preserved in
`benchmarks/development-snapshots-v5.json`, including source text, SHA-256 hashes
and report bindings. Regression tests verify both historical revisions. A later
interrupted pass exposed an invalid fallback from explicit subannual requests to
prior-year operands. The final policy blocks that fallback in both the exact
resolver and the compiler; unsupported quarters and numeric-date comparisons
remain rejected. Interrupted checkpoints are not accuracy reports.

The original 100 development contexts are reused for engineering. Development
results must be labeled as such. Before any fresh comparison is scored, freeze
the candidate, model/runtime, scorer, environment and membership:

The following sequence documents how this experiment was executed. Completed
reports and frozen membership are checked in and deliberately refuse replacement.
For the existing study, use read-only verification:

```sh
.venv/bin/python - <<'PY'
import importlib.util
from pathlib import Path
path = Path("scripts/experiment_v5.py")
spec = importlib.util.spec_from_file_location("study", path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
policy, rows = study.verify_freeze()
print(policy["scope"], len(rows))
PY
```

Data and the recorded environment must already be available. A fresh live replay
belongs in a separate copy retaining the frozen policy and consumption ledger,
with completed output files archived before running the two comparison commands.
Do not run `freeze` again or reset reservations to reuse inspected sources.
Such a replay is reproduction of a consumed comparison, not a new holdout study.
The original execution sequence was:

```sh
# Run from the repository root using the verified local runtime/model.
.venv/bin/python scripts/serve_qa_model.py
# In a second terminal:
.venv/bin/python scripts/experiment_v5.py development
.venv/bin/python scripts/experiment_v5.py freeze
.venv/bin/python scripts/experiment_v5.py comparison --method baseline
.venv/bin/python scripts/experiment_v5.py comparison --method candidate
.venv/bin/python scripts/experiment_v5.py summarize
.venv/bin/python scripts/score_v5_official.py
# Optional standalone figure; see requirements-analysis.txt.
.venv/bin/python scripts/plot_v5.py
```

The prospective comparison contains 150 public-training contexts, excluding
previously used exact source groups through an append-only reservation ledger.
Every consumed source is registered before outcomes are used. The same examples
are evaluated by frozen v4 and v5. The primary metric counts abstentions as
incorrect; coverage and accuracy among answered questions are separate metrics.
Reports verify predictions and aggregates against source labels before summary.

The prospective engineering target is **80% overall numeric accuracy**, with a
paired context-bootstrap 95% interval for improvement strictly above zero. This
is a project target, not a universal industry threshold or a performance promise.
No inference changes or sample reselection are allowed after fresh outcomes.
The final summary records whether each target was actually met.

## Application and tests

```sh
.venv/bin/financial-qa serve --backend program-v5 --port 8001
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

V5 is opt-in and reports `experimental_v5_not_risk_calibrated` in metadata. It
does not inherit a successful accepted-error certificate. Software tests check
synthetic correctness and failure behavior; only the completed model evaluation
can support an accuracy claim.

Source-group exclusion reduces exact source reuse. It does not establish
independent filings, remove public-data pretraining contamination, or establish
performance on PDFs, arbitrary reports, future filings or production traffic.

The final candidate also supports typed inverse-percentage arithmetic for a
single aligned pair of source observations. Dividing an amount by a declared
percentage share uses `amount / (share / 100)` and preserves the numerator's unit.
Unknown dimensions, mismatched periods and unsupported subannual pairs do not
qualify for this extension. The ordinary same-dimension division rules remain
unchanged. A ratio of metrics named "organic change" keeps ratio operators
available rather than being forced into a temporal percentage-change operator.
