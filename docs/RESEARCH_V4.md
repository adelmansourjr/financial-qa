# Layout-aware financial programs

Version four is an explicit experimental backend. It preserves the completed
version-three experiment, prediction components, results and failed calibration
certificate. No old reliability claim transfers to this new planner.

## Why this changes the reasoning path

The [TAT-QA paper](https://aclanthology.org/2021.acl-long.254/) separates evidence
selection from symbolic operations over tables and paragraphs. The
[financial reasoning study](https://aclanthology.org/2024.findings-acl.231/)
also examines the importance of table complexity and intermediate reasoning.
Those findings motivate testing original-layout inputs and a brief evidence
analysis before constrained program generation. This implementation is an
engineering experiment, not a reproduction of either paper's scores.

Training-development inspection identified concrete v3 defects: a percentage
elsewhere in a column could label an employee count as a percentage; labeled year
headers could lose chronology; and a keyword such as `average`, `change`, or
`total` could force an inappropriate operation. A weighted-average metric name
does not necessarily ask for an arithmetic average. A ratio can require sums
on both sides rather than one forced binary operation.

The new grounding module preserves original source IDs and Decimal values while
describing units and observation periods separately from years in entity names.
Explicit local percentage/currency/count/duration metadata replaces cross-row
percentage guessing. Unknown units and paragraph periods remain unresolved.
The model sees the original table with numeric source IDs and grouped source
units; it does not see gold answers, derivations or annotations.

The response contains a short source analysis followed by up to four symbolic
steps. Explicit simple requests constrain one operation and its arity; compound
requests retain multi-step programs. Modifier names such as weighted-average are
distinguished from instructions to average, and a percentage share of a change
is distinguished from percentage change. The compiler retains source-reference, currency,
dimension, expression-size and bounded-Decimal execution checks. It does not
reorder category comparisons. Explicit temporal changes with unambiguous observation
periods use newer minus older. Generated prose and syntactically valid
programs do not establish that the selected evidence is financially correct.

## Measurement protocol

Development uses the already inspected 100 v3 training-development contexts.
Interrupted pilots are retained and labeled as development only. Before scoring
new contexts, the final prompt, model/runtime profile, prediction code, scoring
criterion and runner are frozen.

The comparison selects 200 previously unused public-training contexts, one
eligible arithmetic question per context, with fixed seed 409. All 608 historical
training contexts are excluded as whole contexts, including old calibration and
development samples. The old v3 model and new layout model are evaluated on the
same 200 questions. The primary score includes wrong answers and abstentions in
the denominator. A paired context bootstrap reports uncertainty in the accuracy
difference. Policies or prompts cannot change after these outcomes are scored.

This is a project-held-out public-training arithmetic study. It is not an
official hidden-test score, a new untouched public-development test, or a claim
that the model never encountered the data in pretraining. Context identifiers do
not establish independent source filings. The experiment does not create a new
90% reliability certificate or promote a default based on final outcomes.

## Reproduce

Use the existing documented hash-verified local model launcher. From this project:

```sh
.venv/bin/python scripts/experiment_v4.py development --method layout
.venv/bin/python scripts/experiment_v4.py freeze
.venv/bin/python scripts/experiment_v4.py heldout --method baseline
.venv/bin/python scripts/experiment_v4.py heldout --method layout
.venv/bin/python scripts/experiment_v4.py summarize
```

Completed reports refuse replacement; interrupted stages resume only with the
same source and membership. Model weights are streamed for hash verification at
run start, and trusted local runtime path/alias metadata is checked periodically.
These integrity checks do not attest an adversarial model server. Raw response
caches stay ignored and may contain source data. A resumed/replayed request's
latency must not be called fresh model generation latency.

Serve the new backend explicitly:

```sh
.venv/bin/financial-qa serve --backend program-v4 --port 8001
```

The default semantic backend and v3 risk-controlled mode retain their meanings.
Version four exposes `experimental_v4_not_risk_calibrated` in health and answer
metadata. It requires the existing runtime profile and local model server.

## Separate narrow-policy calibration

A separate acceptance policy was developed only on the original 100 development
contexts, before reading any new comparison or calibration errors. It accepts
explicit temporal changes with two known observation years and two same-row
money/count sources, only when the independent semantic rule and the layout
model agree numerically and in output scale. Development accepted 30 contexts,
29 correct; this feasibility result is not a certificate.

Exactly one policy is frozen for a fresh 300-context calibration sample, with
seed 419, excluding both the 608 historical training contexts and the 200 new
comparison contexts. Its one-sided exact Clopper-Pearson upper accepted-error
bound must be at most 10% at 95% confidence, with at least 20 accepted examples.
The same error target/confidence is retained; unlike the earlier five-policy
grid, this experiment tests one training-developed policy without selection.
It may qualify at narrower coverage or fail. Neither outcome permits retuning
on these 300 labels. Related filings and public-data pretraining exposure remain
limitations to the interpretation of a reference-distribution risk bound.

```sh
.venv/bin/python scripts/calibrate_v4.py freeze
.venv/bin/python scripts/calibrate_v4.py run
financial-qa serve --backend risk-controlled-v4 --port 8001
```

The new backend uses its own certificate and calibration manifest. It starts
unavailable and abstains when metadata, model bytes/runtime identity, calibration
membership, policy code, environment, or the numerical risk target is invalid.
Certificate loading recomputes counts and the bound from strict boolean evidence;
merely changing counts and recalculating a checksum cannot make a policy qualify.
Checksums still assume trusted local evidence rather than attest an adversarial
server. Serving permanently revokes availability after a detected runtime drift.
The original `risk-controlled` backend remains tied to the unchanged v3 result.

Before the first calibration request, a checkpoint-only preflight repair increased
the bounded 300-row checkpoint budget from 100 KB to 256 KB. The original freeze and
repair hashes are preserved in `reports/v4-calibration-initial-freeze.json` and
`reports/v4-calibration-preflight-repair.json`. The acceptance rule and sampled
contexts did not change, and there were zero calibration outcomes before repair.

The content-overlap audit compares lowercased, whitespace-normalized tables and
paragraphs while retaining numeric content. It found no exact content duplicates
within or between the 608 historical contexts, 200 comparison contexts and 300 fresh
calibration contexts. This strengthens the overlap check beyond IDs, but does not
establish independent companies/filings or exclude public-data pretraining exposure.
See [the audit](../reports/v4-content-overlap-audit.json).

A second preflight optimization, also before any calibration requests, skips
generation when the fixed question-type or two-year requirement already fails.
The full acceptance rule, contexts and target remain identical; 160 of the
300 sampled questions are eligible for generation. The audit is preserved in
`reports/v4-calibration-prefilter-audit.json`. Skipped questions count as abstentions
in the full 300-context denominator rather than disappearing from evaluation.
