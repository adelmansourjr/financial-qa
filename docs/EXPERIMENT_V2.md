# Selective financial reasoning experiment

## Question and method

Can explicit financial scope and a conservative rule/model cascade improve a small,
laptop-scale numerical QA system? The earlier model's constrained JSON was reliable,
but metric selection, temporal orientation, units and implicit averages were not.
Version 2 adds explicit abstention and evaluates those failure boundaries rather than
assuming that better formatting establishes better reasoning.

The semantic rule planner matches financial labels and explicitly requested periods,
including years in rows, multirow headers and repeated year/category columns. It
supports scoped differences, percent changes, means and sums. It declines ambiguous
selections, unsupported ratios, missing observations and unresolved operand sets.
Source units are inferred with per-share exceptions and explicit percentage columns;
the model cannot invent a multiplier for an unknown source unit.

The selective model proposes an operation and existing operand IDs under a dynamic
JSON schema. The application compiles expressions and citations. Explicit question
operators and chronological two-period change orientation are normalized without gold
answers. Scope checks reject repeated operands, incompatible source scales, and
unspecified years. An average in 2019 does not automatically include 2018.

The hybrid uses semantic rules first, then the selective model after rule abstention.
The agreement policy accepts only proposals whose value and unit match the rule.
Both policies can share correlated mistakes. No model weights were fine-tuned.

## Separation and frozen choices

- Development: 30 seeded training documents (`benchmarks/train-v2.json`, seed 113).
- Policy selection: 40 different training documents from that manifest.
- Final: 100 seeded public-development documents, one numeric arithmetic question each
  (`benchmarks/dev-holdout-v2.json`, seed 211).
- All 91 previously explored development documents are excluded: the original 41
  documents, the prior 50-document evaluation, and the UI catalog (already in the 41).
  There were 186 eligible uninspected documents before sampling 100.

The final manifest was created before version 2 final scoring. Initial development
outputs remain in `reports/v2-development-*-initial.json`. Training diagnostics led
to fixes for monetary unit/year headers, weighted-average metric names, reversed
temporal operands, ambiguous labels, percentage columns and missing exact metrics.

An independent edge-case audit found that a body value like Revenue=2000 could be
mistaken for a year header. This was fixed after early deterministic validation
summaries were visible; validation candidates were then rerun. The 40-document
training validation set is therefore a policy selection set, not an untouched test.
No prediction code, prompts or policy settings were tuned after final 100 scoring.

`benchmarks/policy-v2.json` records the selection rule. Candidates needed at least 70%
accepted accuracy and 20% coverage on training validation, then were ranked by
unconditional accuracy. `reports/v2-policy-selection.json` freezes the resulting
choice and prediction-code hashes before final scoring.

| Training validation policy | Correct / 40 | Coverage | Accepted accuracy |
| --- | ---: | ---: | ---: |
| Semantic |24|67.5%|88.9%|
| Selective model |20|67.5%|74.1%|
| Hybrid |30|90%|83.3%|
| Agreement |14|40%|87.5%|

Hybrid was selected for explicit model-assisted use. The interactive default remains
semantic because the launcher must run without a model server. Its quality and
coverage passed the same validation thresholds. Historical header-aware rules scored
17/40 on this validation set and remain available as a preserved reference.

## Metrics and uncertainty

The primary scorer compares numeric results in the gold unit, using a 0.005 tolerance
in that unit. It is not official TAT-QA exact match/F1. Unconditional accuracy retains
abstentions in the denominator. Accepted accuracy is conditional on answering and must
be read alongside coverage. Error labels distinguish abstention, wrong source unit
with otherwise matching numeric value, and wrong value/operation.

Reports include 1,000-replicate document-bootstrap 95% intervals for overall accuracy,
coverage and accepted accuracy. Paired comparisons resample the same documents for
both methods. There is one question per document in this evaluation. These are small
sample intervals, not guarantees about a future financial reporting distribution.

## Reproduction

The existing macOS arm64 runtime and 1.5B quantized weights occupy about 1.6 GB with the
historical 0.5B comparison model. No new weights were downloaded for this experiment.
Use the pinned runtime/model instructions in the README and start the server with
context 8,192, two CPU threads, one parallel slot and alias `qwen2.5-1.5b-q4`.

```sh
.venv/bin/python scripts/experiment_v2.py validation --backend semantic \
  --output reports/replay-validation-semantic.json
.venv/bin/python scripts/experiment_v2.py final --backend semantic \
  --output reports/replay-final-semantic.json
.venv/bin/python scripts/experiment_v2.py final --backend selective \
  --output reports/replay-final-selective.json
.venv/bin/python scripts/experiment_v2.py final --backend hybrid \
  --output reports/replay-final-hybrid.json
```

The helper refuses to overwrite measured reports. Model requests are cached under
ignored `artifacts/v2-{stage}-cache/`, keyed by endpoint and complete request body.
Agreement and hybrid reuse exactly the same selective proposals for comparison.
A response replay measures cache/software latency, not fresh inference latency.
The original live model report must be retained for CPU latency claims.

The ordinary CLI also accepts `--backend semantic`, `selective`, `hybrid`, or
`agreement`, with `--manifest benchmarks/dev-holdout-v2.json` on `benchmark`.
Serve model-assisted mode explicitly:

```sh
.venv/bin/financial-qa serve --backend hybrid \
  --llm-url http://127.0.0.1:18080/v1 --model qwen2.5-1.5b-q4 --port 8001
```

## Completed frozen evaluation

| Policy | Correct/all100 | Answered | Correct among answered |
| --- | ---: | ---: | ---: |
| Original baseline |7|18|38.9%|
| Historical header |26|45|57.8%|
| Semantic default |51|59|86.4%|
| Selective model |42|63|66.7%|
| Hybrid, selected before evaluation |59|83|71.1%|
| Agreement |33|34|97.1%|

The semantic rule improvement over the historical header is +25 percentage points,
paired document interval [+15, +35]. The hybrid improvement over semantic is
+8 points [+3, +13], with lower accepted-answer accuracy. Agreement has one
accepted error but rejects 66 questions; its accepted-answer interval is
91.2–100%, conditional on the accepted sample. These descriptive intervals do
not adjust for multiple comparisons. No setting or inference code was changed
after evaluation; `v2-summary.json` verifies the frozen source hashes.

All 100 selective proposals were generated by the existing local 1.5B model on
CPU and cached. Its p95 live latency is 19.54 s. Hybrid/agreement were then scored
from those identical cached proposals; their replay latency is not an estimate
of live model service latency. The hybrid first runs the deterministic method
and calls the model only after abstention when served live.

The independently run, unmodified official TAT-QA scorer gives identical EM/F1
for these numeric arithmetic predictions. It is a secondary metric check after
policy selection and cannot convert this public subset into a full leaderboard
or uncontaminated final-test result. To reproduce after the saved predictions:

```sh
.venv/bin/python -m pip install -r requirements-evaluation.txt
.venv/bin/python scripts/official_score.py \
  reports/v2-final-baseline.json reports/v2-final-header.json \
  reports/v2-final-semantic.json reports/v2-final-selective.json \
  reports/v2-final-hybrid.json reports/v2-final-agreement.json \
  --download-scorer --output reports/replay-official-secondary.json
```

The scorer adapter verifies the known source hashes before loading the upstream
files; raw code stays in ignored artifacts. These optional dependencies are not
needed for inference or the offline package tests. Original reports are preserved.

## Limits

This is a public benchmark subset and model pretraining contamination cannot be ruled
out. Selection covers numeric arithmetic questions, not all TAT-QA question types or
arbitrary filings. The model has 1.5B parameters and a small fixed operation vocabulary;
compound financial formulas, missing source observations and large contexts can be
unsupported. It has no filing retrieval, OCR or production document parsing pipeline.

Unit inference is heuristic. An unlabeled mixed-measure column may still be interpreted
incorrectly. Consistent inferred scales and valid citations cannot establish that the
underlying source interpretation is true. Exact metric matching can miss paraphrases;
lexical matches can select the wrong financial concept. A returned answer remains a
research result requiring source review.
