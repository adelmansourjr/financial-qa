# Source-grounded programs and selective risk control

This experiment combines a small local language model, a bounded symbolic program,
and an separate acceptance calibration. Its purpose is to improve useful
financial arithmetic while measuring how often accepted answers are wrong.

The completed experiment scored 54/86 correct for the preselected model policy. None
of the five policies met the separate 400-context accepted-error target; guarded
mode fails closed. See [the measured results](RESULTS_V3.md).

## Research basis and implementation choices

[Program of Thoughts](https://arxiv.org/abs/2211.12588) separates numerical reasoning
from external computation and evaluates financial QA among other tasks. Our
implementation follows that separation but uses a small JSON operation language
instead of executing generated Python. The model chooses source identifiers;
application code constructs and validates the expression.

[TAT-QA and TAGOP](https://aclanthology.org/2021.acl-long.254/) describe hybrid financial
table/text evidence and symbolic aggregation over selected values. Their work motivates
preserving complete source labels, units and table structure rather than assuming
that correct arithmetic resolves evidence selection. Our arithmetic-only evaluation
is a subset and is not comparable to a full TAT-QA leaderboard result.

[Learn then Test](https://arxiv.org/abs/2110.01052) frames risk calibration as multiple
hypothesis testing. We implement a finite-policy special case: independent document
calibration, exact one-sided binomial bounds on accepted-answer error, and Bonferroni
correction before choosing a policy. This does not claim to reproduce every method
in that paper or provide a guarantee under financial distribution shift.

## Program language

A program has up to four sequential steps. Each step selects a supported operation
and previously available source variables or earlier step outputs. The supported
operators are lookup, sum, average, difference, divide, multiply by an explicit
percentage, percentage ratio, and percentage change. Intermediate results make
ratios of sums possible. Source IDs, operation names and step count are constrained
in the generation schema. Simple questions with an explicit operator use a
single-step, operator-specific grammar and a fixed operand arity.

The compiler rejects missing/forward references, repeated or double-counted
aggregation operands, incompatible dimensions, mixed currencies, dates used as
amounts, omitted explicitly requested years, and excessive expression size. A
percentage multiplier is divided by 100. It never invents an exchange rate or a
monetary scale for an unknown source unit. Arithmetic still runs through the
existing bounded Decimal AST interpreter.

A financial label can remain ambiguous even after these checks. The model can
select the wrong row or a semantically inappropriate subtotal. Program validity
therefore remains separate from numerical answer correctness.

## Development and calibration protocol

The expanded manifest `benchmarks/train-v3-400.json` contains 100 development and 400
calibration documents, with one seeded eligible arithmetic question per document.
It excludes all 108 previously used training documents. There were 2,078 unused
eligible training documents before sampling. The earlier 200-calibration manifest
is retained; the expansion occurred before any calibration labels were viewed.

The final manifest `benchmarks/dev-holdout-v3.json` contains all 86 remaining eligible
previously uninspected development documents, excluding 191 used by earlier work.
The final evaluation is frozen before scoring. Its labels cannot select policies,
change the prompt, tune units or revise acceptance gates.

Two interrupted training-only pilots are preserved explicitly. The first omitted
`min_p`, so the server's 0.05 default applied despite an intended 0.0 profile. The
second diagnosed unnecessary operation freedom. Neither pilot is calibration or
final evidence. The final request profile explicitly sets its sampling parameters,
seed, nonthinking template request and token budget.

## Finite-sample acceptance gate

Five policies are fixed before calibration: semantic rules, model program, rule-first
hybrid, rule/model agreement, and agreement restricted to table operands with known
dimensions. Let `m` be the number of accepted calibration documents and `e` their
incorrect answers. For each policy, we compute the one-sided Clopper-Pearson upper
bound on its conditional error probability at tail probability `0.05/5 = 0.01`.

A policy qualifies only when the upper bound is at most 10% and at least 20 documents
were accepted. The selected qualifying policy maximizes calibration coverage. If none
qualify, risk-controlled mode returns no numerical answers. Perfect performance on
a tiny pilot cannot bypass the bound: with zero errors, the upper bound is
`1 - 0.01^(1/m)`, so sufficient accepted evidence is still necessary.

The simultaneous 95% confidence interpretation requires fixed policies and IID
calibration/future documents drawn from the same distribution, conditional on the
fixed development process. It does not establish that each individual answer is 90%
likely correct. It does not apply under distribution shift, correlated document
sampling, undisclosed policy search, benchmark contamination, or model/runtime drift.

The certificate stores strict boolean outcomes and document/question identities.
Loading recomputes counts, bounds and policy selection from evidence. All 400 frozen
calibration identities must match; a truncated or selected-easy subset cannot qualify.
Model bytes, inference source, prompt, sampling settings, calibration data/manifest,
runtime build, context, chat template and default samplers are hash bound. Checksums
are integrity checks for trusted local artifacts, not cryptographic attestation of
an untrusted server or an external certification.

## Serving safeguards

`--backend program-v3` explicitly runs the experimental program policies.
`--backend risk-controlled` requires a valid separate ID-disjoint calibration artifact and
matching local model/runtime. Missing evidence or drift fails closed. `/health` and
`/answer` expose the active policy and calibration state alongside its assumptions.
The normal deterministic backend remains available without a model server.

The v3 model client disables proxies and redirects, validates a credential-free
loopback endpoint, and bounds requests, response reads, cache reads and timeouts.
Optional `FINANCIAL_QA_LLM_API_KEY` or `LLAMA_API_KEY` becomes an authorization header;
it is excluded from prompts, response-cache identity and reports. Raw caches can
contain source documents and stay outside Git. A trusted foreground model launcher
verifies weights and binds locally. Metadata checks cannot prove what an actively
malicious local server computed.

## Reproduce on a laptop

Install the project and test dependencies as described in the README. Download the
pinned training/development data through the existing CLI. The three optional model
files in this workspace occupy about 4.1 GB combined; running v3 requires only the
single 2.5 GB Qwen3-4B Q4_K_M file plus the small runtime. No paid API is required.

With the verified runtime/model in `local-runtime/`, macOS Apple Silicon can run:

```sh
local-runtime/llama-b11669/llama-server \
  -m local-runtime/qwen3-4b-q4.gguf \
  --host 127.0.0.1 --port 18081 -c 8192 -t 4 -tb 6 --parallel 1 \
  -ngl 99 --cache-ram 0 --alias qwen3-4b-q4-nonthinking \
  --reasoning off --cors-origins localhost --no-cors-credentials --no-webui
```

Use `-ngl 0` for CPU-only inference on a compatible runtime. CPU latency will differ
from the measured Metal run. Obtain a platform-specific binary; the stored runtime
is macOS arm64. Keep the binary's bundled dynamic libraries together.

The model is the official
[Qwen3-4B-GGUF Q4_K_M](https://huggingface.co/Qwen/Qwen3-4B-GGUF), pinned to revision
`bc640142c66e1fdd12af0bd68f40445458f3869b`. Expected size is 2,497,280,256 bytes and
SHA256 is `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`.
The model has an Apache 2.0 license. Code remains MIT; TAT-QA data licensing remains
separate. See [the model provenance artifact](../reports/qwen3-runtime-provenance.json) for
the exact download URL and runtime version. No model weights are committed to Git.

In a second terminal, run from the independent repository directory:

```sh
.venv/bin/financial-qa serve --backend program-v3 --program-policy model \
  --llm-url http://127.0.0.1:18081/v1 --model qwen3-4b-q4-nonthinking --port 8001
.venv/bin/financial-qa serve --backend risk-controlled \
  --llm-url http://127.0.0.1:18081/v1 --model qwen3-4b-q4-nonthinking \
  --model-path local-runtime/qwen3-4b-q4.gguf \
  --risk-certificate reports/v3-risk-certificate.json --port 8001
```

Experimental mode returns uncalibrated
answers. Risk-controlled mode can be deliberately unavailable if the calibration
criterion was not met. `/health` explains its state without fabricating confidence.

The experiment helper preserves existing outputs and checkpoints interrupted runs.
Use the frozen manifests; do not recreate them after viewing results:

```sh
.venv/bin/python scripts/experiment_v3.py development --policy model
.venv/bin/python scripts/experiment_v3.py freeze
.venv/bin/python scripts/experiment_v3.py calibration --policy model
.venv/bin/python scripts/experiment_v3.py calibration --policy semantic
.venv/bin/python scripts/experiment_v3.py calibration --policy hybrid
.venv/bin/python scripts/experiment_v3.py calibration --policy agreement
.venv/bin/python scripts/experiment_v3.py calibration --policy strict_agreement
.venv/bin/python scripts/experiment_v3.py certify
.venv/bin/python scripts/experiment_v3.py final --policy model
```

The other four final policies use the same final model-response cache. Raw model
responses retain request identity, weight hash and original output. Replayed latency
is cache/software latency; it must not replace the saved live model measurement.
Never change prediction code after `freeze` and then reuse the same calibration as
if it remained independent. The runner checks that code/profile identity still matches.

The repository's `Document.id` is a TAT-QA table-plus-paragraph context identifier,
not a verified unique source filing or company. Different contexts may originate
from related filings. The raw data does not supply a reliable filing-cluster key
for this experiment. Context disjointness therefore does not prove independence
between companies/filings; the IID assumption behind the risk bound must be assessed
for any intended deployment distribution, and can fail despite disjoint context IDs.

Reproduction also requires the recorded Python and dependency environment. Source
fingerprints do not attest external libraries, GPU kernels, or the operating system.
Changing those components requires checking the resulting requests/predictions and,
where inference changes, fresh separate ID-disjoint calibration rather than reusing an old bound.
