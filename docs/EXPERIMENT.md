# Financial reasoning upgrade experiment

## Question and scope

Can a bounded local reasoning pipeline improve on a simple row/year lookup while
making source selection and unsupported calculations inspectable? The task is numeric
arithmetic over prepared TAT-QA tables and paragraphs. PDF ingestion, general financial
advice, span extraction, and full-document retrieval are outside this experiment.

## Development and freezing

1. Preserve the original reports: 100-question baseline and two ten-question LLM pilots.
2. Inspect the first five training contexts and iterate using the first three arithmetic
   training questions. Those observations motivated multirow headers, merged column
   labels, footnote-token handling, and short synthetic operation examples.
3. Before new scoring, write `benchmarks/dev-holdout-v1.json`. With seed 71, exclude the
   entire 41-document set represented by the original first 100 arithmetic development
   questions. Sample 50 different remaining documents and one eligible question per
   document. The same manifest is used for every variant.
4. Freeze prediction logic before scoring the selected 50 examples. Do not improve
   prompts or rules using their failures. Formatting, tests, API example loading,
   and report writing can change without altering those predictions.

`freeze_manifest` validates cached manifest size, seed, membership, uniqueness,
question/document identity, and disjointness when reused. Plans receive only `Document`
and question text, never `Example.answer`, derivations, or gold evidence annotations.
The browser uses previously explored development examples, not the holdout catalog.

## Variants

- **Legacy:** a first-row year-header assumption and narrow row-label matching.
- **Header-aware:** preserve all detected header rows outside the six-body-evidence
  budget; recover merged group/year columns and source units; choose conservative
  single-metric lookups, averages, and year changes. Abstain on ambiguous rows/columns.
- **Structured LLM:** the same coherent evidence layout; natural row/column operand
  catalog; dynamic JSON Schema with operation and variable-ID enums; synthetic
  demonstrations; deterministic formula compilation and citation derivation.

The LLM emits operation, operands, and scale rather than free-form code or citations.
Supported operations are lookup, sum, difference, average, divide, percentage ratio,
percentage change, and difference of two averages, plus explicit abstention. The
compiler rejects incompatible explicit units and invalid operation arity. Passing
these checks does not establish that the selected metric, years, or operation are right.

The LLM is Qwen2.5-1.5B-Instruct Q4_K_M with the already downloaded llama.cpp b11669
runtime. Inference used CPU, two threads, one slot, 8,192-token context, temperature 0,
and at most 150 output tokens. No fine-tuning, paid inference, or additional weights
were used. `reports/runtime-provenance.json` identifies the model hash and revision;
`reports/upgrade-comparison.json` records this upgrade's settings and total tokens.

## Scoring

All 50 examples remain in the accuracy denominator, including abstentions and invalid
plans. Numeric predictions are converted to the gold scale and compared with absolute
tolerance 0.005 in that unit. This is a transparent local score, **not official TAT-QA
EM/F1**. It is not directly comparable with leaderboard results.

Intervals use 1,000 paired document bootstrap replicates with seed 71. Each document
has one question in this manifest, but grouping is explicit so comparisons cannot
mistreat related questions as independent if the evaluation expands later.

| Variant | Correct | Accuracy | 95% interval | Coverage | Accepted-answer accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy | 3/50 | 6% | 0–12% | 16% | 37.5% |
| Header-aware | 17/50 | 34% | 21.95–46.05% | 52% | 65.4% |
| Structured LLM | 22/50 | 44% | 30–58% | 98% | 44.9% |

Paired accuracy differences:

- Header-aware minus legacy: +28 points, interval +16 to +40.
- Structured LLM minus legacy: +38 points, interval +26 to +52.
- Structured LLM minus header-aware: +10 points, interval −2 to +24.

Thus the combined upgrade improves on the weak legacy baseline in this sample. The
study does not establish an LLM advantage over the stronger header-aware baseline.
The LLM answers more questions and makes more accepted errors. Its 50/50 valid JSON
responses show protocol improvement, not 100% reasoning reliability. One response had
invalid operation arity; the remaining failures were 27 wrong values and/or units.

## Reproduction

From the project directory with the installed package:

```sh
financial-qa download --split dev
financial-qa benchmark --backend baseline --output reports/replay-legacy.json
financial-qa benchmark --backend header --output reports/replay-header.json
# Start the documented local llama.cpp server first.
financial-qa benchmark --backend structured \
  --llm-url http://127.0.0.1:18080/v1 --model qwen2.5-1.5b-q4 \
  --output reports/replay-structured.json
```

The last command reuses the ignored raw-response cache if available. Cached replays
reproduce predictions but measure cache-I/O latency, not live model latency. Use a
fresh `--cache-dir` to measure live inference again; preserve the original reports.
The manifest and dataset receipt hashes make the selected questions and source snapshot
auditable. Public benchmark contamination in pretrained models remains possible; this
is a development holdout rather than an uncontaminated final-test claim.
