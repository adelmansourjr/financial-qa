# V3 measured results

**Later source audit:** different context IDs repeat some tables/paragraphs across
samples. Independence and IID assumptions remain unverified; saved scores/bounds
are not evidence of filing independence. Predictions and failed certificates are
preserved. Read [the findings and new guards](LEAKAGE_AND_FIT.md).


The preselected Qwen3-4B program policy improved unconditional numeric accuracy over
the deterministic rule policy on the same 86 final contexts. Separate ID-disjoint calibration
failed to establish the intended accepted-error bound. These are separate findings:
better overall accuracy did not make the system reliable enough for guarded answers.

## Final 86-context evaluation

| Policy | Correct | Answered | All-question accuracy | Accepted accuracy |
| --- | ---: | ---: | ---: | ---: |
| Semantic |35|46|40.7%|76.1%|
| Program model, preselected |54|76|62.8%|71.1%|
| Hybrid, descriptive |57|81|66.3%|70.4%|
| Agreement |32|39|37.2%|82.1%|
| Strict agreement |30|37|34.9%|81.1%|

The model was chosen as the experimental default on 100 training-development contexts:
66/100 correct versus 64/100 for hybrid. This choice was recorded before the final 86
outcomes. Hybrid was not promoted after the final comparison. The serverless browser
default remains deterministic `semantic`.

The model's gain over semantic rules is 22.1 percentage points, with a paired
1,000-replicate context-bootstrap 95% interval of 11.6–33.7 points. Its unconditional
accuracy interval is 52.3–73.3%, coverage interval 81.4–95.3%, and accepted-accuracy
interval 60.5–81.6%. For hybrid the descriptive overall-accuracy interval is 55.8–75.6%.
These comparisons are not adjusted for multiple testing.

The model made 22 accepted errors: 18 wrong values/operations and 4 unit-only errors.
It abstained on 10 questions, including one source-budget rejection before generation.
The final model pass contains 85 successful live requests, with end-to-end p95 latency
7.426 seconds on Apple Silicon Metal. The other policies use those identical saved
programs. Their replay latency does not measure fresh LLM inference.

The actual unmodified official TAT-QA evaluator was applied after freezing as a
secondary diagnostic. Its arithmetic-subset EM/F1 matches the primary numeric scores
for these saved predictions. This remains an arithmetic subset, not a full-benchmark
leaderboard result. See [the secondary report](../reports/v3-official-secondary.json).

## Separate 400-context calibration

The five policies were fixed before calibration outcomes. Each conditional-error
upper bound uses an exact one-sided Clopper-Pearson calculation with tail probability
0.05/5. A policy needed an upper bound of at most 10% and at least 20 accepted contexts.

| Policy | Accepted | Incorrect accepted | Empirical accepted accuracy | Error upper bound |
| --- | ---: | ---: | ---: | ---: |
| Semantic |230|53|77.0%|30.16%|
| Model |355|96|73.0%|32.91%|
| Hybrid |369|110|70.2%|35.68%|
| Agreement |191|24|87.4%|19.20%|
| Strict agreement |183|21|88.5%|18.09%|

**No policy qualified.** The selected policy is null and the artifact state is
`uncertified_fail_closed`. The `risk-controlled` backend therefore returns no numerical
answers, including on the final 86 contexts. Its accepted accuracy is undefined; zero
coverage must not be presented as perfect accuracy or as a 90% reliability guarantee.
The guard's final report applies this frozen calibration decision to the saved outcomes
without additional model calls.

The statistical target concerns the local numeric correctness criterion: scale-aware
conversion and a 0.005 tolerance in the gold unit. It does not certify human financial
truth. Even a qualifying bound would require fixed policies and IID future contexts
from the calibration distribution. No such claim is supported here.

## Integrity and practical limits

All 86 previously uninspected eligible development contexts were frozen before scoring,
excluding 191 contexts inspected in prior rounds. Training development and calibration
use distinct context IDs. A context is a table-plus-paragraph example, not a verified
independent filing/company. Related contexts can originate from a shared report;
disjoint IDs do not establish independent filings. Public benchmark contamination
in model pretraining remains possible.

Prediction code, prompt, model bytes, generation settings, runtime build/template,
and calibration identities were frozen before calibration. All ten frozen inference
component hashes still match. Historical results and interrupted training pilots remain
preserved. The serving-only alias/path guard is separately hash recorded and can only
restrict availability; it does not change the frozen predictions.

The compiler validates source references, dimensional compatibility and bounded
arithmetic, but semantic errors remain common. Prepared-context reasoning excludes
filing search, PDF extraction and OCR. Runtime metadata is not cryptographic attestation
of a malicious server or weights replaced behind an unchanged file path.

## Artifacts

- [All policy metrics, intervals and paired comparisons](../reports/v3-summary.json)
- [Frozen policy/profile grid](../benchmarks/policy-v3.json)
- [Failed calibration artifact and underlying outcomes](../reports/v3-risk-certificate.json)
- [Numeric criterion binding](../reports/v3-criterion-binding.json)
- [Training-only default selection](../reports/v3-development-selection.json)
- [Model and runtime provenance](../reports/qwen3-runtime-provenance.json)
- [Research basis, full protocol and reproduction](RESEARCH_V3.md)
