# Source overlap and model-fit audit

The [completed audit](../reports/leakage_audit.json) found repeated source content
across ID-disjoint samples. The earlier complete table-plus-paragraph audit found
no identical bundles; that narrower result did **not** establish table, paragraph
or filing independence. Original predictions and manifests remain preserved.

## Findings

The pinned train/dev snapshots contain 2,479 contexts and 2,240 conservative source
components. Components join identical normalized tables or paragraphs with at
least 80 characters and 12 words, including indirect chains through other contexts.
Numbers and signs remain in the fingerprints. All sources participate, including
contexts not selected in an experiment.

The 300-context v4 calibration sample shares **one exact table each** with prompt
development, the previous calibration sample, and v4 comparison. There are also
**8 shared source components** with prompt development, **10** with previous
calibration, and **8** with comparison. Boilerplate can produce conservative joins;
these are source-relatedness signals rather than verified filing counts. The
report retains context IDs and source hashes without copying source text.

These overlaps weaken the independent-context/IID interpretation of saved bounds
and intervals. The scores remain descriptive observations, and failed certificates
remain failed. No overlapping/error rows were removed, no favorable replacement
bound calculated, and no policy retuned against these inspected outcomes.

## Protections added

`split_integrity.py` enforces complete source-component exclusions and at most
one question per unused component. Gold values, scales, predictions and errors
never rank candidates; answer type and numeric availability define task
eligibility. Annotation-poisoning tests verify identical LLM request bytes when
answers, scales, derivations and answer mappings change.

The historical consumption ledger pins both source hashes, source URLs, byte
counts and three completed training manifests. The entire inspected public-dev
split is retired: **1,386 consumed contexts**, **1,292 source components**. There
are 935 unused arithmetic components before the numeric-eligibility filter.
No new final sample was created or scored. Nested reports and interrupted
checkpoints are checked for additional consumed IDs. This is a backstop, not a
complete record of manual or deleted model calls.

`source_reservations.py` records future manifests in a shared versioned ledger
before publishing them under `benchmarks/future/`. Each new reservation consumes
all earlier reserved components. An exclusive writer lock prevents competing
reservations; every load reconstructs the complete chain. Edited exclusions,
identities, membership, counts, destination aliases and published manifests fail.
Publication failure still consumes the reserved sources. Ledger rollback with
surviving manifests fails, and a missing ledger is never silently reset.

Keep the ledger and manifests in version control, including abandoned studies.
Checksums are not signatures against coordinated rewriting/deletion of local
history. After a writer crash, confirm the owner has stopped before lock recovery;
never delete reservations to reopen inspected sources. Future studies must use
this workflow; frozen historical scripts remain reproduction paths.

## Overfitting and underfitting

No LLM weights were trained locally. Prompts, grammar and acceptance choices were
developed on 100 contexts. V4 scores **72.0%** there and **65.5%** on the later
200-context comparison. The 6.5-point gap is descriptive across different
questions, not a controlled diagnosis. Repeated prompt selection can overfit;
a pretrained model can also lack task capability. There is no local weight-training
learning curve establishing convergence or ruling out underfitting.

The paired improvement remains inconclusive, and the narrow gate's **10.75%**
upper error bound exceeds its unchanged **10%** target, even before resolving
source dependence. Further selection needs a new development design and a
source-group-disjoint evaluation frozen before predictions. Grouping cannot find
all paraphrases, edited tables or common filings, nor inspect model pretraining.
These checks cannot certify absence of every leakage or fitting risk.

## Reproduce

From this project with the existing pinned data:

```sh
.venv/bin/python scripts/audit_leakage.py audit --output reports/leakage-recheck.json
.venv/bin/python scripts/audit_leakage.py audit --require-disjoint --output reports/leakage-strict-recheck.json
```

The strict command intentionally exits 2 for the existing overlapping samples.
Outputs refuse replacement. After deciding a new design, source-only reservations
use explicit size/seed; the second automatically excludes the first:

```sh
.venv/bin/python scripts/audit_leakage.py freeze --size 100 --seed 521 --output benchmarks/future/study-calibration.json
.venv/bin/python scripts/audit_leakage.py freeze --size 100 --seed 523 --output benchmarks/future/study-comparison.json
.venv/bin/python scripts/audit_leakage.py validate --size 100 --seed 521 --output benchmarks/future/study-calibration.json
.venv/bin/python scripts/audit_leakage.py validate --size 100 --seed 523 --output benchmarks/future/study-comparison.json
```

Reservation alone is not an accuracy study: freeze model, prompt, scorer, runtime,
sample sizes and target before outcomes. Preserve all earlier reservations.
Principles follow the [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
and [model-selection discussion](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
Counts and fit observations are measured by this repository's audit.
