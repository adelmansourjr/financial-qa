# V5 engineering diagnosis on reused training development

This analysis uses only the original, already-inspected **100 training-development contexts**. It does not inspect held-out or calibration errors and is not a new accuracy estimate. V4 scored 72/100 correct with 88 answered; v3 scored 66/100 on the same contexts. V4 fixed 15 old misses and introduced 9 regressions.

| Primary cause | Development misses |
|---|---:|
| operator or schema | 3 |
| annotation or question ambiguity | 5 |
| source parsing and missingness | 1 |
| inferred total reasoning | 1 |
| model operand selection | 12 |
| period validation | 2 |
| unit grounding | 3 |
| typed arithmetic coverage | 1 |

The largest recurring issue is model selection of an adjacent row or column, including explicit incorrect year claims. Three cached programs are already correct but blocked by period or count-typing constraints: the three-year effective-rate average, current-year backlog change, and store-count percentage change. Most other rejected programs contain real selection errors; broadly relaxing validators would increase incorrect answers.

The highest-value intervention is to show explicit numeric operand records with their source values, periods and full metric labels. Focused compiler and unit fixes should accompany this, with new synthetic tests. Preserve frozen v3/v4 modules and avoid optimizing toward questionable sign, scale or adjusted-balance annotations.

## Exact development examples

### 94eaf52d-0a96-4437-8ed1-215f3580b0cf

What is the sum of acquisitions and other for Bell Wireless and Bell Wireline in 2018? 

**operator or schema**. Explicit sum is unconstrained: the model selected the correct values 16 and 182 but subtracted. Constrain explicit sum requests while retaining compound ratio-of-sums support.

Observed prediction: `-166` (unscaled); reference: `198` (unscaled).

### df1cce24-6ff2-42f2-8c50-a62ce8a5d6ed

What was the difference between the transitional and other employee related costs in 2019 and 2018?

**operator or schema**. The model selected the correct values 49 and 48 but reversed the difference. Temporal differences need a documented order convention; an unqualified category difference can remain ambiguous.

Observed prediction: `-1` (million); reference: `1` (million).

### fe840462-8387-4566-b234-3cd200c99868

What was the difference between current and non-current deferred revenues after adjustment in 2018?

**annotation or question ambiguity**. The question says after adjustment: adjusted values 107.5 minus 57.7 equal 49.8, matching the prediction. The reference uses unadjusted values 100.7 minus 59.6. Do not change grounded logic to reproduce this label.

Observed prediction: `49.8` (million); reference: `41.1` (million).

### 3daf69dd-76df-402a-afa9-8554100b0441

What is the average total number of shares purchased?

**source parsing and missingness**. Footnoted numeric cells such as 3,646,581 (i) are absent from numeric Evidence variables; a dash is missing rather than explicit zero. The cached model repeated a total variable, correctly rejected by the compiler. Do not silently treat every dash as zero.

Observed prediction: `None` (unscaled); reference: `1581118` (unscaled).

### 0a789647-0d43-4d7c-80ce-bc958586e445

What was the increase / (decrease) in the Average term: long-term debt (in years) from 2018 to 2019?

**annotation or question ambiguity**. The numerical difference -0.8 is correct for a duration in years; reference scale percent contradicts the source row. Do not infer percent from a reference annotation.

Observed prediction: `-0.8` (unscaled); reference: `-0.8` (percent).

### 2af8fdba-3c86-4c4e-9b83-89234f95feb3

What is the amount of net sales in 2019?

**inferred total reasoning**. Net sales are indirectly recoverable from SG&A of 205471 and a 6.7 percent share. This requires amount divided by a percentage fraction, rather than ordinary division of incompatible dimensions.

Observed prediction: `None` (unscaled); reference: `3066731.34` (thousand).

### 2300b901-0c13-4fce-beca-927f3ea9ab1c

What is the percentage change in game operations and distribution costs between 2018 and 2019?

**model operand selection**. The model selected the 2018 amount 1028 and change -63 instead of the 2019 amount 965 and 2018 amount 1028. Explicit operand metadata can distinguish neighboring amount, percentage and change columns.

Observed prediction: `-1731.746031746031746031746032` (percent); reference: `-6.13` (percent).

### 33c585f0-5818-477b-85bd-944714b595ea

What was the average effective rate over the 3 year period from 2017 to 2019? 

**period validation**. The cached model correctly selected 10.2, 54.6 and 14.8. The compiler interprets from 2017 to 2019 as endpoints only and rejects 2018. Expand bounded inclusive ranges for aggregation; do not add intermediate years to a two-point change.

Observed prediction: `None` (unscaled); reference: `26.53` (percent).

### 5e07dca0-5ed3-4863-bbee-182cc5de4f6c

How many percent of organic change in Europe adjusted operating profit is the organic change in Italy EBITDA?

**model operand selection**. The requested ratio is Italy EBITDA 4.6 divided by Europe adjusted operating profit 86.3. The model instead selected Europe EBITDA 13.0 as denominator. Preserve the full metric and section label for every operand.

Observed prediction: `35.38461538461538461538461538` (percent); reference: `5.33` (percent).

### ca2b90ed-7468-4fbb-958e-1c287cb97e46

What is the difference between the FVTPL of the other current and long-term financial assets?

**model operand selection**. The question requests FVTPL values 33796 and 18632; the model selected neighboring amortised-cost values 21959 and 147. Bind the exact column heading to every operand.

Observed prediction: `21812` (unscaled); reference: `15164` (thousand).

### 09a8a7fb-da3c-43b5-8fdc-248809808951

What is the ratio of income  from discontinued operations (net of tax) to income for the performance of services (classified within SG&A expenses) during fiscal 2017?

**model operand selection**. The question requests net-of-tax discontinued income 2.6 divided by services income 16.9; the model selected a tax benefit of -0.3 and services income of 2.2. Match both metric and period across table and paragraph sources.

Observed prediction: `-0.1363636363636363636363636364` (unscaled); reference: `0.15` (unscaled).

### 8483aa56-4f9a-402a-9ebe-9879079df5d3

What was the change in the amount Expiring after the following year in 2019 from 2018?

**annotation or question ambiguity**. Identical Expiring after the following year labels occur in unused tax losses and tax credits sections. The question names neither section. The model answers 373-476; the reference uses 17-18. Clarify or abstain when repeated row labels remain tied.

Observed prediction: `-103` (million); reference: `-1` (million).

### b57b4869-5ae9-4899-b4fa-0729378311f2

What is the increase/ (decrease) Primary service units of Net additions (losses) from 2018 to 2019?

**model operand selection**. The model selected the current service-unit balance 1810366 instead of net additions -2846; the older comparator -35818 is correct. Both the Net additions (losses) column group and period must match.

Observed prediction: `1846184` (unscaled); reference: `32972` (unscaled).

### 98d1243f-7450-47d6-91ba-1204b35d48f6

What is the change in the backlog for Cubic Mission Solutions in 2019?

**period validation**. The cached model correctly proposes 103.7-77.0 for change in 2019. Exact equality with the requested set {2019} rejects the necessary prior-year comparator. Permit a constrained adjacent comparator only for a clear temporal change with the same metric and granularity.

Observed prediction: `None` (unscaled); reference: `26.7` (million).

### 7abea844-a63e-4e8b-9181-446f9c66da92

What is the change in the number of unvested RSUs between 2018 and 2019?

**unit grounding**. The share-count difference 9328-10623 is correct, but its multiplier is lost. The explicit amounts in thousands except per-share declaration should scope the count column while excluding the price column. An unrelated dollar price must not erase the count multiplier.

Observed prediction: `-1295` (unscaled); reference: `-1295` (thousand).

### 26d56b99-69c3-48d8-80e6-4697d74dd5f7

What is the average cost of revenue for 2018 and 2019 as a percentage of total revenues?

**operator or schema**. The question asks for the average of common-size percentages 10.6 and 11.0. Grammar forces percent_ratio because of as a percentage, and the common-size declaration remains untyped. Resolve the main average operation before a percentage-of-total metric qualifier.

Observed prediction: `None` (unscaled); reference: `10.8` (percent).

### f7f0edf0-5b6e-417a-b017-e47182453588

What is the difference in total assets classified as held for sale between 2018 and 2019?

**annotation or question ambiguity**. Temporal newer-minus-older gives -596, consistent with the source values. The reference uses older-minus-newer 596 for an unqualified difference. Keep a documented convention rather than taking the absolute value of every difference.

Observed prediction: `-596` (million); reference: `596` (million).

### 0ccc56dd-4b11-4c82-88c6-672df70228a2

What is the total price of stock options exercised or canceled and expired during the period between December 31, 2018, and 2019?

**typed arithmetic coverage**. Count times price per share is economically valid but unsupported. The cached model also selected the wrong rollforward rows, so adding typed multiplication alone will not repair the case. Preserve source signs and distinguish transaction magnitude explicitly.

Observed prediction: `None` (unscaled); reference: `18408588.96` (unscaled).

### c3937d00-21f3-445e-84bc-372ef138ebf0

What is the total amortization expense for intangible assets for 2017, 2018 and 2019?

**model operand selection**. The correct amortization expenses 1.7, 1.8 and 1.2 billion are paragraph values. The model instead selects goodwill table values and repeats one source. The duplicate-source guard is correct. Flat paragraph records with units and periods may improve selection.

Observed prediction: `None` (unscaled); reference: `4.7` (billion).

### 93b7dc03-aee1-48e4-b926-dbe54a180c58

What is the total foreign component of loss before income tax in 2017 and 2018?

**annotation or question ambiguity**. The source foreign-loss entries are -320 and -469; signed total -789 matches the prediction while the reference reports loss magnitude 789. This is a sign interpretation issue; do not globally strip negative signs.

Observed prediction: `-789` (thousand); reference: `789` (thousand).

### 673d6c1d-9e3b-4310-bbb9-e16dd8ff4910

What is the change in the Aa2/AA and above from 2018 to 2019?

**unit grounding**. The numerical difference -53 is correct, but the header shorthand £m is not recognized as million. Parse unambiguous currency-adjacent multipliers without reinterpreting arbitrary single-letter metric labels.

Observed prediction: `-53` (unscaled); reference: `-53` (million).

### 4930bf0c-863b-4258-bb6f-02abbb36236f

What is the ratio of dividends that the company received from equity method investments to total earning after income taxes in fiscal 2019?

**model operand selection**. The cached model identifies dividend 55.0 correctly but labels r12c2=207.1 as 2019 despite its actual 2018 heading. The required denominator is r12c1=170.0. The period guard correctly rejects the proposal.

Observed prediction: `None` (unscaled); reference: `0.32` (unscaled).

### 79faca5c-720f-4eb7-bc7f-5e3e77444645

What is the percentage change in revenue between 2018 and 2019?

**model operand selection**. The cached model labels 12629 and 12149 as 2019 and 2018 despite their actual 2018 and 2017 headings. Explicit year-value records may reduce this column selection error; do not weaken the correct year guard.

Observed prediction: `None` (unscaled); reference: `-18.36` (percent).

### af7747f0-ba4b-4a9c-bb2c-c03e2b227868

How much was the total amount of foreign currency losses, net and interest income across 2018 and 2019?

**model operand selection**. The cached program selects constant-currency percentage columns and subtracts losses although the question requests combined signed amounts. Unit rejection is correct. Exact metric and column-kind selection are needed.

Observed prediction: `None` (unscaled); reference: `2110` (million).

### d97be352-26c2-4c4b-a8ce-459f5a24f124

What is the change in Aggregate purchase price from year ended December 31, 2018 to 2019?

**model operand selection**. The model selects total cash used, 139.0-315.3, instead of aggregate purchase price, 138.9-315.0. Similar adjacent financial metrics require exact label grounding.

Observed prediction: `-176.3` (million); reference: `-176.1` (million).

### fe33bc0a-1d6f-41fc-be93-51ee98523ec7

What is the percentage change in the number of opened or refitted unites in the centres from 2018 to 2019?

**unit grounding**. The cached model correctly selects 256 units and 262 stores. The count lexicon misses stores; sentence-year propagation assigns the current 256 to parenthetical 2018. Local comparison periods must remain unresolved unless the source states the actual year.

Observed prediction: `None` (unscaled); reference: `-2.29` (percent).

### e8f43fba-733a-49ed-9937-cf11d72f3641

What is the percentage change in the income before income taxes from United States from 2017 to 2018?

**model operand selection**. The model selects the 2019 loss -59876 and 2018 income 128190 for a 2017-to-2018 request, even inventing a negative sign in its explanation. The period guard correctly rejects the proposal. Explicit operand records should bind both value and period.

Observed prediction: `None` (unscaled); reference: `1597.21` (percent).

### 84457adf-9f2a-45fb-bbc4-07381e92fe85

What is the average age of the company's Senior Vice Presidents?

**model operand selection**. The average requires all seven rows whose current Position contains Senior Vice President. The model selects only three. Select the complete matching population using current table positions, excluding historical biographical positions.

Observed prediction: `56` (unscaled); reference: `57.57` (unscaled).

## Validation boundary

No new model call, reserve consumption, evaluation retuning, or existing prediction-module change occurred. Freeze the new policy before any fresh source-group-disjoint evaluation. Improvements on these reused development examples alone do not establish generalization.
