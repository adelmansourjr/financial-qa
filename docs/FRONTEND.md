# Browser application contract

The local demo intentionally has no frontend build step or external CDN dependency.
`src/financial_qa/static/index.html` is semantic markup, `styles.css` defines the
responsive presentation, and `app.js` owns interaction state. FastAPI serves the
assets under `/static`; all three files are included in the standalone wheel.

## Editing and explanation

Use modern browser JavaScript and the Fetch API. Keep source data in `textContent`
or text nodes; do not interpolate company paragraphs, movie titles, API messages,
or generated model output into HTML. Comments should explain a constraint,
assumption, or race boundary rather than repeat a line's syntax.

Async requests use AbortController to reduce unnecessary work and monotonic
revisions to prevent a completed older response from overwriting a newer input.
These solve different problems: a response can finish just before cancellation.
An input or mode change invalidates the previous result. Error messages remain
visible, controls recover after failure, and active computations expose aria-busy.
Initial health metadata cannot overwrite the reasoning status returned by an answer.
An empty or failed initial example catalog cannot replace a context the user edited.

Keyboard focus has a visible outline. Inputs have explicit labels; result changes
use live regions. Layout uses bounded cards and scrollable wide tables rather
than allowing the whole page to overflow on a narrow viewport. A diagram or list
in the UI is a demonstration, not a substitute for the saved evaluation report.

## Verification

From the project directory, run `node --check src/financial_qa/static/app.js`
when Node is available. Python integration tests verify static asset routes and
core API contracts. Backend installation does not require Node.

The browser regression suite used for the published demo screenshots exercised
real local data/artifacts in an isolated installed browser. It checked successful
flows, deliberately delayed responses, visible API outages and recovery, chart
rendering where applicable, and mobile overflow. It did not reuse a personal
browser profile or download browser binaries. Its screenshots are presentation
assets, not evidence of model accuracy.

API schemas and artifact format documentation remain authoritative for requests
and responses. Add a selectable UI method only after the backend supports it,
and identify the method actually served. Preserve compatibility labels for older
artifacts rather than pretending an unavailable model was evaluated.

## Local HTTP boundary

The app page loads packaged scripts under a same-origin Content Security Policy.
Responses include correlation IDs, timing and privacy headers; supplied document
content, prompts and query strings are not written by the request logger. The
CLI disables Uvicorn's separate query-bearing access log. JSON body bytes and
nesting are bounded before framework validation; nonfinite JSON numbers and
invalid encodings are rejected. Ordinary malformed JSON retains the framework's
validation response. Unexpected pre-response failures retain the correlation ID
and a generic error without exposing exception text. CLI concurrency is limited
to 16. These local boundaries are separate from model correctness.
