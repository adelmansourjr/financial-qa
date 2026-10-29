"use strict";

/** Render prepared financial evidence without interpreting source text as HTML. */
const elements = Object.fromEntries(
  ["form", "example", "question", "document", "provenance", "submit", "answer",
    "status", "plan", "sources", "error", "backend", "risk-state"].map(id => [id, document.getElementById(id)])
);
const starter = {
  question: elements.question.value,
  document: elements.document.value,
};
let revision = 0;
let contextRequest;
let answerRequest;
let answerReasoningReceived = false;

async function readJson(response) {
  const data = await response.json();
  if (!response.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
  }
  return data;
}

function resetResult() {
  elements.answer.textContent = "—";
  elements.status.textContent = "Submit a question to inspect its calculation and sources.";
  elements.plan.textContent = "No calculation yet.";
  elements.sources.replaceChildren();
  elements.error.textContent = "";
}

function setBusy(busy) {
  elements.submit.disabled = busy;
  elements.form.setAttribute("aria-busy", String(busy));
  elements.submit.textContent = busy ? "Analyzing…" : "Analyze evidence";
}

function renderEvidence(evidence) {
  elements.sources.replaceChildren();
  for (const source of evidence) {
    const entry = document.createElement("div");
    entry.className = "evidence";
    const title = document.createElement("strong");
    title.textContent = `${source.id} `;
    entry.append(title, document.createTextNode(source.text));
    if (Object.keys(source.variables).length) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "Source variables";
      const code = document.createElement("pre");
      code.textContent = JSON.stringify(source.variables, null, 2);
      details.append(summary, code);
      entry.append(details);
    }
    elements.sources.append(entry);
  }
}

function renderAnswer(result) {
  if (result.reasoning) {
    answerReasoningReceived = true;
    renderRiskState(result.reasoning);
  }
  const scale = result.plan?.scale;
  const suffix = scale === "percent" ? "%" : scale ? ` ${scale}` : "";
  elements.answer.textContent = result.abstained
    ? "Abstained"
    : Number(result.answer).toLocaleString(undefined, { maximumFractionDigits: 4 }) + suffix;
  elements.status.textContent = result.abstained
    ? result.reason
    : `Executed with cited variables · ${result.latency_ms} ms`;
  elements.plan.textContent = JSON.stringify(result.plan, null, 2);
  renderEvidence(result.evidence);
}

/** A valid arithmetic expression does not itself establish answer reliability. */
function renderRiskState(state) {
  if (state.certified) {
    elements["risk-state"].textContent = `Independent calibration: ${state.policy}. ` +
      `Target accepted error ≤${(100 * state.target_error).toFixed(0)}%, ` +
      `${(100 * (1 - state.familywise_alpha)).toFixed(0)}% family-wise confidence under the reference-distribution assumptions. ` +
      "This bound does not cover arbitrary user documents or distribution shifts.";
  } else if (typeof state.state === "string" && state.state.endsWith("fail_closed")) {
    elements["risk-state"].textContent = `Risk-controlled serving is unavailable; this mode abstains. ${state.reason || "No policy met the calibration target."}`;
  } else {
    elements["risk-state"].textContent = "Independent risk calibration is not applied to this backend. " +
      "Read answer accuracy and coverage together; citations and arithmetic checks alone do not establish correctness.";
  }
}

/** Invalidate both in-flight actions when inputs change; cancellation alone can race. */
function invalidate() {
  revision += 1;
  contextRequest?.abort();
  answerRequest?.abort();
  setBusy(false);
  return revision;
}

elements.example.addEventListener("change", async event => {
  const current = invalidate();
  resetResult();
  const identifier = event.target.value;
  if (!identifier) {
    elements.question.value = starter.question;
    elements.document.value = starter.document;
    elements.provenance.textContent = "Current source: synthetic demonstration.";
    return;
  }
  contextRequest = new AbortController();
  elements.submit.disabled = true;
  elements.provenance.textContent = "Loading selected context…";
  try {
    const source = await fetch(`/examples/${encodeURIComponent(identifier)}`, {
      signal: contextRequest.signal,
    }).then(readJson);
    if (current !== revision) return;
    elements.question.value = source.question;
    elements.document.value = JSON.stringify({ table: source.table, paragraphs: source.paragraphs }, null, 2);
    elements.provenance.textContent = `TAT-QA development · ${source.source.document_id} · CC BY 4.0`;
  } catch (error) {
    if (error.name !== "AbortError" && current === revision) {
      elements.error.textContent = error.message;
      elements.provenance.textContent = "Selected context could not be loaded.";
    }
  } finally {
    if (current === revision) setBusy(false);
  }
});

// Edited content becomes user-supplied evidence. Never retain an answer for older inputs.
for (const input of [elements.question, elements.document]) {
  input.addEventListener("input", () => {
    invalidate();
    elements.example.value = "";
    elements.provenance.textContent = "Current source: user-edited context.";
    resetResult();
  });
}

elements.form.addEventListener("submit", async event => {
  event.preventDefault();
  const current = invalidate();
  resetResult();
  setBusy(true);
  answerRequest = new AbortController();
  try {
    const source = JSON.parse(elements.document.value);
    const result = await fetch("/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: answerRequest.signal,
      body: JSON.stringify({ ...source, question: elements.question.value }),
    }).then(readJson);
    if (current === revision) renderAnswer(result);
  } catch (error) {
    if (error.name !== "AbortError" && current === revision) elements.error.textContent = error.message;
  } finally {
    if (current === revision) setBusy(false);
  }
});

async function initialize() {
  const initialRevision = revision;
  try {
    const health = await fetch("/health").then(readJson);
    elements.backend.textContent = health.backend === "CandidateProgramPlanner"
      ? "V5 · grounded programs · experimental"
      : `Backend: ${health.backend}`;
    // The answer may already reflect a later drift/failure than startup health.
    if (!answerReasoningReceived) renderRiskState(health.reasoning || { certified: false });
  } catch {
    elements.backend.textContent = "Backend unavailable";
    if (!answerReasoningReceived) elements["risk-state"].textContent = "The backend and answer-policy status could not be verified.";
  }
  try {
    const examples = await fetch("/examples").then(readJson);
    for (const example of examples) {
      const option = document.createElement("option");
      option.value = example.id;
      option.textContent = example.question;
      elements.example.append(option);
    }
    if (!examples.length && revision === initialRevision) elements.provenance.textContent = "Synthetic demonstration. Download TAT-QA to load real contexts.";
  } catch {
    if (revision === initialRevision) elements.provenance.textContent = "Example catalog unavailable; the supplied context can still be analyzed.";
  }
}

initialize();
