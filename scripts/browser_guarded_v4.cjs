/* Verify the actual failed v4 certificate in a running loopback application.
 * Uses installed Playwright/Chrome and an isolated profile; downloads nothing.
 * Run a risk-controlled-v4 server first. These checks measure serving behavior,
 * not model accuracy. No API response is mocked.
 */
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const base = process.env.QA_BASE_URL || "http://127.0.0.1:8005";
const output = path.resolve(process.env.QA_CHECK_OUTPUT || "../.run/screenshots");

async function main() {
  const url = new URL(base);
  assert.ok(["127.0.0.1", "localhost", "[::1]"].includes(url.hostname));
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({
    executablePath: process.env.CHROME_EXECUTABLE,
    headless: true,
  });
  const errors = [];
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1050 } });
    const page = await context.newPage();
    page.on("pageerror", error => errors.push(error.message));
    const response = await context.request.get(`${base}/health`);
    assert.ok(response.ok());
    const health = await response.json();
    assert.equal(health.reasoning.version, 4);
    assert.equal(health.reasoning.certified, false);
    assert.equal(health.reasoning.state, "uncertified_fail_closed");
    await page.goto(base);
    await page.locator("#risk-state").filter({ hasText: "failed independent calibration" }).waitFor();
    await page.getByRole("button", { name: "Analyze evidence" }).click();
    await page.locator("#answer").filter({ hasText: "Abstained" }).waitFor();
    assert.match(await page.locator("#status").textContent(), /failed independent calibration/);
    assert.equal(await page.locator("#error").textContent(), "");
    const answer = await context.request.post(`${base}/answer`, { data: {
      question: "What was the percentage increase in revenue from 2022 to 2023?",
      table: [["USD million", "2023", "2022"], ["Revenue", "120", "100"]],
      paragraphs: ["Synthetic serving test, not financial accuracy evidence."],
    } });
    assert.ok(answer.ok());
    const result = await answer.json();
    assert.equal(result.abstained, true);
    assert.equal(result.answer, null);
    assert.equal(result.reasoning.state, "uncertified_fail_closed");
    await page.screenshot({ path: path.join(output, "financial-qa-v4-guarded.png"), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.evaluate(() =>
      document.documentElement.scrollWidth <= window.innerWidth + 1), true);
    await page.screenshot({ path: path.join(output, "financial-qa-v4-guarded-mobile.png"), fullPage: true });
    assert.deepEqual(errors, []);
    const report = {
      status: "passed", checked_at_utc: new Date().toISOString(), browser: browser.version(),
      mocked_responses: false, health, answer: result,
      checks: ["actual v4 certificate fails closed", "real API abstains without an answer",
        "visible failed-calibration reason", "mobile overflow", "uncaught exceptions"],
    };
    fs.writeFileSync(path.join(output, "qa-v4-guarded-browser.json"), JSON.stringify(report, null, 2) + "\n");
    console.log(JSON.stringify({ status: report.status, checks: report.checks }));
    await context.close();
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
