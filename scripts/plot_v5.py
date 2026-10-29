"""Render the completed frozen v5 comparison; never fit, select or call a model.

Requires the optional matplotlib package. Run from financial-qa after summarize.
Every plotted outcome is checked against the source-bound experiment reports.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from financial_qa.benchmark import interval
from financial_qa.calibration import file_sha256
from financial_qa.data import load_examples
from financial_qa.storage import write_json


def checked_results() -> dict:
    """Validate membership, source labels, counts, policy and paired uncertainty."""
    path = Path(__file__).with_name("experiment_v5.py")
    spec = importlib.util.spec_from_file_location("plot_v5_study", path)
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    policy, selected = study.verify_freeze()
    policy_sha = file_sha256(study.POLICY)
    summary = json.loads(Path("reports/v5-summary.json").read_text())
    if (
        summary["policy_sha256"] != policy_sha
        or summary["scope"] != policy["scope"]
        or summary["accuracy_target"] != policy["accuracy_target"]
        or summary["limitations"] != policy["limitations"]
    ):
        raise ValueError("Summary does not match the frozen policy")
    lookup = {example.id: example for example in load_examples(study.DATA)}
    reports, methods = {}, {}
    for name in ("baseline", "candidate"):
        report_path = Path(f"reports/v5-comparison-{name}.json")
        report = json.loads(report_path.read_text())
        study.validate_report(report, "comparison", name, selected, lookup, policy["identity"])
        if report["policy_sha256"] != policy_sha:
            raise ValueError("Prediction report does not match the frozen policy")
        if summary["methods"][name] != {
            key: value for key, value in report.items() if key not in {"identity", "predictions"}
        }:
            raise ValueError("Summary method metrics differ from prediction reports")
        reports[name] = report
        rows = report["predictions"]
        methods[name] = {
            "report_sha256": file_sha256(report_path),
            "correct": report["correct"],
            "answered": report["answered"],
            "accuracy": report["accuracy"],
            "coverage": report["coverage"],
            "answered_accuracy": report["answered_accuracy"],
            "p95_latency_ms": report["p95_latency_ms"],
            "accuracy_interval": interval(
                [float(row["correct"]) for row in rows], [row["document_id"] for row in rows]
            ),
        }
    gain = interval(
        [
            float(new["correct"]) - float(old["correct"])
            for old, new in zip(
                reports["baseline"]["predictions"], reports["candidate"]["predictions"], strict=True
            )
        ],
        [row["document_id"] for row in reports["candidate"]["predictions"]],
    )
    if (
        gain != summary["paired_accuracy_gain"]
        or summary["accuracy_target_met"]
        != (methods["candidate"]["accuracy"] >= policy["accuracy_target"])
        or summary["improvement_target_met"] != (gain["ci95"][0] > 0)
    ):
        raise ValueError("Summary decisions differ from recomputed outcomes")
    return {
        "scope": policy["scope"],
        "questions": len(selected),
        "policy_sha256": policy_sha,
        "summary_sha256": file_sha256(Path("reports/v5-summary.json")),
        "plot_source_sha256": file_sha256(Path(__file__)),
        "accuracy_target": policy["accuracy_target"],
        "accuracy_target_met": summary["accuracy_target_met"],
        "paired_accuracy_gain": gain,
        "methods": methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        import matplotlib
    except ImportError:
        parser.error(
            "Install optional figure dependencies: pip install -r requirements-analysis.txt"
        )

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_path = Path("reports/v5-figure-data.json")
    stem = Path("reports/figures/financial-qa-v5")
    outputs = [data_path, stem.with_suffix(".svg"), stem.with_suffix(".png")]
    if any(path.exists() for path in outputs):
        raise ValueError("Preserve existing figure evidence; refusing replacement")
    data = checked_results()
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 11, "svg.hashsalt": "financial-qa-v5"}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle("Financial QA: frozen source-group comparison", x=0.06, ha="left", weight="bold")
    fig.text(0.06, 0.89, data["scope"], color="#475569", fontsize=10)
    names, labels = ("baseline", "candidate"), ("V4 baseline", "V5 candidate")
    colors = ("#64748b", "#146c94")
    ax = axes[0]
    for position, (name, color) in enumerate(zip(names, colors, strict=True)):
        row = data["methods"][name]
        mean = 100 * row["accuracy"]
        low, high = [100 * value for value in row["accuracy_interval"]["ci95"]]
        ax.barh(position, mean, height=0.45, color=color)
        ax.errorbar(mean, position, xerr=[[mean - low], [high - mean]], color="#0f172a", capsize=5)
        ax.text(
            2,
            position,
            f"{row['correct']}/{data['questions']} · {mean:.1f}%",
            va="center",
            color="white",
            weight="bold",
        )
    ax.set_yticks([0, 1], labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Overall numeric accuracy (%)")
    ax.axvline(100 * data["accuracy_target"], color="#c2410c", linestyle="--", linewidth=1)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax = axes[1]
    ax.axis("off")
    ax.text(0, 0.95, "Accuracy and coverage together", weight="bold")
    for y, name, label in zip((0.72, 0.44), names, labels, strict=True):
        row = data["methods"][name]
        accepted = row["answered_accuracy"]
        answer_text = f"{100 * accepted:.1f}%" if accepted is not None else "undefined"
        ax.text(0, y, label, weight="bold")
        ax.text(
            0,
            y - 0.1,
            f"Coverage {100 * row['coverage']:.1f}% · answered accuracy {answer_text}",
            fontsize=10,
        )
        ax.text(
            0,
            y - 0.18,
            f"End-to-end p95 {row['p95_latency_ms'] / 1000:.2f} s",
            fontsize=10,
            color="#475569",
        )
    gain = data["paired_accuracy_gain"]
    low, high = [100 * value for value in gain["ci95"]]
    ax.text(0, 0.1, f"Paired gain: {100 * gain['mean']:+.1f} percentage points", weight="bold")
    ax.text(0, 0.02, f"95% interval [{low:+.1f}, {high:+.1f}]", fontsize=10)
    fig.text(
        0.06,
        0.09,
        "Abstentions count incorrect. Error bars: 95% document-bootstrap intervals; "
        "dashed line: fixed 80% project target.",
        fontsize=9,
        color="#475569",
    )
    fig.text(
        0.06,
        0.05,
        "Public arithmetic subset; source groups are not verified independent filings. "
        "No accepted-error or future-accuracy guarantee.",
        fontsize=9,
        color="#475569",
    )
    fig.subplots_adjust(left=0.16, right=0.98, top=0.8, bottom=0.23, wspace=0.28)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for path in outputs[1:]:
        metadata = {"Date": None} if path.suffix == ".svg" else None
        fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), metadata=metadata)
    plt.close(fig)
    write_json(data_path, data)
    print(stem.with_suffix(".svg"))


if __name__ == "__main__":
    main()
