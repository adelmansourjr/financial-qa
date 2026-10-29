"""Render the frozen paired comparison, keeping its public-training scope visible.

Requires matplotlib in the plotting environment; inference does not depend on it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot(report_path: Path, output: Path) -> None:
    report = json.loads(report_path.read_text())
    methods = report["methods"]
    names = ("baseline", "layout")
    colors = ("#64748b", "#4338ca")
    values = [100 * methods[name]["accuracy"] for name in names]
    bounds = [report["accuracy_intervals"][name]["ci95"] for name in names]
    errors = [
        [value - 100 * pair[0] for value, pair in zip(values, bounds, strict=True)],
        [100 * pair[1] - value for value, pair in zip(values, bounds, strict=True)],
    ]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.1), gridspec_kw={"width_ratios": [1.1, 1]})
    axes[0].bar([0, 1], values, width=0.55, color=colors, yerr=errors, capsize=5)
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["Frozen v3 model", "Layout-aware v4"],
        ylim=(0, 100),
        ylabel="Overall accuracy (%)",
        title="Wrong answers and abstentions both count",
    )
    for index, name in enumerate(names):
        axes[0].text(
            index,
            values[index] / 2,
            f"{methods[name]['correct']}/{methods[name]['evaluated']}\n{values[index]:.1f}%",
            ha="center",
            va="center",
            color="white",
            weight="bold",
        )
    gain = report["paired_accuracy_gain"]
    center = 100 * gain["mean"]
    low, high = [100 * value for value in gain["ci95"]]
    axes[1].axvline(0, color="#64748b", linestyle="--", linewidth=1)
    axes[1].errorbar(
        center,
        0,
        xerr=[[center - low], [high - center]],
        fmt="o",
        color=colors[1],
        capsize=7,
        markersize=9,
    )
    axes[1].set(
        yticks=[],
        ylim=(-1, 1),
        xlabel="Paired accuracy gain (percentage points)",
        title="Same 200 contexts, paired bootstrap",
    )
    axes[1].text(
        center,
        0.3,
        f"{center:+.1f} pp\n95% interval [{low:+.1f}, {high:+.1f}]",
        ha="center",
        va="center",
    )
    figure.suptitle(
        "Financial QA: a frozen comparison on new project-held-out contexts",
        fontsize=14,
        weight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.025,
        "Public-training arithmetic subset · context-level uncertainty · "
        "possible pretraining exposure · no new risk certificate",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.9))
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".svg", ".png"):
        figure.savefig(output.with_suffix(suffix), dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("reports/v4-summary.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/figures/financial-layout-v4"))
    args = parser.parse_args()
    plot(args.report, args.output)


if __name__ == "__main__":
    main()
