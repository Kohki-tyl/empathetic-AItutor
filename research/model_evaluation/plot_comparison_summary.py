from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "data" / "runs" / "baseline_vs_v3_cot_sft_summary.json"
DEFAULT_OUTPUT = BASE_DIR / "results" / "baseline_vs_v3_cot_sft" / "comparison.png"

AXES = [
    ("mathematical_accuracy", "Math accuracy"),
    ("error_diagnosis_recovery", "Error recovery"),
    ("instruction_completion", "Completion"),
    ("scaffolding", "Scaffolding"),
    ("emotional_support", "Emotional support"),
    ("emotion_recognition", "Emotion recognition"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="research比較summaryをPNG/SVGへ可視化する")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def add_bar_labels(axis: plt.Axes, bars, *, percent: bool = False) -> None:
    for bar in bars:
        value = bar.get_height()
        label = f"{value:.1f}%" if percent else f"{value:.2f}"
        axis.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main() -> None:
    args = parse_args()
    summary = json.loads(args.input.read_text(encoding="utf-8"))
    baseline = summary["conditions"]["baseline"]
    sft = summary["conditions"]["v3_cot_sft"]
    comparison = summary["comparisons"]["v3_cot_sft - baseline"]
    colors = {"baseline": "#7A8793", "sft": "#3274A1"}

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, panels = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    figure.suptitle("Baseline vs v3 CoT SFT — Research Evaluation", fontsize=18, fontweight="bold")

    # Overall score
    axis = panels[0, 0]
    overall = [baseline["overall_score_60"]["mean"], sft["overall_score_60"]["mean"]]
    bars = axis.bar(["Baseline", "v3 CoT SFT"], overall, color=[colors["baseline"], colors["sft"]], width=0.58)
    axis.set_title("A. Overall dialogue score")
    axis.set_ylabel("Mean score (0–60)")
    axis.set_ylim(0, 60)
    add_bar_labels(axis, bars)
    effect = comparison["overall_score_60"]
    axis.text(
        0.5, 0.92,
        f"Paired difference: +{effect['mean_difference']:.2f}\n95% CI [{effect['ci95'][0]:.2f}, {effect['ci95'][1]:.2f}]",
        transform=axis.transAxes, ha="center", va="top", fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#B8C2CC"},
    )

    # Six axes
    axis = panels[0, 1]
    positions = np.arange(len(AXES))
    width = 0.36
    baseline_axes = [baseline["axis_scores"][name]["mean"] for name, _ in AXES]
    sft_axes = [sft["axis_scores"][name]["mean"] for name, _ in AXES]
    left = axis.bar(positions - width / 2, baseline_axes, width, label="Baseline", color=colors["baseline"])
    right = axis.bar(positions + width / 2, sft_axes, width, label="v3 CoT SFT", color=colors["sft"])
    axis.set_title("B. Six evaluation axes")
    axis.set_ylabel("Mean score (0–10)")
    axis.set_ylim(0, 10)
    axis.set_xticks(positions, [label for _, label in AXES], rotation=24, ha="right")
    axis.legend(frameon=False)
    add_bar_labels(axis, left)
    add_bar_labels(axis, right)

    # Rates
    axis = panels[1, 0]
    rate_items = [
        ("dialogue_generation_success_rate", "Generation\nsuccess"),
        ("all_applicable_axes_at_least_8_rate", "All axes\n≥ 8"),
        ("critical_failure_rate", "Critical\nfailure"),
        ("instruction_completion_rate", "Instruction\ncompleted"),
        ("transfer_accuracy", "Near-transfer\naccuracy"),
    ]
    positions = np.arange(len(rate_items))
    baseline_rates = [baseline[name] * 100 for name, _ in rate_items]
    sft_rates = [sft[name] * 100 for name, _ in rate_items]
    left = axis.bar(positions - width / 2, baseline_rates, width, label="Baseline", color=colors["baseline"])
    right = axis.bar(positions + width / 2, sft_rates, width, label="v3 CoT SFT", color=colors["sft"])
    axis.set_title("C. Outcome rates")
    axis.set_ylabel("Rate (%)")
    axis.set_ylim(0, 110)
    axis.set_xticks(positions, [label for _, label in rate_items])
    axis.legend(frameon=False)
    add_bar_labels(axis, left, percent=True)
    add_bar_labels(axis, right, percent=True)

    # Paired effects normalized to percentage points of each scale.
    axis = panels[1, 1]
    effects = [
        ("Overall", comparison["overall_score_60"], 60),
        *[(label, comparison["axes"][name], 10) for name, label in AXES],
        ("Critical failure", comparison["critical_failure_rate"], 1),
        ("Instruction completed", comparison["instruction_completion_rate"], 1),
        ("Near-transfer", comparison["transfer_accuracy"], 1),
    ]
    labels = [item[0] for item in effects]
    means = np.array([item[1]["mean_difference"] / item[2] * 100 for item in effects])
    lows = np.array([item[1]["ci95"][0] / item[2] * 100 for item in effects])
    highs = np.array([item[1]["ci95"][1] / item[2] * 100 for item in effects])
    y = np.arange(len(effects))
    axis.axvline(0, color="#333333", linewidth=1)
    axis.errorbar(
        means, y, xerr=np.vstack([means - lows, highs - means]),
        fmt="o", color=colors["sft"], ecolor="#5B8DB8", capsize=3,
    )
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_title("D. Paired v3 CoT SFT − baseline effects")
    axis.set_xlabel("Difference (percentage points of metric scale)\nRight favors SFT except critical failure")
    axis.grid(axis="y", visible=False)

    footer = (
        "Planned cases: 100 per condition. Evaluated: baseline 99, v3 CoT SFT 100. "
        "Error bars are paired-bootstrap 95% intervals."
    )
    figure.text(0.5, -0.015, footer, ha="center", fontsize=9, color="#4A5568")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(args.output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    print(f"PNG: {args.output.resolve()}")
    print(f"SVG: {args.output.with_suffix('.svg').resolve()}")


if __name__ == "__main__":
    main()
