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
DEFAULT_OUTPUT = BASE_DIR / "results" / "baseline_vs_v3_cot_sft" / "comparison_abc_ja.png"

AXES = [
    ("mathematical_accuracy", "数学的正確性\nMath accuracy"),
    ("error_diagnosis_recovery", "誤り診断と回復\nError recovery"),
    ("instruction_completion", "指導完了判定\nCompletion"),
    ("scaffolding", "足場かけ\nScaffolding"),
    ("emotional_support", "情緒的支援\nEmotional support"),
    ("emotion_recognition", "感情把握\nEmotion recognition"),
]


def labels(axis: plt.Axes, bars, *, percent: bool = False) -> None:
    for bar in bars:
        value = bar.get_height()
        axis.annotate(
            f"{value:.1f}%" if percent else f"{value:.2f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="A・B・Cのみの日本語併記比較図を作る")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = json.loads(args.input.read_text(encoding="utf-8"))
    baseline = summary["conditions"]["baseline"]
    sft = summary["conditions"]["v3_cot_sft"]
    effect = summary["comparisons"]["v3_cot_sft - baseline"]["overall_score_60"]

    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams["font.family"] = "Yu Gothic"
    figure, panels = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
    figure.suptitle("Baseline と v3 CoT SFT のresearch評価比較", fontsize=19, fontweight="bold")
    colors = ["#7A8793", "#3274A1"]
    names = ["Baseline\nベースライン", "v3 CoT SFT"]

    axis = panels[0]
    values = [baseline["overall_score_60"]["mean"], sft["overall_score_60"]["mean"]]
    bars = axis.bar(names, values, color=colors, width=0.58)
    axis.set_title("A. 対話全体得点\nOverall dialogue score")
    axis.set_ylabel("平均得点（0–60）")
    axis.set_ylim(0, 60)
    labels(axis, bars)
    axis.text(
        0.5, 0.91,
        f"対応差 +{effect['mean_difference']:.2f}\n95% CI [{effect['ci95'][0]:.2f}, {effect['ci95'][1]:.2f}]",
        transform=axis.transAxes, ha="center", va="top", fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#B8C2CC"},
    )

    axis = panels[1]
    x = np.arange(len(AXES))
    width = 0.36
    base_values = [baseline["axis_scores"][key]["mean"] for key, _ in AXES]
    sft_values = [sft["axis_scores"][key]["mean"] for key, _ in AXES]
    left = axis.bar(x - width / 2, base_values, width, label="Baseline", color=colors[0])
    right = axis.bar(x + width / 2, sft_values, width, label="v3 CoT SFT", color=colors[1])
    axis.set_title("B. 6評価軸\nSix evaluation axes")
    axis.set_ylabel("平均得点（0–10）")
    axis.set_ylim(0, 10)
    axis.set_xticks(x, [name for _, name in AXES], rotation=25, ha="right")
    axis.legend(frameon=False)
    labels(axis, left)
    labels(axis, right)

    axis = panels[2]
    rates = [
        ("dialogue_generation_success_rate", "対話生成成功率\nGeneration success"),
        ("all_applicable_axes_at_least_8_rate", "全適用軸8点以上\nAll axes ≥ 8"),
        ("critical_failure_rate", "重大失敗率\nCritical failure"),
        ("instruction_completion_rate", "指導完了率\nInstruction completed"),
        ("transfer_accuracy", "類似問題正答率\nNear-transfer accuracy"),
    ]
    x = np.arange(len(rates))
    base_values = [baseline[key] * 100 for key, _ in rates]
    sft_values = [sft[key] * 100 for key, _ in rates]
    left = axis.bar(x - width / 2, base_values, width, label="Baseline", color=colors[0])
    right = axis.bar(x + width / 2, sft_values, width, label="v3 CoT SFT", color=colors[1])
    axis.set_title("C. 主要な割合指標\nOutcome rates")
    axis.set_ylabel("割合（%）")
    axis.set_ylim(0, 110)
    axis.set_xticks(x, [name for _, name in rates], rotation=22, ha="right")
    axis.legend(frameon=False)
    labels(axis, left, percent=True)
    labels(axis, right, percent=True)

    figure.text(
        0.5, -0.03,
        "予定件数：各100件／評価数：Baseline 99件、v3 CoT SFT 100件",
        ha="center", fontsize=10, color="#4A5568",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(args.output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
