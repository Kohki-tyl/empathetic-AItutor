from __future__ import annotations

import argparse
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import AXES, axis_scores, describe, read_jsonl, rounded, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="教師条件ごとの主要評価と対応比較を集計する")
    parser.add_argument("--input", type=Path, action="append", required=True, help="複数回指定可能")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("空のpercentile")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def paired_bootstrap(differences: list[float], samples: int, seed: int) -> dict[str, Any]:
    if not differences:
        return {"n": 0, "mean_difference": None, "ci95": [None, None]}
    rng = random.Random(seed)
    n = len(differences)
    draws = [statistics.mean(differences[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    return {
        "n": n,
        "mean_difference": rounded(statistics.mean(differences)),
        "ci95": [rounded(percentile(draws, 0.025)), rounded(percentile(draws, 0.975))],
    }


def condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    planned = len(rows)
    generated = [row for row in rows if row.get("dialogue_generation_succeeded") is True]
    evaluated = [row for row in rows if row.get("evaluation_status") == "evaluated"]
    overall = [row["evaluation"]["overall_score_60"] for row in evaluated]
    axis_values: dict[str, list[float | None]] = {name: [] for name in AXES}
    for row in evaluated:
        scores = axis_scores(row["evaluation"])
        for name in AXES:
            axis_values[name].append(scores[name])
    threshold_passes = sum(
        all(score >= 8 for score in axis_scores(row["evaluation"]).values() if score is not None)
        for row in evaluated
    )
    transfer_judged = [
        row for row in evaluated if isinstance(row.get("transfer_evaluation"), dict)
        and isinstance(row["transfer_evaluation"].get("is_correct"), bool)
    ]
    return {
        "planned_cases": planned,
        "dialogue_generation_successes": len(generated),
        "dialogue_generation_success_rate": rounded(len(generated) / planned if planned else None),
        "evaluated_cases": len(evaluated),
        "all_applicable_axes_at_least_8": threshold_passes,
        "all_applicable_axes_at_least_8_rate": rounded(
            threshold_passes / len(evaluated) if evaluated else None
        ),
        "overall_score_60": describe(overall),
        "axis_scores": {name: {**describe(values), "na_count": sum(value is None for value in values)}
                        for name, values in axis_values.items()},
        "instruction_group_mean": describe(row["evaluation"]["instruction_group_mean"] for row in evaluated),
        "empathy_group_mean": describe(row["evaluation"]["empathy_group_mean"] for row in evaluated),
        "critical_failures": sum(row["evaluation"]["critical_failure"] is True for row in evaluated),
        "critical_failure_rate": rounded(
            sum(row["evaluation"]["critical_failure"] is True for row in evaluated) / len(evaluated)
            if evaluated else None
        ),
        "instruction_completed": sum(row["evaluation"]["instruction_completed"] is True for row in evaluated),
        "instruction_completion_rate": rounded(
            sum(row["evaluation"]["instruction_completed"] is True for row in evaluated) / len(evaluated)
            if evaluated else None
        ),
        "max_turn_reached": sum(row.get("termination_reason") == "max_turns" for row in evaluated),
        "transfer_judged": len(transfer_judged),
        "transfer_correct": sum(row["transfer_evaluation"]["is_correct"] is True for row in transfer_judged),
        "transfer_accuracy": rounded(
            sum(row["transfer_evaluation"]["is_correct"] is True for row in transfer_judged) / len(transfer_judged)
            if transfer_judged else None
        ),
    }


def comparison(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    all_a = {str(row["case_id"]): row for row in rows_a}
    all_b = {str(row["case_id"]): row for row in rows_b}
    common_planned = sorted(set(all_a) & set(all_b))
    evaluated_a = {str(row["case_id"]): row for row in rows_a if row.get("evaluation_status") == "evaluated"}
    evaluated_b = {str(row["case_id"]): row for row in rows_b if row.get("evaluation_status") == "evaluated"}
    common = sorted(set(evaluated_a) & set(evaluated_b))
    mismatched_initial = [
        key for key in common_planned
        if all_a[key].get("initial_response_sha256") != all_b[key].get("initial_response_sha256")
    ]
    if mismatched_initial:
        raise ValueError(f"教師条件間で初回生徒発話が一致しません: {mismatched_initial[:5]}")
    result: dict[str, Any] = {
        "common_planned_cases": len(common_planned),
        "common_evaluated_cases": len(common),
        "dialogue_generation_success_rate": paired_bootstrap([
            float(all_b[key].get("dialogue_generation_succeeded") is True)
            - float(all_a[key].get("dialogue_generation_succeeded") is True)
            for key in common_planned
        ], samples, seed + 20),
        "overall_score_60": paired_bootstrap([
            float(evaluated_b[key]["evaluation"]["overall_score_60"])
            - float(evaluated_a[key]["evaluation"]["overall_score_60"])
            for key in common
        ], samples, seed),
        "axes": {},
    }
    result["critical_failure_rate"] = paired_bootstrap([
        float(evaluated_b[key]["evaluation"]["critical_failure"] is True)
        - float(evaluated_a[key]["evaluation"]["critical_failure"] is True)
        for key in common
    ], samples, seed + 21)
    result["instruction_completion_rate"] = paired_bootstrap([
        float(evaluated_b[key]["evaluation"]["instruction_completed"] is True)
        - float(evaluated_a[key]["evaluation"]["instruction_completed"] is True)
        for key in common
    ], samples, seed + 22)
    transfer_common = [
        key for key in common
        if isinstance(evaluated_a[key].get("transfer_evaluation"), dict)
        and isinstance(evaluated_b[key].get("transfer_evaluation"), dict)
        and isinstance(evaluated_a[key]["transfer_evaluation"].get("is_correct"), bool)
        and isinstance(evaluated_b[key]["transfer_evaluation"].get("is_correct"), bool)
    ]
    result["transfer_accuracy"] = paired_bootstrap([
        float(evaluated_b[key]["transfer_evaluation"]["is_correct"] is True)
        - float(evaluated_a[key]["transfer_evaluation"]["is_correct"] is True)
        for key in transfer_common
    ], samples, seed + 23)
    for offset, name in enumerate(AXES):
        differences: list[float] = []
        for key in common:
            a = axis_scores(evaluated_a[key]["evaluation"])[name]
            b = axis_scores(evaluated_b[key]["evaluation"])[name]
            if a is not None and b is not None:
                differences.append(b - a)
        result["axes"][name] = paired_bootstrap(differences, samples, seed + offset + 1)
    return result


def markdown_report(summary: dict[str, Any]) -> str:
    lines = ["# 教師モデル評価サマリー", "", "## 条件別主要結果", "",
             "| 条件 | 予定 | 対話生成成功率 | 評価数 | 全体得点平均 / 60 | 全適用軸8点以上 | 重大失敗率 | 指導完了率 | 類似問題正答率 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for condition, value in summary["conditions"].items():
        pct = lambda x: "NA" if x is None else f"{x * 100:.1f}%"
        mean = value["overall_score_60"]["mean"]
        lines.append(
            f"| {condition} | {value['planned_cases']} | {pct(value['dialogue_generation_success_rate'])} | "
            f"{value['evaluated_cases']} | {'NA' if mean is None else f'{mean:.2f}'} | "
            f"{pct(value['all_applicable_axes_at_least_8_rate'])} | "
            f"{pct(value['critical_failure_rate'])} | {pct(value['instruction_completion_rate'])} | "
            f"{pct(value['transfer_accuracy'])} |"
        )
    for condition, value in summary["conditions"].items():
        lines.extend(["", f"## {condition} の6軸", "", "| 評価軸 | n | 平均 | 中央値 | NA |", "| --- | ---: | ---: | ---: | ---: |"])
        for name in AXES:
            item = value["axis_scores"][name]
            lines.append(f"| {name} | {item['n']} | {item['mean']} | {item['median']} | {item['na_count']} |")
    if summary["comparisons"]:
        lines.extend(["", "## 対応比較", ""])
        for name, value in summary["comparisons"].items():
            item = value["overall_score_60"]
            generation = value["dialogue_generation_success_rate"]
            lines.append(
                f"- {name}: 計画共通{value['common_planned_cases']}件、生成成功率差={generation['mean_difference']} "
                f"(95%区間={generation['ci95']})。評価共通{value['common_evaluated_cases']}件、"
                f"全体得点差={item['mean_difference']} (95%区間={item['ci95']})"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in args.input:
        for row in read_jsonl(path.resolve()):
            by_condition[str(row["condition"])].append(row)
    for condition, rows in by_condition.items():
        case_ids = [str(row["case_id"]) for row in rows]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(f"条件{condition}にcase_id重複があります")
    conditions = sorted(by_condition)
    summary: dict[str, Any] = {
        "schema_version": "teacher-model-summary-v1",
        "conditions": {name: condition_summary(by_condition[name]) for name in conditions},
        "comparisons": {},
    }
    for left_index, left in enumerate(conditions):
        for right_index in range(left_index + 1, len(conditions)):
            right = conditions[right_index]
            summary["comparisons"][f"{right} - {left}"] = comparison(
                by_condition[left], by_condition[right], args.bootstrap_samples,
                args.seed + left_index * 100 + right_index,
            )
    write_json(args.output_json.resolve(), summary)
    markdown = markdown_report(summary)
    args.output_markdown.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.resolve().write_text(markdown, encoding="utf-8")
    print(f"集計JSON: {args.output_json.resolve()}")
    print(f"集計Markdown: {args.output_markdown.resolve()}")


if __name__ == "__main__":
    main()
