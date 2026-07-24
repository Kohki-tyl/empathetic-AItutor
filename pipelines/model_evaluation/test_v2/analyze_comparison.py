"""再構築版v2のBase/SFT結果を対応比較し、欠測を明示する。"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE_DIR / "data" / "rebuilt" / "base_swallow8b" / "evaluated_results.jsonl")
    parser.add_argument("--sft", type=Path, default=BASE_DIR / "data" / "rebuilt" / "v2_cot_sft_swallow" / "evaluated_results.jsonl")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "data" / "rebuilt" / "comparison_report.md")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def nested(row: dict[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def judge_complete(row: dict[str, Any]) -> bool:
    return all(isinstance(value, dict) and not value.get("error") for value in (
        row.get("math_judge"), row.get("empathic_instruction_evaluation"),
        row.get("mathematical_instruction_evaluation"), row.get("student_realism_evaluation"),
    ))


def summarize(rows: list[dict[str, Any]], getter: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    values = [getter(row) for row in rows]
    numeric = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    booleans = [value for value in values if isinstance(value, bool)]
    if booleans:
        return {"n": len(booleans), "missing": len(rows) - len(booleans), "value": sum(booleans) / len(booleans)}
    return {
        "n": len(numeric), "missing": len(rows) - len(numeric),
        "value": statistics.fmean(numeric) if numeric else None,
    }


def fmt(summary: dict[str, Any], percent: bool = False) -> str:
    value = summary["value"]
    rendered = "NA" if value is None else (f"{value * 100:.1f}%" if percent else f"{value:.3f}")
    return f"{rendered} (n={summary['n']}, missing={summary['missing']})"


def main() -> None:
    args = parse_args()
    base_all = read_jsonl(args.base)
    sft_all = read_jsonl(args.sft)
    base_by_id = {str(row["run_id"]): row for row in base_all}
    sft_by_id = {str(row["run_id"]): row for row in sft_all}
    paired_ids = sorted(set(base_by_id) & set(sft_by_id))
    base = [base_by_id[key] for key in paired_ids]
    sft = [sft_by_id[key] for key in paired_ids]
    metrics: list[tuple[str, Callable[[dict[str, Any]], Any], bool]] = [
        ("Phase 1 completion", lambda row: row.get("phase1_is_completed"), True),
        ("Phase 1 turns", lambda row: row.get("phase1_turns"), False),
        ("Phase 2 accuracy", lambda row: row.get("phase2_is_correct"), True),
        ("Empathic instruction total", lambda row: nested(row, "empathic_instruction_evaluation", "total_score"), False),
        ("Mathematical instruction total", lambda row: nested(row, "mathematical_instruction_evaluation", "total_score"), False),
        ("Student realism", lambda row: nested(row, "student_realism_evaluation", "realism_score"), False),
    ]
    report: dict[str, Any] = {
        "base_input_count": len(base_all), "sft_input_count": len(sft_all),
        "paired_count": len(paired_ids),
        "base_generation_errors": sum(bool(row.get("generation_error")) for row in base),
        "sft_generation_errors": sum(bool(row.get("generation_error")) for row in sft),
        "base_incomplete_judge_runs": sum(not judge_complete(row) for row in base),
        "sft_incomplete_judge_runs": sum(not judge_complete(row) for row in sft),
        "metrics": {},
    }
    lines = [
        "# Rebuilt test v2 comparison", "",
        f"対応ペア数: {len(paired_ids)}（Base入力 {len(base_all)}、SFT入力 {len(sft_all)}）", "",
        f"生成エラー: Base {report['base_generation_errors']}、SFT {report['sft_generation_errors']}", "",
        f"Judge未完了: Base {report['base_incomplete_judge_runs']}、SFT {report['sft_incomplete_judge_runs']}", "",
        "| Metric | Base | v2 CoT-SFT |", "| --- | ---: | ---: |",
    ]
    for name, getter, percent in metrics:
        base_summary = summarize(base, getter)
        sft_summary = summarize(sft, getter)
        report["metrics"][name] = {"base": base_summary, "sft": sft_summary}
        lines.append(f"| {name} | {fmt(base_summary, percent)} | {fmt(sft_summary, percent)} |")
    lines.extend(["", "欠測値は0として扱わず、各指標の`missing`へ計上した。", ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Comparison report: {args.output}")


if __name__ == "__main__":
    main()
