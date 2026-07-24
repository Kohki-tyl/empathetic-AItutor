"""v2の原問題・近接転移問題を対応検証し、疑義のあるペアを除外する。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
SHARED_DIR = BASE_DIR.parent / "shared"
SUSPICIOUS_SOLUTION_PATTERNS = {
    "solution_rejects_structure": r"元の問題と同じ構造にな(?:らない|っていない)",
    "solution_changes_setting": r"(?:設定|数値設定|問題文).{0,30}(?:見直す|見直し|不適切)",
    "solution_rewrites_question": r"(?:修正後|最終的)の(?:類似)?問題|問題(?:文)?の式.{0,30}ではなく",
    "solution_changes_requested_expression": r"求める式を.{0,30}(?:変更|直す)",
    "solution_admits_invalid_problem": r"問題設定自体が不適切|問題の形式と合わない",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=SHARED_DIR / "questions" / "test_math_questions.jsonl")
    parser.add_argument("--similar-questions", type=Path, default=SHARED_DIR / "questions" / "similar_test_math_questions.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-valid", type=int, default=1)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def pair_issues(original: dict[str, Any], similar: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for field in ("translated_question", "translated_solution"):
        if not isinstance(original.get(field), str) or not original[field].strip():
            issues.append(f"missing_original_{field}")
    for field in ("similar_question", "similar_solution"):
        if not isinstance(similar.get(field), str) or not similar[field].strip():
            issues.append(f"missing_{field}")
    if original.get("translated_question") != similar.get("original_question"):
        issues.append("original_question_mismatch")
    solution = str(similar.get("similar_solution", ""))
    if not re.search(r"\\boxed\s*\{", solution):
        issues.append("solution_has_no_boxed_answer")
    for issue, pattern in SUSPICIOUS_SOLUTION_PATTERNS.items():
        if re.search(pattern, solution, flags=re.DOTALL):
            issues.append(issue)
    return sorted(set(issues))


def main() -> None:
    args = parse_args()
    originals = read_jsonl(args.questions)
    similars = read_jsonl(args.similar_questions)
    original_by_id = {str(row.get("id") or row.get("source_id")): row for row in originals}
    similar_by_id = {str(row.get("source_id") or row.get("id")): row for row in similars}
    all_ids = sorted(set(original_by_id) | set(similar_by_id))
    valid_ids: list[str] = []
    excluded: list[dict[str, Any]] = []
    for source_id in all_ids:
        original = original_by_id.get(source_id)
        similar = similar_by_id.get(source_id)
        if original is None or similar is None:
            excluded.append({"source_id": source_id, "issues": ["unpaired_record"]})
            continue
        issues = pair_issues(original, similar)
        if issues:
            excluded.append({"source_id": source_id, "issues": issues})
        else:
            valid_ids.append(source_id)
    if len(valid_ids) < args.minimum_valid:
        raise SystemExit(f"有効な問題ペアが不足しています: {len(valid_ids)} < {args.minimum_valid}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "questions.jsonl", [original_by_id[key] for key in valid_ids])
    write_jsonl(args.output_dir / "similar_questions.jsonl", [similar_by_id[key] for key in valid_ids])
    report = {
        "input_questions": str(args.questions.resolve()),
        "input_similar_questions": str(args.similar_questions.resolve()),
        "input_count": len(all_ids),
        "valid_count": len(valid_ids),
        "excluded_count": len(excluded),
        "excluded": excluded,
        "validation_scope": "structural checks and explicit inconsistency markers; not a proof of mathematical correctness",
    }
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Validated {len(valid_ids)}/{len(all_ids)} pairs; excluded {len(excluded)}")


if __name__ == "__main__":
    main()
