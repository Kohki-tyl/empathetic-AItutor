from __future__ import annotations

import argparse
from pathlib import Path

from common import read_jsonl, sha256_file, write_json, write_jsonl, utc_now


def main() -> None:
    parser = argparse.ArgumentParser(description="評価shardをcase_id順に検証・結合する")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected = [str(row["case_id"]) for row in read_jsonl(args.selection)]
    rows = [row for path in args.input for row in read_jsonl(path)]
    by_id = {str(row["case_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("shard間に重複case_idがあります")
    missing = [case_id for case_id in expected if case_id not in by_id]
    extra = sorted(set(by_id) - set(expected))
    if missing or extra:
        raise ValueError(f"欠番または範囲外caseがあります: missing={missing[:5]}, extra={extra[:5]}")
    ordered = [by_id[case_id] for case_id in expected]
    if any(row.get("evaluation_status") != "evaluated" for row in ordered):
        raise ValueError("未評価またはJudge失敗のcaseがあります")

    write_jsonl(args.output, ordered)
    write_json(args.output.with_suffix(".manifest.json"), {
        "schema_version": "combined-teacher-dialogue-evaluation-manifest-v1",
        "created_at_utc": utc_now(),
        "selection_sha256": sha256_file(args.selection),
        "input_shards": [str(path.resolve()) for path in args.input],
        "planned_cases": len(expected),
        "evaluated_cases": len(ordered),
        "unique_cases": len(by_id),
        "output_sha256": sha256_file(args.output),
    })
    print(f"結合完了: {len(ordered)}件 -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
