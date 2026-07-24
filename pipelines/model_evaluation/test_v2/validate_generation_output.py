"""Judge開始前に生成結果の完全性と漏洩不在を検証する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LEAK_MARKERS = ('"state_before"', '"state_after"', '"recent_dialogue"', '<analysis>', '<final>')


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    report = json.loads(args.validation_report.read_text(encoding="utf-8"))
    expected = int(report["valid_count"])
    issues: list[str] = []
    run_ids = [str(row.get("run_id", "")) for row in rows]
    if len(rows) != expected:
        issues.append(f"record count {len(rows)} != expected {expected}")
    if len(set(run_ids)) != len(run_ids) or any(not run_id for run_id in run_ids):
        issues.append("run_id is missing or duplicated")
    failed = [str(row.get("run_id")) for row in rows if row.get("generation_error")]
    zero_turn = [str(row.get("run_id")) for row in rows if int(row.get("phase1_turns", 0)) < 1]
    leaked: list[str] = []
    for row in rows:
        for turn in row.get("dialogue_log", []):
            content = str(turn.get("content", ""))
            if any(marker in content.lower() for marker in LEAK_MARKERS):
                leaked.append(str(row.get("run_id")))
                break
    if failed:
        issues.append(f"generation_error in {len(failed)} runs")
    if zero_turn:
        issues.append(f"zero Phase 1 turns in {len(zero_turn)} runs")
    if leaked:
        issues.append(f"hidden JSON/tag leakage in {len(leaked)} runs")
    if issues:
        raise SystemExit("Generation validation failed: " + "; ".join(issues))
    print(f"Generation validation passed: {len(rows)} records")


if __name__ == "__main__":
    main()
