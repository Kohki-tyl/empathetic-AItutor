"""先頭800問と後半200問からscope_relation均等の固定選択表を作成する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ASSIGNMENTS = BASE_DIR / "assignments" / "problem_profile_assignments.jsonl"
DEFAULT_CORPUS_SELECTION = BASE_DIR / "assignments" / "corpus_120_selection.json"
DEFAULT_TEST_SELECTION = BASE_DIR / "assignments" / "test_120_selection.json"
DEFAULT_TEST_ASSIGNMENTS = (
    BASE_DIR / "assignments" / "test_problem_profile_assignments.jsonl"
)
DEFAULT_EXCLUDED_IDS = BASE_DIR / "questions" / "excluded_test_question_ids.json"
RELATIONS = ("mastered", "frontier", "one_step_beyond", "far_beyond")
SELECTION_POLICY_VERSION = "balanced-scope-30-v2"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_excluded_ids(path: Path) -> set[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("excluded_source_ids", value) if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise ValueError("excluded_source_idsは配列にしてください")
    return {str(item) for item in rows}


def select_balanced(
    rows: list[dict[str, Any]], *, start: int, end: int,
    per_relation: int, seed: int, excluded_ids: set[str] | None = None,
    required_source_ids: set[str] | None = None,
) -> dict[str, Any]:
    if not (0 <= start < end <= len(rows)):
        raise ValueError("選択範囲が対応表の範囲外です")
    excluded = excluded_ids or set()
    required = required_source_ids or set()
    partition = [row for row in rows[start:end] if str(row["source_id"]) not in excluded]
    review_excluded = {
        str(row["source_id"]) for row in partition
        if bool(row["curriculum_annotation"].get("requires_human_review"))
    }
    pool = [row for row in partition if str(row["source_id"]) not in review_excluded]
    pool_by_id = {str(row["source_id"]): row for row in pool}
    missing_required = required - set(pool_by_id)
    if missing_required:
        raise ValueError(f"必須問題が選択範囲にありません: {sorted(missing_required)}")
    selected: list[dict[str, Any]] = []
    available_counts = Counter(str(row["scope_relation"]) for row in pool)
    for relation_index, relation in enumerate(RELATIONS):
        candidates = [row for row in pool if row["scope_relation"] == relation]
        required_for_relation = [
            pool_by_id[source_id] for source_id in sorted(required)
            if pool_by_id[source_id]["scope_relation"] == relation
        ]
        remaining_count = per_relation - len(required_for_relation)
        candidates = [row for row in candidates if str(row["source_id"]) not in required]
        if remaining_count < 0 or len(candidates) < remaining_count:
            raise ValueError(
                f"{start}:{end}の{relation}は{len(candidates) + len(required_for_relation)}件で、"
                f"必要な{per_relation}件に不足しています"
            )
        chosen = random.Random(seed + relation_index).sample(candidates, remaining_count)
        selected.extend([*required_for_relation, *chosen])
    selected.sort(key=lambda row: int(row["order_index"]))
    knowledge_errors: list[str] = []
    for row in selected:
        audit = row.get("knowledge_boundary_audit") or {}
        if not bool(audit.get("relation_consistent")):
            knowledge_errors.append(f"{row['source_id']}: relation_consistent=false")
        if row["scope_relation"] == "mastered" and audit.get("not_in_prior_knowledge"):
            knowledge_errors.append(
                f"{row['source_id']}: masteredだが未習概念={audit['not_in_prior_knowledge']}"
            )
    if knowledge_errors:
        raise ValueError("選択問題の知識境界監査に失敗しました: " + "; ".join(knowledge_errors))
    records = [
        {
            "source_id": row["source_id"],
            "order_index": row["order_index"],
            "scope_relation": row["scope_relation"],
            "profile_id": row["profile_id"],
            "question_sha256": row["question_sha256"],
        }
        for row in selected
    ]
    digest = hashlib.sha256(
        json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "policy_version": SELECTION_POLICY_VERSION,
        "assignment_policy_version": rows[0]["policy_version"],
        "source_partition": {"start": start, "end_exclusive": end},
        "seed": seed,
        "per_scope_relation": per_relation,
        "selected_count": len(records),
        "available_counts_after_exclusion": dict(available_counts),
        "excluded_source_ids": sorted(excluded),
        "excluded_human_review_source_ids": sorted(review_excluded),
        "required_source_ids": sorted(required),
        "knowledge_boundary_audit": {
            "checked_records": len(records),
            "mastered_without_prior_support": 0,
            "relation_inconsistencies": 0,
            "passed": True,
        },
        "selection_sha256": digest,
        "records": records,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--corpus-output", type=Path, default=DEFAULT_CORPUS_SELECTION)
    parser.add_argument("--test-output", type=Path, default=DEFAULT_TEST_SELECTION)
    parser.add_argument("--test-assignments-output", type=Path, default=DEFAULT_TEST_ASSIGNMENTS)
    parser.add_argument("--excluded-test-ids", type=Path, default=DEFAULT_EXCLUDED_IDS)
    parser.add_argument("--train-pool-size", type=int, default=800)
    parser.add_argument("--per-relation", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(args.assignments)
    if len(rows) != 1000:
        raise ValueError(f"対応表は1000件必要です: {len(rows)}")
    if args.train_pool_size != 800:
        raise ValueError("本実験では先頭800件をコーパス、後半200件をテストに固定します")
    corpus = select_balanced(
        rows, start=0, end=800, per_relation=args.per_relation, seed=args.seed,
        required_source_ids={"math_train_0"},
    )
    test = select_balanced(
        rows, start=800, end=1000, per_relation=args.per_relation,
        seed=args.seed, excluded_ids=read_excluded_ids(args.excluded_test_ids),
    )
    write_json(args.corpus_output, corpus)
    write_json(args.test_output, test)
    write_jsonl(args.test_assignments_output, rows[800:])
    print(json.dumps({
        "corpus_output": str(args.corpus_output),
        "corpus_counts": Counter(row["scope_relation"] for row in corpus["records"]),
        "test_output": str(args.test_output),
        "test_counts": Counter(row["scope_relation"] for row in test["records"]),
        "test_assignment_rows": 200,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
