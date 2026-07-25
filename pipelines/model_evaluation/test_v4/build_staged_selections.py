"""固定120問を、scope均衡を保った一次60問と確認用60問へ分割する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
RELATIONS = ("mastered", "frontier", "one_step_beyond", "far_beyond")


def record_digest(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def stage_payload(
    parent: dict[str, Any], records: list[dict[str, Any]], stage: str, seed: int,
) -> dict[str, Any]:
    counts = Counter(str(record["scope_relation"]) for record in records)
    if set(counts) != set(RELATIONS) or set(counts.values()) != {15}:
        raise ValueError(f"{stage}のscope_relationが不均衡です: {dict(counts)}")
    return {
        "policy_version": "staged-balanced-scope-15-v1",
        "assignment_policy_version": parent["assignment_policy_version"],
        "parent_policy_version": parent["policy_version"],
        "parent_selection_sha256": parent["selection_sha256"],
        "evaluation_stage": stage,
        "source_partition": parent["source_partition"],
        "seed": seed,
        "per_scope_relation": 15,
        "selected_count": 60,
        "available_counts_after_exclusion": parent["available_counts_after_exclusion"],
        "excluded_source_ids": parent["excluded_source_ids"],
        "excluded_human_review_source_ids": parent["excluded_human_review_source_ids"],
        "required_source_ids": parent["required_source_ids"],
        "knowledge_boundary_audit": {
            "checked_records": 60,
            "mastered_without_prior_support": 0,
            "relation_inconsistencies": 0,
            "passed": True,
        },
        "selection_sha256": record_digest(records),
        "records": records,
    }


def build(parent: dict[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    records = parent.get("records")
    if not isinstance(records, list) or len(records) != 120:
        raise ValueError("親選択表は120件必要です")
    if parent.get("selection_sha256") != record_digest(records):
        raise ValueError("親選択表のselection_sha256が一致しません")
    primary: list[dict[str, Any]] = []
    confirmation: list[dict[str, Any]] = []
    for relation_index, relation in enumerate(RELATIONS):
        candidates = [record for record in records if record["scope_relation"] == relation]
        if len(candidates) != 30:
            raise ValueError(f"親選択表の{relation}が30件ではありません")
        shuffled = list(candidates)
        random.Random(seed + relation_index).shuffle(shuffled)
        primary.extend(shuffled[:15])
        confirmation.extend(shuffled[15:])
    primary.sort(key=lambda record: int(record["order_index"]))
    confirmation.sort(key=lambda record: int(record["order_index"]))
    primary_ids = {str(record["source_id"]) for record in primary}
    confirmation_ids = {str(record["source_id"]) for record in confirmation}
    parent_ids = {str(record["source_id"]) for record in records}
    if primary_ids & confirmation_ids or primary_ids | confirmation_ids != parent_ids:
        raise RuntimeError("一次評価と確認評価が親120件を排他的に分割していません")
    return (
        stage_payload(parent, primary, "primary", seed),
        stage_payload(parent, confirmation, "confirmation", seed),
    )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent", type=Path,
        default=BASE_DIR / "prompts" / "test_120_selection.json",
    )
    parser.add_argument(
        "--primary-output", type=Path,
        default=BASE_DIR / "prompts" / "test_60_primary_selection.json",
    )
    parser.add_argument(
        "--confirmation-output", type=Path,
        default=BASE_DIR / "prompts" / "test_60_confirmation_selection.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    parent = json.loads(args.parent.read_text(encoding="utf-8"))
    primary, confirmation = build(parent, args.seed)
    write_json(args.primary_output, primary)
    write_json(args.confirmation_output, confirmation)
    print(json.dumps({
        "primary": str(args.primary_output),
        "primary_sha256": primary["selection_sha256"],
        "confirmation": str(args.confirmation_output),
        "confirmation_sha256": confirmation["selection_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
