from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any

from common import read_json, read_jsonl, sha256_file, stable_fingerprint, write_json


BASE_DIR = Path(__file__).resolve().parent
RELATIONS = ("mastered", "frontier", "one_step_beyond", "far_beyond")


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return resolved.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="翻訳済み問題の末尾200問からscope均衡100問を選択する")
    parser.add_argument(
        "--questions", type=Path,
        default=BASE_DIR / "assets/test_math_questions.jsonl",
    )
    parser.add_argument(
        "--assignments", type=Path,
        default=BASE_DIR / "assets/problem_profile_assignments.jsonl",
    )
    parser.add_argument(
        "--exclusion-source", type=Path,
        default=BASE_DIR / "assets/excluded_test_question_ids.json",
    )
    parser.add_argument("--tail-size", type=int, default=200)
    parser.add_argument("--per-scope", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=BASE_DIR / "selections" / "evaluation_100.json")
    return parser.parse_args()


def build_selection(*, questions_path: Path, assignments_path: Path, exclusion_path: Path,
                    tail_size: int, per_scope: int, seed: int) -> dict[str, Any]:
    if tail_size < 1 or per_scope < 1:
        raise ValueError("tail-sizeとper-scopeは1以上にしてください")
    questions = read_jsonl(questions_path.resolve())
    if len(questions) < tail_size:
        raise ValueError(f"翻訳済み問題は{len(questions)}件しかなく、末尾{tail_size}件を取得できません")
    tail = questions[-tail_size:]
    tail_ids = [str(row["id"]) for row in tail]
    if len(tail_ids) != len(set(tail_ids)):
        raise ValueError("末尾問題集合にid重複があります")
    assignments = {str(row["source_id"]): row for row in read_jsonl(assignments_path.resolve())}
    exclusion_source = read_json(exclusion_path.resolve())
    excluded = {str(value) for value in exclusion_source.get("excluded_source_ids", [])}

    candidates: dict[str, list[dict[str, Any]]] = {relation: [] for relation in RELATIONS}
    missing_assignments: list[str] = []
    for position, source_id in enumerate(tail_ids):
        assignment = assignments.get(source_id)
        if assignment is None:
            missing_assignments.append(source_id)
            continue
        if source_id in excluded:
            continue
        relation = str(assignment.get("scope_relation"))
        if relation not in candidates:
            raise ValueError(f"{source_id}: scope_relationが不正です: {relation}")
        candidates[relation].append({
            "source_id": source_id,
            "order_index": int(assignment.get("order_index", position)),
            "scope_relation": relation,
            "profile_id": str(assignment["profile_id"]),
            "question_sha256": assignment["question_sha256"],
        })
    if missing_assignments:
        raise ValueError(f"割当がない問題があります: {missing_assignments[:10]}")

    selected: list[dict[str, Any]] = []
    for relation_index, relation in enumerate(RELATIONS):
        pool = list(candidates[relation])
        if len(pool) < per_scope:
            raise ValueError(f"{relation}は{len(pool)}件しかなく、{per_scope}件を選べません")
        random.Random(seed + relation_index).shuffle(pool)
        selected.extend(pool[:per_scope])
    selected.sort(key=lambda row: int(row["order_index"]))
    counts = Counter(str(row["scope_relation"]) for row in selected)
    expected = {relation: per_scope for relation in RELATIONS}
    if dict(counts) != expected:
        raise RuntimeError(f"scope均衡に失敗しました: {dict(counts)}")
    source_ids = [str(row["source_id"]) for row in selected]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("選択結果にsource_id重複があります")

    return {
        "policy_version": f"research-tail{tail_size}-balanced-scope-{per_scope}-v1",
        "source_policy": "translated_questions_last_n_in_file_order",
        "source_file": portable_path(questions_path),
        "source_file_sha256": sha256_file(questions_path.resolve()),
        "assignments_file": portable_path(assignments_path),
        "assignments_file_sha256": sha256_file(assignments_path.resolve()),
        "exclusion_source": portable_path(exclusion_path),
        "exclusion_source_sha256": sha256_file(exclusion_path.resolve()),
        "tail_size": tail_size,
        "tail_first_source_id": tail_ids[0],
        "tail_last_source_id": tail_ids[-1],
        "seed": seed,
        "per_scope_relation": per_scope,
        "selected_count": len(selected),
        "available_counts_after_exclusion": {
            relation: len(candidates[relation]) for relation in RELATIONS
        },
        "excluded_source_ids": sorted(excluded & set(tail_ids)),
        "profile_counts": dict(sorted(Counter(str(row["profile_id"]) for row in selected).items())),
        "selection_sha256": stable_fingerprint(selected),
        "records": selected,
    }


def main() -> None:
    args = parse_args()
    selection = build_selection(
        questions_path=args.questions,
        assignments_path=args.assignments,
        exclusion_path=args.exclusion_source,
        tail_size=args.tail_size,
        per_scope=args.per_scope,
        seed=args.seed,
    )
    write_json(args.output.resolve(), selection)
    print(f"選択: {selection['selected_count']}件")
    print(f"範囲: {selection['tail_first_source_id']} .. {selection['tail_last_source_id']}")
    print(f"出力: {args.output.resolve()}")


if __name__ == "__main__":
    main()
