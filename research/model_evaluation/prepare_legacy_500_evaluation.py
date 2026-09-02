from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import read_jsonl, sha256_file, sha256_text, write_json, write_jsonl


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
DEFAULT_CORPUS = REPO_ROOT / "pipelines" / "corpus_creation" / "500_empathetic_dialogues.jsonl"
DEFAULT_QUESTIONS = REPO_ROOT / "pipelines" / "corpus_creation" / "questions" / "translated_1000_math.jsonl"
DEFAULT_OUTPUT = BASE_DIR / "selections" / "legacy_500_dialogues.jsonl"
DEFAULT_MANIFEST = BASE_DIR / "selections" / "legacy_500.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="旧500対話をresearchの可視発話評価形式へ変換する"
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def convert_dialogue(
    record: dict[str, Any], question: dict[str, Any], *, corpus_index: int
) -> dict[str, Any]:
    source_id = str(record["source_id"])
    conversation = record.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        raise ValueError(f"conversationがありません: {source_id}")

    dialogue: list[dict[str, Any]] = []
    first_student_response = ""
    teacher_turns = 0
    for item in conversation:
        role = item.get("role")
        content = item.get("content")
        if role not in {"student", "teacher"}:
            raise ValueError(f"未対応のroleです: {source_id}: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"空の発話があります: {source_id}")
        visible = content.strip()
        if role == "student" and not first_student_response:
            first_student_response = visible
        if role == "teacher":
            teacher_turns += 1
        dialogue.append({
            "turn": int(item.get("turn", 0)),
            "role": role,
            "content": visible,
        })

    if not first_student_response or teacher_turns == 0:
        raise ValueError(f"評価可能な対話ではありません: {source_id}")
    problem = str(record["problem"])
    expected_problem = str(question["translated_question"])
    if problem != expected_problem:
        raise ValueError(f"問題文が元データと一致しません: {source_id}")
    profile = record.get("student_profile") or {}
    completed = record.get("is_completed") is True
    return {
        "schema_version": "legacy-500-dialogue-record-v1-visible-only",
        "case_id": f"legacy-500-{source_id}",
        "source_id": source_id,
        "condition": "legacy-500-corpus",
        "teacher_model": "legacy-corpus-generator-not-recorded",
        "profile_id": str(profile.get("id", "not_available")),
        "learning_status": "not_available_in_legacy_corpus",
        "initial_emotion": "not_available_in_legacy_corpus",
        "initial_response_sha256": sha256_text(first_student_response),
        "problem": problem,
        "reference_solution": str(question["translated_solution"]),
        "dialogue": dialogue,
        "dialogue_generation_succeeded": True,
        "termination_reason": "teacher_completed" if completed else "corpus_ended",
        "teacher_turns": teacher_turns,
        "transfer_generation_succeeded": False,
        "transfer_answer": None,
        "similar_question": None,
        "similar_solution": None,
        "corpus_index": corpus_index,
    }


def build_evaluation_input(
    corpus_path: Path, questions_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    corpus = read_jsonl(corpus_path)
    questions = {str(row["id"]): row for row in read_jsonl(questions_path)}
    if len(questions) != 1000:
        raise ValueError(f"問題IDが一意ではないか、1000件ではありません: {len(questions)}")
    source_ids = [str(row["source_id"]) for row in corpus]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("コーパス内に重複source_idがあります")
    missing = [source_id for source_id in source_ids if source_id not in questions]
    if missing:
        raise ValueError(f"参照問題が見つかりません: {missing[:5]}")

    converted = [
        convert_dialogue(row, questions[str(row["source_id"])], corpus_index=index)
        for index, row in enumerate(corpus)
    ]
    manifest = {
        "schema_version": "legacy-500-evaluation-selection-v1",
        "dataset": "pipelines/corpus_creation/500_empathetic_dialogues.jsonl",
        "population_size": len(corpus),
        "selected_count": len(converted),
        "selection_method": "full_population",
        "corpus_sha256": sha256_file(corpus_path),
        "questions_sha256": sha256_file(questions_path),
        "teacher_internal_reasoning_in_output": False,
        "judge_input_policy": "problem, reference solution, and visible dialogue only",
    }
    return converted, manifest


def main() -> None:
    args = parse_args()
    corpus_path = args.corpus.resolve()
    questions_path = args.questions.resolve()
    output_path = args.output.resolve()
    manifest_path = args.manifest.resolve()
    rows, manifest = build_evaluation_input(corpus_path, questions_path)
    write_jsonl(output_path, rows)
    manifest["output"] = str(output_path)
    manifest["output_sha256"] = sha256_file(output_path)
    write_json(manifest_path, manifest)
    print(f"コーパス評価入力: {len(rows)}件")
    print(f"評価入力: {output_path}")
    print(f"選定記録: {manifest_path}")


if __name__ == "__main__":
    main()
