from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import (
    extract_visible_teacher_utterance,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS = BASE_DIR.parent / "corpus" / "v3_397_dialogues.jsonl"
DEFAULT_METADATA = BASE_DIR.parent / "corpus" / "v3_397_metadata.jsonl"
DEFAULT_OUTPUT = BASE_DIR / "selections" / "corpus_v3_50_dialogues.jsonl"
DEFAULT_MANIFEST = BASE_DIR / "selections" / "corpus_v3_50.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="シャッフル済み397対話から固定件数を可視発話評価形式へ変換する"
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def _initial_student_text(content: str, problem: str) -> str:
    prefix = f"問題: {problem}\n\n"
    return content[len(prefix):].strip() if content.startswith(prefix) else content.strip()


def convert_dialogue(
    record: dict[str, Any], metadata: dict[str, Any], *, corpus_index: int
) -> dict[str, Any]:
    if metadata.get("corpus_index") != corpus_index:
        raise ValueError(f"metadataのcorpus_indexが不一致です: {corpus_index}")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"messagesがありません: {corpus_index}")

    problem = str(metadata["problem"])
    dialogue: list[dict[str, Any]] = []
    student_turn = 0
    teacher_turn = 0
    completed = False
    first_student_response = ""
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"空の発話があります: {corpus_index}")
        if role == "system":
            continue
        if role == "user":
            visible = (
                _initial_student_text(content, problem)
                if student_turn == 0
                else content.strip()
            )
            if not first_student_response:
                first_student_response = visible
            dialogue.append({"turn": student_turn, "role": "student", "content": visible})
            student_turn += 1
        elif role == "assistant":
            visible, has_marker = extract_visible_teacher_utterance(content, "[指導完了]")
            completed = completed or has_marker
            dialogue.append({"turn": teacher_turn, "role": "teacher", "content": visible})
            teacher_turn += 1
        else:
            raise ValueError(f"未対応のroleです: {role!r}")

    if not first_student_response or teacher_turn == 0:
        raise ValueError(f"評価可能な対話ではありません: {corpus_index}")
    source_id = str(metadata["source_id"])
    return {
        "schema_version": "existing-corpus-dialogue-record-v1-visible-only",
        "case_id": f"corpus-v3-{source_id}",
        "source_id": source_id,
        "condition": "corpus-v3",
        "teacher_model": "gpt-5.4",
        "profile_id": str(metadata.get("profile_id", "not_available")),
        "learning_status": "not_available_in_v3_corpus",
        "initial_emotion": "not_available_in_v3_corpus",
        "initial_response_sha256": sha256_text(first_student_response),
        "problem": problem,
        "reference_solution": str(metadata["reference_solution"]),
        "dialogue": dialogue,
        "dialogue_generation_succeeded": True,
        "termination_reason": "teacher_completed" if completed else "corpus_completed",
        "teacher_turns": teacher_turn,
        "transfer_generation_succeeded": False,
        "transfer_answer": None,
        "similar_question": None,
        "similar_solution": None,
        "corpus_index": corpus_index,
    }


def build_subset(
    corpus_path: Path, metadata_path: Path, count: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    corpus = read_jsonl(corpus_path)
    metadata = read_jsonl(metadata_path)
    if len(corpus) != len(metadata):
        raise ValueError(f"corpusとmetadataの件数が不一致です: {len(corpus)} != {len(metadata)}")
    if count < 1 or count > len(corpus):
        raise ValueError(f"countは1〜{len(corpus)}で指定してください")

    selected = [
        convert_dialogue(corpus[index], metadata[index], corpus_index=index)
        for index in range(count)
    ]
    manifest = {
        "schema_version": "existing-corpus-evaluation-selection-v1",
        "dataset": "v3_turn_audited_cot_sft",
        "population_size": len(corpus),
        "selected_count": count,
        "selection_method": "first_n_from_existing_seeded_shuffle",
        "source_shuffle_seed": 42,
        "sampling_without_replacement": True,
        "corpus_sha256": sha256_file(corpus_path),
        "metadata_sha256": sha256_file(metadata_path),
        "teacher_internal_reasoning_in_output": False,
        "judge_input_policy": "problem, reference solution, and visible dialogue only",
        "selected_cases": [
            {"corpus_index": row["corpus_index"], "source_id": row["source_id"], "case_id": row["case_id"]}
            for row in selected
        ],
    }
    return selected, manifest


def main() -> None:
    args = parse_args()
    corpus_path = args.corpus.resolve()
    metadata_path = args.metadata.resolve()
    output_path = args.output.resolve()
    manifest_path = args.manifest.resolve()
    rows, manifest = build_subset(corpus_path, metadata_path, args.count)
    write_jsonl(output_path, rows)
    try:
        manifest["output"] = output_path.relative_to(BASE_DIR).as_posix()
    except ValueError:
        manifest["output"] = str(output_path)
    manifest["output_sha256"] = sha256_file(output_path)
    write_json(manifest_path, manifest)
    print(f"コーパス評価サブセット: {len(rows)}/{manifest['population_size']}件")
    print(f"評価入力: {output_path}")
    print(f"選定記録: {manifest_path}")


if __name__ == "__main__":
    main()
