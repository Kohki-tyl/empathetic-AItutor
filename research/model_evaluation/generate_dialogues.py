from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import (
    call_json_model, call_text_model, extract_visible_teacher_utterance, openai_client,
    read_json, read_jsonl, resolve_path, retry_call, sha256_file, sha256_text,
    stable_fingerprint, validate_nonempty_utterance, visible_dialogue_text, write_json, write_jsonl,
    utc_now,
)
from schemas import STUDENT_UTTERANCE_SCHEMA, TRANSFER_ANSWER_SCHEMA


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="固定初回発話から教師モデルとの対話を生成する")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.example.json")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--teacher-api-key-env", default="TEACHER_API_KEY")
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--initial-responses", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _seed(base: int, case_id: str, offset: int) -> int:
    return base + int(sha256_text(case_id)[:8], 16) % 1_000_000 + offset


def profile_with_initial_response(case: dict[str, Any], dialogue: list[dict[str, Any]]) -> dict[str, Any]:
    if not dialogue or dialogue[0].get("role") != "student":
        raise ValueError("対話の先頭に固定初期応答がありません")
    profile = case["student_profile"]
    return {
        "grade": profile["grade"],
        "speech_style": profile["speech_style"],
        "initial_state": {
            **profile["initial_state"],
            "initial_response": dialogue[0]["content"],
        },
    }


def student_payload(case: dict[str, Any], dialogue: list[dict[str, Any]]) -> str:
    return json.dumps({
        "problem": case["problem"],
        "student_profile": profile_with_initial_response(case, dialogue),
        "visible_dialogue": visible_dialogue_text(dialogue),
        "task": "教師の直前発話に対する次の生徒発話を生成してください。",
    }, ensure_ascii=False)


def teacher_payload(case: dict[str, Any], dialogue: list[dict[str, Any]]) -> str:
    return (
        f"【問題】\n{case['problem']}\n\n"
        f"【これまでの可視対話】\n{visible_dialogue_text(dialogue)}\n\n"
        "次の教師発話を生成してください。"
    )


def transfer_payload(case: dict[str, Any], dialogue: list[dict[str, Any]]) -> str:
    return json.dumps({
        "student_profile": profile_with_initial_response(case, dialogue),
        "original_problem": case["problem"],
        "visible_phase1_dialogue": visible_dialogue_text(dialogue),
        "similar_problem": case["similar_question"],
        "task": "類似問題を解き、最終回答だけを返してください。",
    }, ensure_ascii=False)


def validate_initial_responses(cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    initial_by_case: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in initial_by_case:
            raise ValueError(f"固定初回発話のcase_idが空または重複しています: {case_id!r}")
        initial_by_case[case_id] = row
    missing = [
        str(case["case_id"]) for case in cases
        if not initial_by_case.get(str(case["case_id"]), {}).get("generation_succeeded")
    ]
    if missing:
        raise RuntimeError(f"成功した固定初回発話がないケースがあります: {missing[:5]}")
    for case in cases:
        row = initial_by_case[str(case["case_id"])]
        if str(row.get("profile_id")) != str(case["profile_id"]):
            raise ValueError(f"{case['case_id']}: 固定初回発話のprofile_idが不一致です")
        validate_nonempty_utterance(row.get("initial_response"))
    return initial_by_case


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("shard設定が不正です")
    config_path = args.config.resolve()
    config = read_json(config_path)
    student = config["student"]
    teacher = config["teacher"]
    cases_path = args.cases.resolve() if args.cases else resolve_path(config_path, config["paths"]["cases"])
    initial_path = args.initial_responses.resolve() if args.initial_responses else resolve_path(
        config_path, config["paths"]["initial_responses"]
    )
    output = args.output.resolve()
    cases = read_jsonl(cases_path)
    if args.limit is not None:
        cases = cases[:args.limit]
    cases = [case for index, case in enumerate(cases) if index % args.num_shards == args.shard_index]
    initial_by_case = validate_initial_responses(cases, read_jsonl(initial_path))

    teacher_prompt_path = BASE_DIR / "prompts" / "teacher_system.txt"
    student_prompt_path = BASE_DIR / "prompts" / "student_followup_system.txt"
    transfer_prompt_path = BASE_DIR / "prompts" / "transfer_student_system.txt"
    teacher_prompt = teacher_prompt_path.read_text(encoding="utf-8")
    student_prompt = student_prompt_path.read_text(encoding="utf-8")
    transfer_prompt = transfer_prompt_path.read_text(encoding="utf-8")
    fingerprint_inputs = {
        "schema_version": "teacher-dialogue-generation-v3-four-state-profile",
        "condition": args.condition,
        "teacher_model": args.teacher_model,
        "teacher_base_url": args.teacher_base_url,
        "student": student,
        "teacher": teacher,
        "cases_sha256": sha256_file(cases_path),
        "initial_responses_sha256": sha256_file(initial_path),
        "prompts": {
            "teacher": sha256_file(teacher_prompt_path),
            "student": sha256_file(student_prompt_path),
            "transfer": sha256_file(transfer_prompt_path),
        },
        "shard": [args.shard_index, args.num_shards],
    }
    fingerprint = stable_fingerprint(fingerprint_inputs)
    manifest_path = output.with_suffix(".manifest.json")
    if args.overwrite:
        existing: dict[str, dict[str, Any]] = {}
    else:
        existing = {str(row["case_id"]): row for row in read_jsonl(output)}
        if existing:
            if not manifest_path.exists() or read_json(manifest_path).get("run_fingerprint") != fingerprint:
                raise RuntimeError("既存出力のmanifestまたはfingerprintが一致しません")

    student_client = openai_client(
        base_url=student["base_url"], api_key_env=student["api_key_env"],
        timeout=float(student["request_timeout"]),
    )
    teacher_client = openai_client(
        base_url=args.teacher_base_url, api_key_env=args.teacher_api_key_env,
        fallback_key_env="OPENAI_API_KEY", timeout=float(teacher["request_timeout"]), default_key="EMPTY",
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        previous = existing.get(case_id)
        if previous and previous.get("dialogue_generation_succeeded") is True:
            rows.append(previous)
            continue
        initial = initial_by_case[case_id]
        initial_state = case["student_profile"]["initial_state"]
        dialogue: list[dict[str, Any]] = [{
            "turn": 0, "role": "student", "content": initial["initial_response"],
        }]
        call_metadata: list[dict[str, Any]] = []
        completed = False
        generation_error: str | None = None
        teacher_turns = 0
        for turn in range(int(teacher["max_turns"])):
            try:
                raw_teacher, retry_index = retry_call(lambda attempt: call_text_model(
                    teacher_client,
                    model=args.teacher_model,
                    messages=[
                        {"role": "system", "content": teacher_prompt},
                        {"role": "user", "content": teacher_payload(case, dialogue)},
                    ],
                    temperature=float(teacher["temperature"]),
                    top_p=float(teacher["top_p"]),
                    max_tokens=int(teacher["max_tokens"]),
                    seed=_seed(int(teacher["seed"]), case_id, 10_000 + turn * 20 + attempt),
                ), attempts=int(teacher["retries"]))
                visible, completed = extract_visible_teacher_utterance(raw_teacher, teacher["completion_marker"])
                dialogue.append({"turn": turn, "role": "teacher", "content": visible})
                teacher_turns += 1
                call_metadata.append({
                    "turn": turn, "role": "teacher", "retry_index": retry_index,
                    "raw_response_sha256": sha256_text(raw_teacher),
                })
            except Exception as exc:
                generation_error = f"teacher turn {turn}: {type(exc).__name__}: {exc}"
                break
            if completed:
                break
            if turn + 1 >= int(teacher["max_turns"]):
                break
            try:
                value, retry_index = retry_call(lambda attempt: call_json_model(
                    student_client,
                    model=student["model"],
                    messages=[
                        {"role": "system", "content": student_prompt},
                        {"role": "user", "content": student_payload(case, dialogue)},
                    ],
                    schema=STUDENT_UTTERANCE_SCHEMA,
                    temperature=float(student["temperature"]),
                    top_p=float(student["top_p"]),
                    max_tokens=int(student["max_tokens"]),
                    seed=_seed(int(student["seed"]), case_id, 20_000 + turn * 20 + attempt),
                    reasoning_effort=student.get("reasoning_effort"),
                ), attempts=int(student["retries"]))
                utterance = validate_nonempty_utterance(value.get("utterance"))
                dialogue.append({"turn": turn + 1, "role": "student", "content": utterance})
                call_metadata.append({"turn": turn + 1, "role": "student", "retry_index": retry_index})
            except Exception as exc:
                generation_error = f"student turn {turn + 1}: {type(exc).__name__}: {exc}"
                break

        dialogue_success = generation_error is None and teacher_turns > 0
        termination_reason = "teacher_completed" if completed else (
            "max_turns" if dialogue_success else "generation_error"
        )
        transfer_answer: str | None = None
        transfer_error: str | None = None
        transfer_retry_index: int | None = None
        if dialogue_success:
            try:
                value, transfer_retry_index = retry_call(lambda attempt: call_json_model(
                    student_client,
                    model=student["model"],
                    messages=[
                        {"role": "system", "content": transfer_prompt},
                        {"role": "user", "content": transfer_payload(case, dialogue)},
                    ],
                    schema=TRANSFER_ANSWER_SCHEMA,
                    temperature=float(student["temperature"]),
                    top_p=float(student["top_p"]),
                    max_tokens=int(student["max_tokens"]),
                    seed=_seed(int(student["seed"]), case_id, 30_000 + attempt),
                    reasoning_effort=student.get("reasoning_effort"),
                ), attempts=int(student["retries"]))
                transfer_answer = validate_nonempty_utterance(value.get("final_answer"))
            except Exception as exc:
                transfer_error = f"{type(exc).__name__}: {exc}"

        row = {
            "schema_version": "teacher-dialogue-record-v3-four-state-profile",
            "generated_at_utc": utc_now(),
            "case_id": case_id,
            "source_id": case["source_id"],
            "condition": args.condition,
            "teacher_model": args.teacher_model,
            "student_model": student["model"],
            "problem": case["problem"],
            "reference_solution": case["reference_solution"],
            "similar_question": case["similar_question"],
            "similar_solution": case["similar_solution"],
            "profile_id": case["profile_id"],
            "learning_status": initial_state["learning_status"],
            "initial_emotion": initial_state["emotion"],
            "initial_response_sha256": initial["initial_response_sha256"],
            "dialogue": dialogue,
            "teacher_turns": teacher_turns,
            "teacher_declared_completion": completed,
            "termination_reason": termination_reason,
            "dialogue_generation_succeeded": dialogue_success,
            "generation_error": generation_error,
            "call_metadata": call_metadata,
            "transfer_answer": transfer_answer,
            "transfer_generation_succeeded": transfer_answer is not None,
            "transfer_retry_index": transfer_retry_index,
            "transfer_error": transfer_error,
        }
        rows.append(row)
        write_jsonl(output, rows)

    write_json(manifest_path, {
        "schema_version": "teacher-dialogue-generation-manifest-v3-four-state-profile",
        "created_at_utc": utc_now(),
        "run_fingerprint": fingerprint,
        "fingerprint_inputs": fingerprint_inputs,
        "planned_cases": len(cases),
        "dialogue_generation_successes": sum(row.get("dialogue_generation_succeeded") is True for row in rows),
        "transfer_generation_successes": sum(row.get("transfer_generation_succeeded") is True for row in rows),
        "teacher_internal_reasoning_saved": False,
        "knowledge_boundary_filtering": False,
    })
    print(f"対話生成成功: {sum(row.get('dialogue_generation_succeeded') is True for row in rows)}/{len(cases)}")
    print(f"出力: {output}")


if __name__ == "__main__":
    main()
