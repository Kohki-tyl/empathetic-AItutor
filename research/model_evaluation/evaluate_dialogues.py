from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import (
    AXES, EMPATHY_AXES, INSTRUCTION_AXES, axis_scores, call_json_model, group_score,
    openai_client, overall_score, read_json, read_jsonl, retry_call, sha256_file,
    stable_fingerprint, visible_dialogue_text, write_json, write_jsonl,
    utc_now,
)
from schemas import DIALOGUE_JUDGE_SCHEMA, TRANSFER_JUDGE_SCHEMA


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可視教師発話だけを対話全体で6軸評価する")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.example.json")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dialogue_judge_payload(row: dict[str, Any]) -> str:
    """問題、参照解答、可視対話以外を意図的に含めない。"""
    return json.dumps({
        "problem": row["problem"],
        "reference_solution": row["reference_solution"],
        "visible_dialogue": visible_dialogue_text(row["dialogue"]),
        "task": "教師の可視発話だけを、対話全体として6軸評価してください。",
    }, ensure_ascii=False)


def transfer_judge_payload(row: dict[str, Any]) -> str:
    return json.dumps({
        "similar_question": row["similar_question"],
        "reference_solution": row["similar_solution"],
        "student_final_answer": row["transfer_answer"],
    }, ensure_ascii=False)


def validate_judge_result(value: dict[str, Any]) -> dict[str, Any]:
    scores = axis_scores(value)
    if scores["mathematical_accuracy"] is None or scores["instruction_completion"] is None:
        raise ValueError("数学的正確性と指導完了判定はNAにできません")
    failures = value.get("critical_failure_details")
    if not isinstance(failures, list):
        raise ValueError("critical_failure_detailsが配列ではありません")
    value["critical_failure"] = bool(failures)
    value["applicable_axis_count"] = sum(score is not None for score in scores.values())
    value["overall_score_60"] = overall_score(scores)
    value["instruction_group_mean"] = group_score(scores, INSTRUCTION_AXES)
    value["empathy_group_mean"] = group_score(scores, EMPATHY_AXES)
    return value


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    judge = dict(config["judge"])
    if args.judge_model:
        judge["model"] = args.judge_model
    if not str(judge.get("model", "")).strip():
        raise SystemExit("Judgeモデルを--judge-modelまたはconfig.judge.modelで指定してください")
    input_path = args.input.resolve()
    output = args.output.resolve()
    dialogue_prompt_path = BASE_DIR / "prompts" / "dialogue_judge_system.txt"
    transfer_prompt_path = BASE_DIR / "prompts" / "transfer_judge_system.txt"
    dialogue_prompt = dialogue_prompt_path.read_text(encoding="utf-8")
    transfer_prompt = transfer_prompt_path.read_text(encoding="utf-8")
    rows = read_jsonl(input_path)
    if args.limit is not None:
        rows = rows[:args.limit]

    fingerprint_inputs = {
        "schema_version": "teacher-dialogue-evaluation-v4-visible-dialogue-only",
        "input_sha256": sha256_file(input_path),
        "dialogue_prompt_sha256": sha256_file(dialogue_prompt_path),
        "transfer_prompt_sha256": sha256_file(transfer_prompt_path),
        "judge": judge,
        "axes": list(AXES),
    }
    fingerprint = stable_fingerprint(fingerprint_inputs)
    manifest_path = output.with_suffix(".manifest.json")
    if args.overwrite:
        existing: dict[str, dict[str, Any]] = {}
    else:
        existing = {str(row["case_id"]): row for row in read_jsonl(output)}
        if existing:
            if not manifest_path.exists() or read_json(manifest_path).get("run_fingerprint") != fingerprint:
                raise RuntimeError("既存評価のmanifestまたはfingerprintが一致しません")

    client = openai_client(
        base_url=judge["base_url"], api_key_env=judge["api_key_env"],
        fallback_key_env=judge.get("fallback_api_key_env"), timeout=float(judge["request_timeout"]),
    )
    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        case_id = str(row["case_id"])
        previous = existing.get(case_id)
        if previous and previous.get("evaluation_status") == "evaluated":
            results.append(previous)
            continue
        base = {
            "schema_version": "teacher-dialogue-evaluation-record-v4-visible-dialogue-only",
            "evaluated_at_utc": utc_now(),
            "case_id": case_id,
            "source_id": row["source_id"],
            "condition": row["condition"],
            "teacher_model": row["teacher_model"],
            "profile_id": row["profile_id"],
            "learning_status": row["learning_status"],
            "initial_emotion": row["initial_emotion"],
            "initial_response_sha256": row["initial_response_sha256"],
            "dialogue_generation_succeeded": row["dialogue_generation_succeeded"],
            "termination_reason": row["termination_reason"],
            "teacher_turns": row["teacher_turns"],
            "transfer_generation_succeeded": row["transfer_generation_succeeded"],
        }
        if row.get("dialogue_generation_succeeded") is not True:
            results.append({
                **base,
                "evaluation_status": "not_evaluable_generation_failure",
                "evaluation": None,
                "transfer_evaluation": None,
                "error": row.get("generation_error"),
            })
            write_jsonl(output, results)
            continue
        seed = int(judge["seed"]) + index * 100
        try:
            evaluation, retry_index = retry_call(lambda attempt: call_json_model(
                client,
                model=judge["model"],
                messages=[
                    {"role": "system", "content": dialogue_prompt},
                    {"role": "user", "content": dialogue_judge_payload(row)},
                ],
                schema=DIALOGUE_JUDGE_SCHEMA,
                temperature=float(judge["temperature"]),
                top_p=float(judge["top_p"]),
                max_tokens=int(judge["max_tokens"]),
                seed=seed + attempt,
                reasoning_effort=judge.get("reasoning_effort"),
            ), attempts=int(judge["retries"]))
            evaluation = validate_judge_result(evaluation)
        except Exception as exc:
            results.append({
                **base,
                "evaluation_status": "judge_failure",
                "evaluation": None,
                "transfer_evaluation": None,
                "error": f"{type(exc).__name__}: {exc}",
            })
            write_jsonl(output, results)
            continue

        transfer_evaluation: dict[str, Any] | None = None
        transfer_retry_index: int | None = None
        transfer_error: str | None = None
        if row.get("transfer_generation_succeeded") is True:
            try:
                transfer_evaluation, transfer_retry_index = retry_call(lambda attempt: call_json_model(
                    client,
                    model=judge["model"],
                    messages=[
                        {"role": "system", "content": transfer_prompt},
                        {"role": "user", "content": transfer_judge_payload(row)},
                    ],
                    schema=TRANSFER_JUDGE_SCHEMA,
                    temperature=float(judge["temperature"]),
                    top_p=float(judge["top_p"]),
                    max_tokens=int(judge["max_tokens"]),
                    seed=seed + 50 + attempt,
                    reasoning_effort=judge.get("reasoning_effort"),
                ), attempts=int(judge["retries"]))
            except Exception as exc:
                transfer_error = f"{type(exc).__name__}: {exc}"
        results.append({
            **base,
            "evaluation_status": "evaluated",
            "judge_model": judge["model"],
            "judge_retry_index": retry_index,
            "evaluation": evaluation,
            "transfer_evaluation": transfer_evaluation,
            "transfer_judge_retry_index": transfer_retry_index,
            "transfer_error": transfer_error,
            "error": None,
        })
        write_jsonl(output, results)

    write_json(manifest_path, {
        "schema_version": "teacher-dialogue-evaluation-manifest-v4-visible-dialogue-only",
        "created_at_utc": utc_now(),
        "run_fingerprint": fingerprint,
        "fingerprint_inputs": fingerprint_inputs,
        "planned_cases": len(rows),
        "dialogue_generation_successes": sum(row.get("dialogue_generation_succeeded") is True for row in rows),
        "evaluated_cases": sum(row.get("evaluation_status") == "evaluated" for row in results),
        "judge_failures": sum(row.get("evaluation_status") == "judge_failure" for row in results),
        "student_quality_judged": False,
        "teacher_internal_reasoning_in_judge_input": False,
    })
    print(f"対話評価完了: {sum(row.get('evaluation_status') == 'evaluated' for row in results)}/{len(rows)}")
    print(f"出力: {output}")


if __name__ == "__main__":
    main()
