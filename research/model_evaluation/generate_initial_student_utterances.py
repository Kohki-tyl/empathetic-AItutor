from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import (
    call_json_model, openai_client, read_json, read_jsonl, resolve_path, retry_call,
    sha256_file, sha256_text, stable_fingerprint, validate_nonempty_utterance, write_json, write_jsonl,
    utc_now,
)
from schemas import STUDENT_UTTERANCE_SCHEMA


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT-5.4で初回生徒発話を一度だけ事前生成する")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.example.json")
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def case_seed(base_seed: int, case_id: str) -> int:
    return base_seed + int(sha256_text(case_id)[:8], 16) % 1_000_000


def initial_payload(case: dict[str, Any]) -> str:
    return json.dumps({
        "problem": case["problem"],
        "student_profile": case["student_profile"],
        "task": "この条件で、教師と話す前の最初の生徒発話を1つ生成してください。",
    }, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    student = config["student"]
    cases_path = args.cases.resolve() if args.cases else resolve_path(config_path, config["paths"]["cases"])
    output = args.output.resolve() if args.output else resolve_path(config_path, config["paths"]["initial_responses"])
    prompt_path = BASE_DIR / "prompts" / "student_initial_system.txt"
    prompt = prompt_path.read_text(encoding="utf-8")
    cases = read_jsonl(cases_path)
    if args.limit is not None:
        cases = cases[:args.limit]

    fingerprint_payload = {
        "schema_version": "initial-student-responses-v2-four-state-profile",
        "cases_sha256": sha256_file(cases_path),
        "prompt_sha256": sha256_file(prompt_path),
        "student": student,
    }
    fingerprint = stable_fingerprint(fingerprint_payload)
    manifest_path = output.with_suffix(".manifest.json")
    if args.overwrite:
        existing: dict[str, dict[str, Any]] = {}
    else:
        existing = {str(row["case_id"]): row for row in read_jsonl(output)}
        if existing:
            if not manifest_path.exists():
                raise RuntimeError("既存出力にmanifestがありません。--overwriteか別出力を使用してください")
            old = read_json(manifest_path)
            if old.get("run_fingerprint") != fingerprint:
                raise RuntimeError("既存初回発話と現在設定のfingerprintが一致しません")

    client = openai_client(
        base_url=student["base_url"], api_key_env=student["api_key_env"],
        timeout=float(student["request_timeout"]),
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        previous = existing.get(case_id)
        if previous and previous.get("generation_succeeded") is True:
            rows.append(previous)
            continue
        seed = case_seed(int(student["seed"]), case_id)
        try:
            value, retry_index = retry_call(lambda attempt: call_json_model(
                client,
                model=student["model"],
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": initial_payload(case)},
                ],
                schema=STUDENT_UTTERANCE_SCHEMA,
                temperature=float(student["temperature"]),
                top_p=float(student["top_p"]),
                max_tokens=int(student["max_tokens"]),
                seed=seed + attempt,
                reasoning_effort=student.get("reasoning_effort"),
            ), attempts=int(student["retries"]))
            utterance = validate_nonempty_utterance(value.get("utterance"))
            initial_state = case["student_profile"]["initial_state"]
            row = {
                "case_id": case_id,
                "source_id": case["source_id"],
                "profile_id": case["profile_id"],
                "learning_status": initial_state["learning_status"],
                "initial_emotion": initial_state["emotion"],
                "student_model": student["model"],
                "seed": seed,
                "retry_index": retry_index,
                "initial_response": utterance,
                "initial_response_sha256": sha256_text(utterance),
                "generated_at_utc": utc_now(),
                "generation_succeeded": True,
                "error": None,
            }
        except Exception as exc:
            initial_state = case["student_profile"]["initial_state"]
            row = {
                "case_id": case_id,
                "source_id": case["source_id"],
                "profile_id": case["profile_id"],
                "learning_status": initial_state["learning_status"],
                "initial_emotion": initial_state["emotion"],
                "student_model": student["model"],
                "seed": seed,
                "generated_at_utc": utc_now(),
                "generation_succeeded": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        write_jsonl(output, rows)

    write_json(manifest_path, {
        "schema_version": "initial-student-responses-manifest-v2-four-state-profile",
        "created_at_utc": utc_now(),
        "run_fingerprint": fingerprint,
        "fingerprint_inputs": fingerprint_payload,
        "planned_cases": len(cases),
        "successful_cases": sum(row.get("generation_succeeded") is True for row in rows),
        "selection_policy": "API・空応答・形式破損だけを再試行し、内容では選別しない",
    })
    print(f"初回発話: {sum(row.get('generation_succeeded') is True for row in rows)}/{len(cases)}")
    print(f"出力: {output}")


if __name__ == "__main__":
    main()
