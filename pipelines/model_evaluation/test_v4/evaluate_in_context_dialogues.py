"""生成済みv4インコンテキスト転移対話をv4基準で評価する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
SHARED_DIR = BASE_DIR.parent / "shared"
TRANSFER_MODE = "v4_in_context"

MATH_JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "math_judge",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"is_correct": {"type": "boolean"}, "judge_reason": {"type": "string"}},
            "required": ["is_correct", "judge_reason"],
            "additionalProperties": False,
        },
    },
}

V4_INSTRUCTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "v4_instruction_judge",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "mathematical_accuracy_score": {"type": "integer", "minimum": 0, "maximum": 10},
                "error_diagnosis_recovery_score": {"type": "integer", "minimum": 0, "maximum": 10},
                "cognitive_empathy_score": {"type": "integer", "minimum": 0, "maximum": 10},
                "emotional_support_score": {"type": "integer", "minimum": 0, "maximum": 10},
                "adaptive_scaffolding_score": {"type": "integer", "minimum": 0, "maximum": 10},
                "verification_completion_score": {"type": "integer", "minimum": 0, "maximum": 10},
                "total_score": {"type": "integer", "minimum": 0, "maximum": 60},
                "false_affirmation_count": {"type": "integer", "minimum": 0},
                "direct_answer_without_need_count": {"type": "integer", "minimum": 0},
                "critical_failure": {"type": "boolean"},
                "analysis_reflected_in_utterance": {"type": "boolean"},
                "judge_reason": {"type": "string"},
            },
            "required": [
                "mathematical_accuracy_score", "error_diagnosis_recovery_score",
                "cognitive_empathy_score", "emotional_support_score",
                "adaptive_scaffolding_score", "verification_completion_score",
                "total_score", "false_affirmation_count",
                "direct_answer_without_need_count", "critical_failure",
                "analysis_reflected_in_utterance", "judge_reason",
            ],
            "additionalProperties": False,
        },
    },
}

REALISM_JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "student_realism_judge",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "realism_score": {"type": "integer", "minimum": 0, "maximum": 10},
                "tutor_leak_count": {"type": "integer", "minimum": 0},
                "knowledge_violation_count": {"type": "integer", "minimum": 0},
                "style_violation_count": {"type": "integer", "minimum": 0},
                "implausible_update_count": {"type": "integer", "minimum": 0},
                "blind_agreement_count": {"type": "integer", "minimum": 0},
                "judge_reason": {"type": "string"},
            },
            "required": [
                "realism_score", "tutor_leak_count", "knowledge_violation_count",
                "style_violation_count", "implausible_update_count",
                "blind_agreement_count", "judge_reason",
            ],
            "additionalProperties": False,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE_DIR / "data" / "dialogues.jsonl")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "data" / "evaluated_results.jsonl")
    parser.add_argument("--similar-questions", type=Path, default=SHARED_DIR / "questions" / "similar_test_math_questions.jsonl")
    parser.add_argument(
        "--excluded-question-ids", type=Path,
        default=SHARED_DIR / "questions" / "excluded_test_question_ids.json",
    )
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL_NAME", "gpt-5.6-terra"))
    parser.add_argument("--judge-reasoning-effort", default="high")
    parser.add_argument("--judge-proxy", default=os.getenv("JUDGE_PROXY"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--judge-retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--skip-connection-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_excluded_ids(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(source_id) for source_id in data.get("excluded_source_ids", [])}


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluation_fingerprint(args: argparse.Namespace) -> tuple[str, dict[str, str]]:
    source_paths = {
        "input": args.input,
        "similar_questions": args.similar_questions,
        "excluded_question_ids": args.excluded_question_ids,
        "instruction_judge_system": BASE_DIR / "prompts" / "instruction_judge_system.txt",
        "student_realism_judge_system": BASE_DIR / "prompts" / "student_realism_judge_system.txt",
        "math_judge_system": SHARED_DIR / "prompts" / "eval_judge_system.txt",
    }
    hashes = {name: sha256(path) for name, path in source_paths.items()}
    payload = {
        "source_sha256": hashes,
        "judge_model": args.judge_model,
        "judge_reasoning_effort": args.judge_reasoning_effort,
        "limit": args.limit,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "transfer_mode": TRANSFER_MODE,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), hashes


def parse_json_response(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def call_judge(client: OpenAI, model: str, system: str, user: str,
               schema: dict[str, Any], seed: int, retries: int,
               reasoning_effort: str | None = None) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_completion_tokens": 2048,
                "response_format": schema,
                "seed": seed + attempt,
            }
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
            else:
                kwargs["temperature"] = 0.0
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Judge returned empty content")
            result = parse_json_response(content)
            result["judge_attempts"] = attempt + 1
            return result
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 8))
    return {"error": f"{type(last_error).__name__}: {last_error}", "judge_attempts": retries}


def judge_succeeded(result: Any) -> bool:
    return isinstance(result, dict) and not result.get("error")


def evaluation_succeeded(row: dict[str, Any]) -> bool:
    return all(judge_succeeded(row.get(field)) for field in (
        "math_judge", "v4_instruction_evaluation", "student_realism_evaluation",
    ))


def dialogue_text(dialogue: list[dict[str, Any]]) -> str:
    blocks = []
    for turn in dialogue:
        block = f"{turn['role']}: {turn['content']}"
        if turn.get("role") == "teacher" and turn.get("analysis"):
            block += f"\n教師の明示的判断記録: {turn['analysis']}"
        if turn.get("role") == "teacher":
            block += f"\n教師の完了判定: {bool(turn.get('is_completed'))}"
        blocks.append(block)
    return "\n".join(blocks)


def validate_generation_inputs(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Judge対象の生成結果が空です")
    seen: set[str] = set()
    failures: list[str] = []
    for index, row in enumerate(rows):
        run_id = str(row.get("run_id", "")).strip()
        reasons: list[str] = []
        if not run_id:
            reasons.append("run_id欠損")
        elif run_id in seen:
            reasons.append("run_id重複")
        seen.add(run_id)
        if row.get("generation_error"):
            reasons.append(f"generation_error={row['generation_error']}")
        if int(row.get("phase1_turns", 0)) <= 0:
            reasons.append("Phase 1対話なし")
        dialogue = row.get("dialogue_log")
        if not isinstance(dialogue, list) or not dialogue:
            reasons.append("dialogue_log欠損")
        if not str(row.get("phase2_student_answer", "")).strip():
            reasons.append("Phase 2回答なし")
        if not isinstance(row.get("phase2_student_trace"), dict):
            reasons.append("Phase 2構造記録なし")
        if reasons:
            failures.append(f"{run_id or f'row-{index}'}: {', '.join(reasons)}")
    if failures:
        preview = "; ".join(failures[:5])
        raise ValueError(
            f"生成未完了のためJudgeを開始しません: {len(failures)}/{len(rows)}件。"
            f"先に生成を再実行してください。{preview}"
        )


def recompute_total(result: dict[str, Any], score_fields: list[str]) -> dict[str, Any]:
    if all(isinstance(result.get(field), int) for field in score_fields):
        result = dict(result)
        result["total_score"] = sum(result[field] for field in score_fields)
    return result


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--num-shardsと--shard-indexの組み合わせが不正です。")
    if args.judge_retries < 1:
        raise SystemExit("--judge-retriesは1以上にしてください。")
    api_key = os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("GPT_API_KEY または OPENAI_API_KEY を設定してください。")
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("生成ログを保持するため、--inputと--outputには別ファイルを指定してください。")

    rows = read_jsonl(args.input)
    validate_generation_inputs(rows)
    excluded_ids = read_excluded_ids(args.excluded_question_ids)
    excluded_rows = [str(row.get("source_id")) for row in rows if str(row.get("source_id")) in excluded_ids]
    if excluded_rows:
        raise ValueError(f"共通除外問題が生成ログに{len(excluded_rows)}件含まれています。")
    mismatched = [
        row.get("run_id") for row in rows
        if row.get("transfer_mode") not in (None, TRANSFER_MODE)
    ]
    if mismatched:
        raise ValueError(f"v4インコンテキスト転移と異なる生成ログが{len(mismatched)}件あります。")
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        rows = rows[:args.limit]
    similar_by_id = {
        str(row.get("source_id") or row.get("id")): row
        for row in read_jsonl(args.similar_questions)
    }
    instruction_system = (BASE_DIR / "prompts" / "instruction_judge_system.txt").read_text(encoding="utf-8")
    math_system = (SHARED_DIR / "prompts" / "eval_judge_system.txt").read_text(encoding="utf-8")
    realism_system = (BASE_DIR / "prompts" / "student_realism_judge_system.txt").read_text(encoding="utf-8")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        args.output.write_text("", encoding="utf-8")
    existing_rows = read_jsonl(args.output) if args.output.exists() else []
    existing_by_id = {str(row["run_id"]): row for row in existing_rows if row.get("run_id")}
    manifest_path = args.output.with_suffix(".manifest.json")
    fingerprint, source_hashes = evaluation_fingerprint(args)
    if existing_rows and not manifest_path.exists() and not args.overwrite:
        raise RuntimeError("既存評価にmanifestがないため安全に再開できません。別の--outputを使用してください。")
    if manifest_path.exists() and not args.overwrite:
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("run_fingerprint") != fingerprint:
            raise RuntimeError(
                "既存評価と入力・Judge・prompt・shard設定が一致しません。"
                "別の--outputを使うか、意図的に再評価する場合だけ--overwriteを指定してください。"
            )
    http_client = httpx.Client(proxy=args.judge_proxy, timeout=args.request_timeout) if args.judge_proxy else httpx.Client(timeout=args.request_timeout)
    client = OpenAI(api_key=api_key, http_client=http_client)
    if not args.skip_connection_check:
        try:
            client.models.list()
        except Exception as exc:
            http_client.close()
            raise SystemExit(f"Judge API接続確認に失敗しました: {type(exc).__name__}: {exc}") from exc

    manifest_path.write_text(json.dumps({
        "phase": "evaluation", "input": str(args.input), "output": str(args.output),
        "judge_model": args.judge_model,
        "judge_reasoning_effort": args.judge_reasoning_effort,
        "source_sha256": source_hashes,
        "run_fingerprint": fingerprint,
        "transfer_mode": TRANSFER_MODE,
        "excluded_question_ids": str(args.excluded_question_ids),
        "planned_runs": len(rows), "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "judge_retries": args.judge_retries, "request_timeout": args.request_timeout,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    for index, row in enumerate(tqdm(rows, desc="test v4 judging")):
        run_id = str(row["run_id"])
        previous = existing_by_id.get(run_id, {})
        if evaluation_succeeded(previous):
            continue
        source_id = str(row["source_id"])
        similar = similar_by_id.get(source_id, {})
        solution = row.get("similar_solution") or similar.get("similar_solution")
        if not solution:
            raise ValueError(f"類似問題の模範解答がありません: {source_id}")
        run_seed = int(row.get("seed", 42)) + index * 100
        log_text = dialogue_text(row.get("dialogue_log", []))

        instruction = previous.get("v4_instruction_evaluation")
        if not judge_succeeded(instruction):
            instruction = call_judge(
                client, args.judge_model, instruction_system,
                f"【対話ログ】\n{log_text}", V4_INSTRUCTION_SCHEMA,
                run_seed + 91, args.judge_retries, args.judge_reasoning_effort,
            )
        instruction = recompute_total(instruction, [
            "mathematical_accuracy_score", "error_diagnosis_recovery_score",
            "cognitive_empathy_score", "emotional_support_score",
            "adaptive_scaffolding_score", "verification_completion_score",
        ])
        realism_payload = json.dumps({
            "student_profile": row.get("student_profile_used"),
            "initial_emotion": row.get("initial_emotion"),
            "generation_condition": row.get("generation_condition"),
            "initial_state": row.get("initial_student_state"),
            "final_state": row.get("final_student_state"),
            "dialogue": [{key: value for key, value in turn.items() if key != "analysis"}
                         for turn in row.get("dialogue_log", [])],
            "phase2_student_answer": row.get("phase2_student_answer"),
            "phase2_student_trace": row.get("phase2_student_trace"),
        }, ensure_ascii=False)
        realism = previous.get("student_realism_evaluation")
        if not judge_succeeded(realism):
            realism = call_judge(
                client, args.judge_model, realism_system, realism_payload,
                REALISM_JUDGE_SCHEMA, run_seed + 93, args.judge_retries,
                args.judge_reasoning_effort,
            )
        math_result = previous.get("math_judge")
        if not judge_succeeded(math_result):
            math_result = call_judge(client, args.judge_model, math_system,
                f"【生徒の最終解答】\n{row.get('phase2_student_answer', '')}\n\n【模範解答】\n{solution}",
                MATH_JUDGE_SCHEMA, run_seed + 94, args.judge_retries,
                args.judge_reasoning_effort)

        evaluated = dict(row)
        evaluated.update({
            "judge_model": args.judge_model,
            "phase2_is_correct": math_result.get("is_correct"),
            "math_judge": math_result,
            "v4_instruction_evaluation": instruction,
            "instruction_evaluation_summary": {
                "combined_total": instruction.get("total_score"),
                "maximum_score": 60,
            },
            "student_realism_evaluation": realism,
        })
        existing_by_id[run_id] = evaluated
        write_jsonl(args.output, [existing_by_id[key] for key in sorted(existing_by_id)])

    http_client.close()
    failed = [run_id for run_id, row in existing_by_id.items() if not evaluation_succeeded(row)]
    if failed:
        raise SystemExit(f"Judge評価が未完了のrunが{len(failed)}件あります。再実行で失敗Judgeだけを再試行できます。")
    print(f"評価完了: {args.output}")
    print(f"設定: {manifest_path}")


if __name__ == "__main__":
    main()
