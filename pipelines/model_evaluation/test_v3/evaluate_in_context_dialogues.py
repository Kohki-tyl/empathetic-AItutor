"""生成済みテストv3インコンテキスト対話をJudgeモデルで評価する。"""

from __future__ import annotations

import argparse
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
TRANSFER_MODE = "in_context"

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

EMPATHIC_INSTRUCTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "empathic_instruction_judge",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "emotion_recognition_score": {"type": "integer", "minimum": 0, "maximum": 10},
                "cognitive_empathy_score": {"type": "integer", "minimum": 0, "maximum": 10},
                "emotional_support_score": {"type": "integer", "minimum": 0, "maximum": 10},
                "total_score": {"type": "integer", "minimum": 0, "maximum": 30},
                "judge_reason": {"type": "string"},
            },
            "required": [
                "emotion_recognition_score", "cognitive_empathy_score",
                "emotional_support_score", "total_score", "judge_reason",
            ],
            "additionalProperties": False,
        },
    },
}

MATHEMATICAL_INSTRUCTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "mathematical_instruction_judge",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "mathematical_correctness_score": {"type": "integer", "minimum": 0, "maximum": 10},
                "error_diagnosis_score": {"type": "integer", "minimum": 0, "maximum": 10},
                "adaptive_scaffolding_score": {"type": "integer", "minimum": 0, "maximum": 10},
                "learning_verification_score": {"type": "integer", "minimum": 0, "maximum": 10},
                "cognitive_load_control_score": {"type": "integer", "minimum": 0, "maximum": 10},
                "total_score": {"type": "integer", "minimum": 0, "maximum": 50},
                "false_affirmation_count": {"type": "integer", "minimum": 0},
                "direct_answer_count": {"type": "integer", "minimum": 0},
                "judge_reason": {"type": "string"},
            },
            "required": [
                "mathematical_correctness_score", "error_diagnosis_score",
                "adaptive_scaffolding_score", "learning_verification_score",
                "cognitive_load_control_score", "total_score",
                "false_affirmation_count", "direct_answer_count", "judge_reason",
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
    parser = argparse.ArgumentParser(description="生成済みインコンテキスト学習テストv3をJudge APIで評価する")
    parser.add_argument("--input", type=Path, default=BASE_DIR / "data" / "dialogues.jsonl")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "data" / "evaluated_results.jsonl")
    parser.add_argument("--similar-questions", type=Path, default=SHARED_DIR / "questions" / "similar_test_math_questions.jsonl")
    parser.add_argument(
        "--excluded-question-ids", type=Path,
        default=SHARED_DIR / "questions" / "excluded_test_question_ids.json",
    )
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL_NAME", "gpt-5.4"))
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


def parse_json_response(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def call_judge(client: OpenAI, model: str, system: str, user: str,
               schema: dict[str, Any], seed: int, retries: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0, max_completion_tokens=1024,
                response_format=schema, seed=seed + attempt,
            )
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
        "math_judge", "empathic_instruction_evaluation",
        "mathematical_instruction_evaluation", "student_realism_evaluation",
    ))


def dialogue_text(dialogue: list[dict[str, Any]]) -> str:
    blocks = []
    for turn in dialogue:
        block = f"{turn['role']}: {turn['content']}"
        blocks.append(block)
    return "\n".join(blocks)


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
    excluded_ids = read_excluded_ids(args.excluded_question_ids)
    excluded_rows = [str(row.get("source_id")) for row in rows if str(row.get("source_id")) in excluded_ids]
    if excluded_rows:
        raise ValueError(f"v2/v3共通除外問題が生成ログに{len(excluded_rows)}件含まれています。")
    mismatched = [
        row.get("run_id") for row in rows
        if row.get("transfer_mode") not in (None, TRANSFER_MODE)
    ]
    if mismatched:
        raise ValueError(f"テストv3と異なる生成ログが{len(mismatched)}件あります。")
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        rows = rows[:args.limit]
    similar_by_id = {
        str(row.get("source_id") or row.get("id")): row
        for row in read_jsonl(args.similar_questions)
    }
    empathic_instruction_system = (SHARED_DIR / "prompts" / "eval_empathic_instruction_judge_system.txt").read_text(encoding="utf-8")
    mathematical_instruction_system = (SHARED_DIR / "prompts" / "eval_mathematical_instruction_judge_system.txt").read_text(encoding="utf-8")
    math_system = (SHARED_DIR / "prompts" / "eval_judge_system.txt").read_text(encoding="utf-8")
    realism_system = (BASE_DIR / "prompts" / "v3_student_realism_judge_system.txt").read_text(encoding="utf-8")

    if args.overwrite:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("", encoding="utf-8")
    existing_rows = read_jsonl(args.output) if args.output.exists() else []
    existing_by_id = {str(row["run_id"]): row for row in existing_rows if row.get("run_id")}
    http_client = httpx.Client(proxy=args.judge_proxy, timeout=args.request_timeout) if args.judge_proxy else httpx.Client(timeout=args.request_timeout)
    client = OpenAI(api_key=api_key, http_client=http_client)
    if not args.skip_connection_check:
        try:
            client.models.list()
        except Exception as exc:
            http_client.close()
            raise SystemExit(f"Judge API接続確認に失敗しました: {type(exc).__name__}: {exc}") from exc

    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({
        "phase": "evaluation", "input": str(args.input), "output": str(args.output),
        "judge_model": args.judge_model, "transfer_mode": TRANSFER_MODE,
        "excluded_question_ids": str(args.excluded_question_ids),
        "planned_runs": len(rows), "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "judge_retries": args.judge_retries, "request_timeout": args.request_timeout,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    for index, row in enumerate(tqdm(rows, desc="test v3 judging")):
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

        empathic_instruction = previous.get("empathic_instruction_evaluation")
        if not judge_succeeded(empathic_instruction):
            empathic_instruction = call_judge(client, args.judge_model, empathic_instruction_system,
                f"【対話ログ】\n{log_text}", EMPATHIC_INSTRUCTION_SCHEMA, run_seed + 91, args.judge_retries)
        mathematical_instruction = previous.get("mathematical_instruction_evaluation")
        if not judge_succeeded(mathematical_instruction):
            mathematical_instruction = call_judge(client, args.judge_model, mathematical_instruction_system,
                f"【対話ログ】\n{log_text}", MATHEMATICAL_INSTRUCTION_SCHEMA, run_seed + 92, args.judge_retries)
        empathic_instruction = recompute_total(empathic_instruction, [
            "emotion_recognition_score", "cognitive_empathy_score", "emotional_support_score",
        ])
        mathematical_instruction = recompute_total(mathematical_instruction, [
            "mathematical_correctness_score", "error_diagnosis_score",
            "adaptive_scaffolding_score", "learning_verification_score",
            "cognitive_load_control_score",
        ])
        realism_payload = json.dumps({
            "student_profile": row.get("student_profile_used"),
            "initial_state": row.get("initial_student_state"),
            "dialogue": [{key: value for key, value in turn.items() if key != "analysis"}
                         for turn in row.get("dialogue_log", [])],
        }, ensure_ascii=False)
        realism = previous.get("student_realism_evaluation")
        if not judge_succeeded(realism):
            realism = call_judge(client, args.judge_model, realism_system,
                realism_payload, REALISM_JUDGE_SCHEMA, run_seed + 93, args.judge_retries)
        math_result = previous.get("math_judge")
        if not judge_succeeded(math_result):
            math_result = call_judge(client, args.judge_model, math_system,
                f"【生徒の最終解答】\n{row.get('phase2_student_answer', '')}\n\n【模範解答】\n{solution}",
                MATH_JUDGE_SCHEMA, run_seed + 94, args.judge_retries)

        evaluated = dict(row)
        evaluated.update({
            "judge_model": args.judge_model,
            "phase2_is_correct": math_result.get("is_correct"),
            "math_judge": math_result,
            "empathic_instruction_evaluation": empathic_instruction,
            "mathematical_instruction_evaluation": mathematical_instruction,
            "instruction_evaluation_summary": {
                "empathic_instruction_total": empathic_instruction.get("total_score"),
                "mathematical_instruction_total": mathematical_instruction.get("total_score"),
                "combined_total": (
                    empathic_instruction["total_score"] + mathematical_instruction["total_score"]
                    if "total_score" in empathic_instruction and "total_score" in mathematical_instruction
                    else None
                ),
                "maximum_score": 80,
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
