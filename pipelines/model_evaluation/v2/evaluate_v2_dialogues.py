"""生成済みv2対話をJudgeモデルで評価する。"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
SHARED_DIR = BASE_DIR.parent / "shared"

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

EMPATHY_JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "empathy_judge",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "emotion_alignment_score": {"type": "integer"},
                "pedagogical_empathy_score": {"type": "integer"},
                "length_control_score": {"type": "integer"},
                "total_score": {"type": "integer"},
                "empathy_reason": {"type": "string"},
            },
            "required": [
                "emotion_alignment_score", "pedagogical_empathy_score",
                "length_control_score", "total_score", "empathy_reason",
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
    parser = argparse.ArgumentParser(description="生成済みv2対話をJudge APIで評価する")
    parser.add_argument("--input", type=Path, default=BASE_DIR / "data" / "v2_dialogues.jsonl")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "data" / "v2_evaluated_results.jsonl")
    parser.add_argument("--similar-questions", type=Path, default=SHARED_DIR / "questions" / "similar_test_math_questions.jsonl")
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL_NAME", "gpt-5.4"))
    parser.add_argument("--judge-proxy", default=os.getenv("JUDGE_PROXY"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_json_response(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def call_judge(client: OpenAI, model: str, system: str, user: str,
               schema: dict[str, Any], seed: int) -> dict[str, Any]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=1024,
            response_format=schema,
            seed=seed,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Judge returned empty content")
        return parse_json_response(content)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["run_id"]) for row in read_jsonl(path) if row.get("run_id")}


def dialogue_text(dialogue: list[dict[str, Any]]) -> str:
    blocks = []
    for turn in dialogue:
        block = f"{turn['role']}: {turn['content']}"
        if turn.get("analysis"):
            block += f"\nteacher_analysis: {turn['analysis']}"
        blocks.append(block)
    return "\n".join(blocks)


def main() -> None:
    args = parse_args()
    api_key = os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("GPT_API_KEY または OPENAI_API_KEY を設定してください。")
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("生成ログを保持するため、--inputと--outputには別ファイルを指定してください。")

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[:args.limit]
    similar_by_id = {
        str(row.get("source_id") or row.get("id")): row
        for row in read_jsonl(args.similar_questions)
    }
    empathy_system = (SHARED_DIR / "prompts" / "eval_empathy_judge_system.txt").read_text(encoding="utf-8")
    math_system = (SHARED_DIR / "prompts" / "eval_judge_system.txt").read_text(encoding="utf-8")
    realism_system = (BASE_DIR / "prompts" / "v2_student_realism_judge_system.txt").read_text(encoding="utf-8")

    if args.overwrite:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("", encoding="utf-8")
    done = completed_ids(args.output)
    http_client = httpx.Client(proxy=args.judge_proxy) if args.judge_proxy else httpx.Client()
    client = OpenAI(api_key=api_key, http_client=http_client)

    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({
        "phase": "evaluation", "input": str(args.input), "output": str(args.output),
        "judge_model": args.judge_model, "planned_runs": len(rows),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    for index, row in enumerate(tqdm(rows, desc="v2 judging")):
        run_id = str(row["run_id"])
        if run_id in done:
            continue
        source_id = str(row["source_id"])
        similar = similar_by_id.get(source_id, {})
        solution = row.get("similar_solution") or similar.get("similar_solution")
        if not solution:
            raise ValueError(f"類似問題の模範解答がありません: {source_id}")
        run_seed = int(row.get("seed", 42)) + index * 100
        log_text = dialogue_text(row.get("dialogue_log", []))

        empathy = call_judge(
            client, args.judge_model, empathy_system,
            f"【対話ログ】\n{log_text}", EMPATHY_JUDGE_SCHEMA, run_seed + 91,
        )
        realism_payload = json.dumps({
            "student_profile": row.get("student_profile_used"),
            "initial_state": row.get("initial_student_state"),
            "dialogue": row.get("dialogue_log", []),
        }, ensure_ascii=False)
        realism = call_judge(
            client, args.judge_model, realism_system,
            realism_payload, REALISM_JUDGE_SCHEMA, run_seed + 92,
        )
        math_result = call_judge(
            client, args.judge_model, math_system,
            f"【生徒の最終解答】\n{row.get('phase2_student_answer', '')}\n\n【模範解答】\n{solution}",
            MATH_JUDGE_SCHEMA, run_seed + 93,
        )

        evaluated = dict(row)
        evaluated.update({
            "judge_model": args.judge_model,
            "phase2_is_correct": math_result.get("is_correct"),
            "math_judge": math_result,
            "empathy_evaluation": empathy,
            "student_realism_evaluation": realism,
        })
        append_jsonl(args.output, evaluated)

    http_client.close()
    print(f"評価完了: {args.output}")
    print(f"設定: {manifest_path}")


if __name__ == "__main__":
    main()
