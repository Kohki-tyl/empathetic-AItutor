"""Teacher/student分離型のv2対話評価。

教師だけを評価条件間で変更し、生徒シミュレーター、Judge、問題、profile、seedを固定する。
Phase 2へは対話全文ではなく、生徒モデルが更新した学習状態だけを渡す。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    teacher_base_url: str
    teacher_model: str
    student_base_url: str
    student_model: str
    judge_model: str
    max_turns: int
    student_temperature: float
    teacher_temperature: float
    phase2_temperature: float
    seed: int


STUDENT_TURN_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "student_turn",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "state_after": {
                    "type": "object",
                    "properties": {
                        "understanding_level": {"type": "integer", "minimum": 0, "maximum": 4},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "active_misconception": {"type": "string"},
                        "emotion": {
                            "type": "string",
                            "enum": ["engaged", "curious", "neutral", "confused", "frustrated", "anxious", "relieved", "proud"],
                        },
                        "acquired_knowledge": {"type": "array", "items": {"type": "string"}},
                        "remaining_unknowns": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "understanding_level", "confidence", "active_misconception",
                        "emotion", "acquired_knowledge", "remaining_unknowns",
                    ],
                    "additionalProperties": False,
                },
                "state_update_reason": {"type": "string"},
                "utterance": {"type": "string"},
            },
            "required": ["state_after", "state_update_reason", "utterance"],
            "additionalProperties": False,
        },
    },
}

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
            "required": ["emotion_alignment_score", "pedagogical_empathy_score", "length_control_score", "total_score", "empathy_reason"],
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
                "style_violation_count", "implausible_update_count", "blind_agreement_count", "judge_reason",
            ],
            "additionalProperties": False,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分離した教師・生徒モデルでv2評価を実行する")
    parser.add_argument("--teacher-model", default=os.getenv("TEACHER_MODEL_NAME"), required=os.getenv("TEACHER_MODEL_NAME") is None)
    parser.add_argument("--teacher-base-url", default=os.getenv("TEACHER_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--student-model", default=os.getenv("STUDENT_MODEL_NAME"), required=os.getenv("STUDENT_MODEL_NAME") is None)
    parser.add_argument("--student-base-url", default=os.getenv("STUDENT_BASE_URL", "http://localhost:8001/v1"))
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL_NAME", "gpt-5.4"))
    parser.add_argument("--judge-proxy", default=os.getenv("JUDGE_PROXY"))
    parser.add_argument("--questions", type=Path, default=BASE_DIR / "questions" / "test_math_questions.jsonl")
    parser.add_argument("--similar-questions", type=Path, default=BASE_DIR / "questions" / "similar_test_math_questions.jsonl")
    parser.add_argument("--profiles", type=Path, default=BASE_DIR / "prompts" / "v2_student_profiles.json")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "v2_evaluation_results.jsonl")
    parser.add_argument("--limit", type=int, help="先頭から実行する問題数。パイロットでは20を推奨")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--student-temperature", type=float, default=0.6)
    parser.add_argument("--teacher-temperature", type=float, default=0.2)
    parser.add_argument("--phase2-temperature", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true", help="既存出力を消して最初から実行")
    parser.add_argument("--skip-judges", action="store_true", help="対話生成だけを検証する")
    return parser.parse_args()


def read_text(name: str) -> str:
    return (BASE_DIR / "prompts" / name).read_text(encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def call_model(client: OpenAI, model: str, messages: list[dict[str, str]], temperature: float,
               max_tokens: int = 512, response_format: dict[str, Any] | None = None,
               seed: int | None = None) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format
    if seed is not None:
        kwargs["seed"] = seed
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    if not content:
        raise ValueError(f"{model} returned empty content")
    return content.strip()


def parse_json_response(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def profile_text(profile: dict[str, Any]) -> str:
    return "\n".join(f"- {key}: {value}" for key, value in profile.items())


def initial_state(profile: dict[str, Any]) -> dict[str, Any]:
    confidence = 0.35 if profile.get("confidence_bias") == "underconfident" else 0.55
    return {
        "understanding_level": max(0, min(4, int(profile["ability_level"]) - 1)),
        "confidence": confidence,
        "active_misconception": profile["target_misconception"],
        "emotion": "anxious" if profile.get("emotional_reactivity") == "high" else "neutral",
        "acquired_knowledge": [],
        "remaining_unknowns": [profile["unknown_knowledge"]],
    }


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["run_id"]) for row in read_jsonl(path) if row.get("run_id")}


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def judge_or_error(client: OpenAI, model: str, system: str, user: str, schema: dict[str, Any], seed: int) -> dict[str, Any]:
    try:
        raw = call_model(client, model, [{"role": "system", "content": system}, {"role": "user", "content": user}], 0.0, 1024, schema, seed)
        return parse_json_response(raw)
    except Exception as exc:  # 評価を中断せず、失敗をログへ残す
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    args = parse_args()
    config = Config(
        args.teacher_base_url, args.teacher_model, args.student_base_url, args.student_model,
        args.judge_model, args.max_turns, args.student_temperature, args.teacher_temperature,
        args.phase2_temperature, args.seed,
    )
    if config.teacher_base_url == config.student_base_url and config.teacher_model == config.student_model:
        raise SystemExit("教師と生徒が同じURL・モデルです。比較の交絡を避けるため別モデルを指定してください。")

    api_key = os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not args.skip_judges and not api_key:
        raise SystemExit("Judge用に GPT_API_KEY または OPENAI_API_KEY を設定してください。")

    teacher_client = OpenAI(api_key="EMPTY", base_url=config.teacher_base_url)
    student_client = OpenAI(api_key="EMPTY", base_url=config.student_base_url)
    judge_http = httpx.Client(proxy=args.judge_proxy) if args.judge_proxy else httpx.Client()
    judge_client = OpenAI(api_key=api_key or "SKIPPED", http_client=judge_http)

    teacher_system = read_text("v2_teacher_system.txt")
    student_template = read_text("v2_student_system.txt")
    phase2_template = read_text("v2_phase2_student_system.txt")
    empathy_system = read_text("eval_empathy_judge_system.txt")
    math_system = read_text("eval_judge_system.txt")
    realism_system = read_text("v2_student_realism_judge_system.txt")
    profiles = json.loads(args.profiles.read_text(encoding="utf-8"))

    originals = read_jsonl(args.questions)
    similar_by_id = {str(row.get("source_id") or row.get("id")): row for row in read_jsonl(args.similar_questions)}
    pairs = [(row, similar_by_id.get(str(row.get("id") or row.get("source_id")))) for row in originals]
    pairs = [(original, similar) for original, similar in pairs if similar is not None]
    if args.limit is not None:
        pairs = pairs[:args.limit]

    if args.overwrite:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("", encoding="utf-8")
    done = completed_ids(args.output)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({
        "config": asdict(config),
        "questions": str(args.questions), "similar_questions": str(args.similar_questions),
        "profiles": str(args.profiles), "planned_runs": len(pairs), "skip_judges": args.skip_judges,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    rng = random.Random(config.seed)
    profile_offset = rng.randrange(len(profiles))
    for index, (original, similar) in enumerate(tqdm(pairs, desc="v2 evaluation")):
        source_id = str(original.get("id") or original.get("source_id"))
        run_id = f"{source_id}:seed-{config.seed}"
        if run_id in done:
            continue
        profile = profiles[(index + profile_offset) % len(profiles)]
        formatted_profile = profile_text(profile)
        state = initial_state(profile)
        problem = original["translated_question"]
        dialogue: list[dict[str, Any]] = []
        teacher_history = [
            {"role": "system", "content": teacher_system},
            {"role": "user", "content": f"問題: {problem}\n\n生徒の最初の発話を待ち、対話指導を開始してください。"},
        ]
        last_teacher = "まだ教師からの説明はありません。問題を読み、自分の理解の範囲で取り組み始めてください。"
        is_completed = False
        generation_error: str | None = None
        run_seed = config.seed + index * 100

        for turn in range(config.max_turns):
            student_input = {
                "problem": problem,
                "turn": turn + 1,
                "state_before": state,
                "latest_teacher_utterance": last_teacher,
                "recent_dialogue": dialogue[-4:],
            }
            try:
                raw = call_model(
                    student_client, config.student_model,
                    [{"role": "system", "content": student_template.replace("{STUDENT_PROFILE}", formatted_profile)},
                     {"role": "user", "content": json.dumps(student_input, ensure_ascii=False)}],
                    config.student_temperature, 700, STUDENT_TURN_SCHEMA, run_seed + turn * 2,
                )
                student_turn = parse_json_response(raw)
                state = student_turn["state_after"]
                utterance = student_turn["utterance"].strip()
                dialogue.append({"role": "student", "content": utterance, "state_after": state,
                                 "state_update_reason": student_turn["state_update_reason"]})
                teacher_history.append({"role": "user", "content": utterance})

                teacher_raw = call_model(
                    teacher_client, config.teacher_model, teacher_history,
                    config.teacher_temperature, seed=run_seed + turn * 2 + 1,
                )
                is_completed = "[指導完了]" in teacher_raw
                last_teacher = teacher_raw.replace("[指導完了]", "").strip()
                dialogue.append({"role": "teacher", "content": last_teacher})
                teacher_history.append({"role": "assistant", "content": teacher_raw})
                if is_completed:
                    break
            except Exception as exc:
                generation_error = f"{type(exc).__name__}: {exc}"
                break

        phase2_input = {
            "learning_state_after_phase1": state,
            "new_problem": similar["similar_question"],
        }
        try:
            phase2_answer = call_model(
                student_client, config.student_model,
                [{"role": "system", "content": phase2_template.replace("{STUDENT_PROFILE}", formatted_profile)},
                 {"role": "user", "content": json.dumps(phase2_input, ensure_ascii=False)}],
                config.phase2_temperature, 256, seed=run_seed + 90,
            )
        except Exception as exc:
            phase2_answer = ""
            generation_error = generation_error or f"Phase2 {type(exc).__name__}: {exc}"

        dialogue_text = "\n".join(f"{item['role']}: {item['content']}" for item in dialogue)
        if args.skip_judges:
            empathy, realism, math_result = {}, {}, {}
        else:
            empathy = judge_or_error(judge_client, config.judge_model, empathy_system, f"【対話ログ】\n{dialogue_text}", EMPATHY_JUDGE_SCHEMA, run_seed + 91)
            realism_payload = json.dumps({"student_profile": profile, "initial_state": initial_state(profile), "dialogue": dialogue}, ensure_ascii=False)
            realism = judge_or_error(judge_client, config.judge_model, realism_system, realism_payload, REALISM_JUDGE_SCHEMA, run_seed + 92)
            math_result = judge_or_error(
                judge_client, config.judge_model, math_system,
                f"【生徒の最終解答】\n{phase2_answer}\n\n【模範解答】\n{similar['similar_solution']}", MATH_JUDGE_SCHEMA, run_seed + 93,
            )

        append_jsonl(args.output, {
            "run_id": run_id, "source_id": source_id, "seed": config.seed,
            "teacher_model": config.teacher_model, "student_model": config.student_model,
            "student_profile_used": profile, "initial_student_state": initial_state(profile),
            "final_student_state": state, "phase1_turns": sum(item["role"] == "student" for item in dialogue),
            "phase1_is_completed": is_completed, "phase2_student_answer": phase2_answer,
            "phase2_is_correct": math_result.get("is_correct"), "math_judge": math_result,
            "empathy_evaluation": empathy, "student_realism_evaluation": realism,
            "dialogue_log": dialogue, "generation_error": generation_error,
        })

    judge_http.close()
    print(f"完了: {args.output}")
    print(f"設定: {manifest_path}")


if __name__ == "__main__":
    main()
