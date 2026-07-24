"""Teacher/student分離型のv2対話・Phase 2解答生成。

教師だけを実験条件間で変更し、生徒モデル、問題、profile、seedを固定する。
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

from openai import OpenAI
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
SHARED_DIR = BASE_DIR.parent / "shared"
REPO_ROOT = BASE_DIR.parents[2]
DEFAULT_STUDENT_MODEL = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
TRANSFER_MODE = "profile_update"


@dataclass(frozen=True)
class Config:
    teacher_base_url: str
    teacher_model: str
    student_base_url: str
    student_model: str
    max_turns: int
    student_temperature: float
    teacher_temperature: float
    phase2_temperature: float
    seed: int
    transfer_mode: str


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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分離した教師・生徒モデルでv2対話を生成する")
    parser.add_argument("--teacher-model", default=os.getenv("TEACHER_MODEL_NAME"), required=os.getenv("TEACHER_MODEL_NAME") is None)
    parser.add_argument("--teacher-base-url", default=os.getenv("TEACHER_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument(
        "--student-model",
        default=os.getenv("STUDENT_MODEL_NAME", DEFAULT_STUDENT_MODEL),
        help=f"固定生徒モデル（既定: {DEFAULT_STUDENT_MODEL}）",
    )
    parser.add_argument("--student-base-url", default=os.getenv("STUDENT_BASE_URL", "http://localhost:8001/v1"))
    parser.add_argument("--questions", type=Path, default=SHARED_DIR / "questions" / "test_math_questions.jsonl")
    parser.add_argument("--similar-questions", type=Path, default=SHARED_DIR / "questions" / "similar_test_math_questions.jsonl")
    parser.add_argument("--profiles", type=Path, default=BASE_DIR / "prompts" / "v2_student_profiles.json")
    parser.add_argument(
        "--teacher-system-prompt", type=Path,
        default=BASE_DIR / "prompts" / "v2_teacher_system.txt",
        help="CoTモデルでは prompts/v2_cot_teacher_system.txt を指定する",
    )
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiments" / "test_v2" / "dialogues.jsonl")
    parser.add_argument("--limit", type=int, help="先頭から実行する問題数。パイロットでは20を推奨")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--student-temperature", type=float, default=0.6)
    parser.add_argument("--teacher-temperature", type=float, default=0.2)
    parser.add_argument("--phase2-temperature", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true", help="既存出力を消して最初から実行")
    return parser.parse_args()


def read_text(name: str, shared: bool = False) -> str:
    prompt_dir = SHARED_DIR / "prompts" if shared else BASE_DIR / "prompts"
    return (prompt_dir / name).read_text(encoding="utf-8")


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


def parse_teacher_response(raw: str) -> tuple[str, str | None, bool]:
    """CoT形式から生徒向け発話だけを取り出す。通常形式も受け付ける。"""
    analysis_match = re.search(r"<analysis>\s*(.*?)\s*</analysis>", raw, flags=re.DOTALL | re.IGNORECASE)
    final_match = re.search(r"<final>\s*(.*?)\s*</final>", raw, flags=re.DOTALL | re.IGNORECASE)
    analysis = analysis_match.group(1).strip() if analysis_match else None
    final = final_match.group(1).strip() if final_match else raw.strip()
    completed = "[指導完了]" in final
    return final.replace("[指導完了]", "").strip(), analysis, completed


def profile_text(profile: dict[str, Any]) -> str:
    lines = []
    for key, value in profile.items():
        if isinstance(value, list):
            lines.append(f"- {key}:")
            lines.extend(f"  - {item}" for item in value)
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def initial_state(profile: dict[str, Any]) -> dict[str, Any]:
    confidence = 0.35 if profile.get("confidence_bias") == "underconfident" else 0.55
    unknown_knowledge = profile["unknown_knowledge"]
    if isinstance(unknown_knowledge, str):
        unknown_knowledge = [unknown_knowledge]
    return {
        "understanding_level": max(0, min(4, int(profile["ability_level"]) - 1)),
        "confidence": confidence,
        "active_misconception": profile["target_misconception"],
        "emotion": "anxious" if profile.get("emotional_reactivity") == "high" else "neutral",
        "acquired_knowledge": [],
        "remaining_unknowns": list(unknown_knowledge),
    }


def build_phase2_input(
    profile: dict[str, Any],
    state: dict[str, Any],
    similar_question: str,
) -> dict[str, Any]:
    return {
        "updated_student_profile": {
            "base_profile": profile,
            "learning_state_after_phase1": state,
        },
        "new_problem": similar_question,
    }


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["run_id"]) for row in read_jsonl(path) if row.get("run_id")}


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    config = Config(
        args.teacher_base_url, args.teacher_model, args.student_base_url, args.student_model,
        args.max_turns, args.student_temperature, args.teacher_temperature,
        args.phase2_temperature, args.seed, TRANSFER_MODE,
    )
    if config.teacher_base_url == config.student_base_url and config.teacher_model == config.student_model:
        raise SystemExit("教師と生徒が同じURL・モデルです。比較の交絡を避けるため別モデルを指定してください。")

    teacher_client = OpenAI(api_key="EMPTY", base_url=config.teacher_base_url)
    student_client = OpenAI(api_key="EMPTY", base_url=config.student_base_url)

    teacher_system = args.teacher_system_prompt.read_text(encoding="utf-8").strip()
    student_template = read_text("v2_student_system.txt")
    phase2_template = read_text("v2_phase2_student_system.txt")
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
        "profiles": str(args.profiles), "teacher_system_prompt": str(args.teacher_system_prompt),
        "phase": "generation", "planned_runs": len(pairs),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    rng = random.Random(config.seed)
    profile_offset = rng.randrange(len(profiles))
    for index, (original, similar) in enumerate(tqdm(pairs, desc="test v2 generation")):
        source_id = str(original.get("id") or original.get("source_id"))
        run_id = f"{source_id}:{config.transfer_mode}:seed-{config.seed}"
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
                last_teacher, teacher_analysis, is_completed = parse_teacher_response(teacher_raw)
                teacher_log = {"role": "teacher", "content": last_teacher}
                if teacher_analysis is not None:
                    teacher_log["analysis"] = teacher_analysis
                dialogue.append(teacher_log)
                teacher_history.append({"role": "assistant", "content": teacher_raw})
                if is_completed:
                    break
            except Exception as exc:
                generation_error = f"{type(exc).__name__}: {exc}"
                break

        phase2_input = build_phase2_input(
            profile, state, similar["similar_question"],
        )
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

        append_jsonl(args.output, {
            "run_id": run_id, "source_id": source_id, "seed": config.seed,
            "transfer_mode": config.transfer_mode,
            "teacher_model": config.teacher_model, "student_model": config.student_model,
            "student_profile_used": profile, "initial_student_state": initial_state(profile),
            "final_student_state": state, "phase1_turns": sum(item["role"] == "student" for item in dialogue),
            "phase1_is_completed": is_completed, "phase2_student_answer": phase2_answer,
            "similar_question": similar["similar_question"],
            "similar_solution": similar["similar_solution"],
            "dialogue_log": dialogue, "generation_error": generation_error,
        })

    print(f"完了: {args.output}")
    print(f"設定: {manifest_path}")


if __name__ == "__main__":
    main()
