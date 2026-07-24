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
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
SHARED_DIR = BASE_DIR.parent / "shared"
DEFAULT_STUDENT_MODEL = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
TRANSFER_MODE = "profile_update"
STUDENT_STATE_KEYS = {
    "understanding_level", "confidence", "active_misconception", "emotion",
    "acquired_knowledge", "remaining_unknowns",
}
STUDENT_LEAK_MARKERS = (
    '"state_before"', '"state_after"', '"latest_teacher_utterance"',
    '"recent_dialogue"', '"problem"', '"utterance"', '<analysis>', '<final>',
)


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
    parser.add_argument("--output", type=Path, default=BASE_DIR / "data" / "dialogues.jsonl")
    parser.add_argument("--limit", type=int, help="先頭から実行する問題数。パイロットでは20を推奨")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--student-temperature", type=float, default=0.6)
    parser.add_argument("--teacher-temperature", type=float, default=0.2)
    parser.add_argument("--phase2-temperature", type=float, default=0.0)
    parser.add_argument("--response-retries", type=int, default=3)
    parser.add_argument("--teacher-checkpoint", help="実際にロードした教師base checkpoint/HF ID")
    parser.add_argument("--teacher-adapter", help="実際にロードした教師adapterの絶対パス")
    parser.add_argument("--student-checkpoint", help="実際にロードした生徒checkpoint/HF ID")
    parser.add_argument("--overwrite", action="store_true", help="既存出力を消して最初から実行")
    return parser.parse_args()


def read_text(name: str, shared: bool = False) -> str:
    prompt_dir = SHARED_DIR / "prompts" if shared else BASE_DIR / "prompts"
    return (prompt_dir / name).read_text(encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        if normalized and normalized[-1]["role"] == message["role"]:
            normalized[-1]["content"] += "\n\n" + message["content"]
        else:
            normalized.append(dict(message))
    return normalized


def teacher_user_message(problem: str, utterance: str, turn: int) -> str:
    if turn == 0:
        return f"問題: {problem}\n\n生徒発話: {utterance}"
    return utterance


def call_model(client: OpenAI, model: str, messages: list[dict[str, str]], temperature: float,
               max_tokens: int = 512, response_format: dict[str, Any] | None = None,
               seed: int | None = None) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": normalize_messages(messages),
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
    def loads_lenient(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            repaired = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', value)
            return json.loads(repaired)

    try:
        parsed = loads_lenient(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        parsed = loads_lenient(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")
    return parsed


def validate_student_state(state: Any, previous_state: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(state, dict) or not STUDENT_STATE_KEYS.issubset(state):
        raise ValueError("state_after is missing required fields")
    state = {key: state[key] for key in STUDENT_STATE_KEYS}
    normalized_fields: list[str] = []
    level = state["understanding_level"]
    confidence = state["confidence"]
    if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 4:
        raise ValueError("understanding_level is invalid")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("confidence is invalid")
    if abs(level - int(previous_state["understanding_level"])) > 1:
        raise ValueError("understanding_level changed by more than one step")
    if not isinstance(state["active_misconception"], str) or not state["active_misconception"].strip():
        state["active_misconception"] = str(previous_state["active_misconception"])
        normalized_fields.append("active_misconception:preserved_previous")
    else:
        state["active_misconception"] = state["active_misconception"].strip()
    if state["emotion"] not in STUDENT_TURN_SCHEMA["json_schema"]["schema"]["properties"]["state_after"]["properties"]["emotion"]["enum"]:
        raise ValueError("emotion is invalid")
    for field in ("acquired_knowledge", "remaining_unknowns"):
        if not isinstance(state[field], list) or not all(isinstance(item, str) and item.strip() for item in state[field]):
            raise ValueError(f"{field} is invalid")
    return state, normalized_fields


def validate_student_utterance(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("utterance is not a string")
    utterance = value.strip()
    for _ in range(2):
        if not utterance.startswith("{"):
            break
        try:
            nested = parse_json_response(utterance)
        except (json.JSONDecodeError, ValueError):
            break
        nested_utterance = nested.get("utterance")
        if not isinstance(nested_utterance, str) or nested_utterance.strip() == utterance:
            break
        utterance = nested_utterance.strip()
    lowered = utterance.lower()
    if not utterance or len(utterance) > 500:
        raise ValueError("utterance is empty or too long")
    if utterance.startswith(("{", "[")) or any(marker in lowered for marker in STUDENT_LEAK_MARKERS):
        raise ValueError("utterance contains JSON, state, or hidden-tag leakage")
    return utterance


def parse_student_turn(raw: str, previous_state: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_json_response(raw)
    required = {"state_after", "state_update_reason", "utterance"}
    if not required.issubset(parsed):
        raise ValueError(f"student response is missing required fields; returned keys={sorted(parsed)}")
    normalized = {key: parsed[key] for key in required}
    normalized["state_after"], state_normalizations = validate_student_state(
        normalized["state_after"], previous_state,
    )
    normalized["_state_normalizations"] = state_normalizations
    normalized["utterance"] = validate_student_utterance(normalized["utterance"])
    if not isinstance(normalized["state_update_reason"], str) or not normalized["state_update_reason"].strip():
        raise ValueError("state_update_reason is invalid")
    if "構造化応答を解析できなかった" in normalized["state_update_reason"]:
        raise ValueError("student model reported a structured-response parse failure")
    return normalized


def parse_teacher_response(raw: str) -> tuple[str, str | None, bool]:
    """CoT形式から生徒向け発話だけを取り出す。通常形式も受け付ける。"""
    analysis_match = re.search(r"<analysis>\s*(.*?)(?:\s*</analysis>|\s*<final>)", raw, flags=re.DOTALL | re.IGNORECASE)
    final_match = re.search(r"<final>\s*(.*?)(?:\s*</final>|\Z)", raw, flags=re.DOTALL | re.IGNORECASE)
    analysis = analysis_match.group(1).strip() if analysis_match else None
    if final_match:
        final = final_match.group(1).strip()
    elif re.search(r"<analysis>", raw, flags=re.IGNORECASE):
        raise ValueError("teacher response contains analysis without a final answer")
    else:
        final = raw.strip()
    if not final or re.search(r"</?analysis>|</?final>", final, flags=re.IGNORECASE):
        raise ValueError("teacher final answer is empty or leaks hidden tags")
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


def generation_succeeded(row: dict[str, Any]) -> bool:
    return bool(row.get("run_id")) and not row.get("generation_error") and int(row.get("phase1_turns", 0)) > 0


def call_and_parse_student(
    client: OpenAI, model: str, messages: list[dict[str, str]], temperature: float,
    previous_state: dict[str, Any], seed: int, retries: int,
) -> tuple[dict[str, Any], int]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            retry_messages = list(messages)
            if attempt:
                retry_messages.append({
                    "role": "user",
                    "content": (
                        "前回の出力形式が不正でした。problem、turn、state_before等をコピーせず、"
                        "state_after、state_update_reason、utteranceの3キーだけを持つJSONを返してください。"
                    ),
                })
            raw = call_model(client, model, retry_messages, temperature, 700, STUDENT_TURN_SCHEMA, seed + attempt)
            return parse_student_turn(raw, previous_state), attempt
        except Exception as exc:
            last_error = exc
            time.sleep(0.2 * (attempt + 1))
    raise ValueError(f"student structured response failed after {retries} attempts: {last_error}")


def call_and_parse_teacher(
    client: OpenAI, model: str, messages: list[dict[str, str]], temperature: float,
    seed: int, retries: int,
) -> tuple[str, str | None, bool, str, int]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            raw = call_model(client, model, messages, temperature, 1024, seed=seed + attempt)
            final, analysis, completed = parse_teacher_response(raw)
            return final, analysis, completed, raw, attempt
        except Exception as exc:
            last_error = exc
            time.sleep(0.2 * (attempt + 1))
    raise ValueError(f"teacher response failed after {retries} attempts: {last_error}")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--num-shardsと--shard-indexの組み合わせが不正です。")
    if args.response_retries < 1:
        raise SystemExit("--response-retriesは1以上にしてください。")
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
    pairs = [pair for index, pair in enumerate(pairs) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        pairs = pairs[:args.limit]

    if args.overwrite:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("", encoding="utf-8")
    existing_rows = read_jsonl(args.output) if args.output.exists() else []
    existing_by_id = {str(row["run_id"]): row for row in existing_rows if row.get("run_id")}
    done = {run_id for run_id, row in existing_by_id.items() if generation_succeeded(row)}
    failed_before_resume = sum(not generation_succeeded(row) for row in existing_by_id.values())
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({
        "config": asdict(config),
        "questions": str(args.questions), "similar_questions": str(args.similar_questions),
        "profiles": str(args.profiles), "teacher_system_prompt": str(args.teacher_system_prompt),
        "phase": "generation", "planned_runs": len(pairs),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "loaded_models": {
            "teacher_checkpoint": args.teacher_checkpoint or args.teacher_model,
            "teacher_adapter": args.teacher_adapter,
            "teacher_served_model": args.teacher_model,
            "student_checkpoint": args.student_checkpoint or args.student_model,
            "student_served_model": args.student_model,
        },
        "response_retries": args.response_retries,
        "resume": {
            "existing_records": len(existing_by_id),
            "successful_records_skipped": len(done),
            "failed_records_planned_for_retry": failed_before_resume,
            "overwrite": args.overwrite,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    rng = random.Random(config.seed)
    profile_offset = rng.randrange(len(profiles))
    for index, (original, similar) in enumerate(tqdm(pairs, desc="test v2 generation")):
        source_id = str(original.get("id") or original.get("source_id"))
        run_id = f"{source_id}:{config.transfer_mode}:seed-{config.seed}"
        if run_id in done:
            continue
        previous_record = existing_by_id.get(run_id, {})
        previous_attempt = int(previous_record.get("generation_attempt", 1 if previous_record else 0))
        generation_attempt = previous_attempt + 1
        profile = profiles[(index + profile_offset) % len(profiles)]
        formatted_profile = profile_text(profile)
        state = initial_state(profile)
        problem = original["translated_question"]
        dialogue: list[dict[str, Any]] = []
        teacher_history = [{"role": "system", "content": teacher_system}]
        last_teacher = "まだ教師からの説明はありません。問題を読み、自分の理解の範囲で取り組み始めてください。"
        is_completed = False
        generation_error: str | None = None
        validation_retries = {"student": 0, "teacher": 0}
        run_seed = config.seed + index * 100 + (generation_attempt - 1) * 1_000_000

        for turn in range(config.max_turns):
            student_input = {
                "problem": problem,
                "turn": turn + 1,
                "state_before": state,
                "latest_teacher_utterance": last_teacher,
                "recent_dialogue": dialogue[-4:],
            }
            try:
                previous_state = state
                student_turn, retry_count = call_and_parse_student(
                    student_client, config.student_model,
                    [{"role": "system", "content": student_template.replace("{STUDENT_PROFILE}", formatted_profile)},
                     {"role": "user", "content": json.dumps(student_input, ensure_ascii=False)}],
                    config.student_temperature, state, run_seed + turn * 20, args.response_retries,
                )
                validation_retries["student"] += retry_count
                state_normalizations = student_turn.pop("_state_normalizations", [])
                state = student_turn["state_after"]
                utterance = student_turn["utterance"].strip()
                dialogue.append({
                    "role": "student",
                    "content": utterance,
                    "state_after": state,
                    "state_update_validated": True,
                    "state_changed": state != previous_state,
                    "state_normalizations": state_normalizations,
                    "state_update_reason": student_turn.get(
                        "state_update_reason",
                        "生徒モデルの応答で更新理由が省略されました。",
                    ),
                })
                teacher_user_content = teacher_user_message(problem, utterance, turn)
                teacher_history.append({"role": "user", "content": teacher_user_content})

                last_teacher, teacher_analysis, is_completed, teacher_raw, retry_count = call_and_parse_teacher(
                    teacher_client, config.teacher_model, teacher_history,
                    config.teacher_temperature, run_seed + turn * 20 + 10, args.response_retries,
                )
                validation_retries["teacher"] += retry_count
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

        record = {
            "run_id": run_id, "source_id": source_id, "seed": config.seed,
            "generation_attempt": generation_attempt,
            "transfer_mode": config.transfer_mode,
            "teacher_model": config.teacher_model, "student_model": config.student_model,
            "student_profile_used": profile, "initial_student_state": initial_state(profile),
            "final_student_state": state, "phase1_turns": sum(item["role"] == "student" for item in dialogue),
            "phase1_is_completed": is_completed, "phase2_student_answer": phase2_answer,
            "similar_question": similar["similar_question"],
            "similar_solution": similar["similar_solution"],
            "dialogue_log": dialogue, "generation_error": generation_error,
            "loaded_models": {
                "teacher_checkpoint": args.teacher_checkpoint or args.teacher_model,
                "teacher_adapter": args.teacher_adapter,
                "teacher_served_model": config.teacher_model,
                "student_checkpoint": args.student_checkpoint or args.student_model,
                "student_served_model": config.student_model,
            },
            "validation_retries": validation_retries,
        }
        existing_by_id[run_id] = record
        write_jsonl(args.output, [existing_by_id[key] for key in sorted(existing_by_id)])

    print(f"完了: {args.output}")
    print(f"設定: {manifest_path}")


if __name__ == "__main__":
    main()
