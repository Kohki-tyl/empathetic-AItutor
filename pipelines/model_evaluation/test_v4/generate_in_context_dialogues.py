"""v4コーパス準拠の生徒を用いるインコンテキスト転移テスト生成。

教師だけを実験条件間で変更し、生徒モデル、プロフィール、初期感情、問題、seedを固定する。
Phase 2へは元問題とPhase 1の自然言語対話だけを渡し、構造化学習状態は渡さない。
"""

from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_STUDENT_MODEL = "tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2"
DEFAULT_STUDENT_REVISION = "496cd5558fef4af1d426e96327d7a74681063280"
TRANSFER_MODE = "v4_in_context"
STUDENT_STATE_KEYS = {
    "understanding_level", "confidence", "active_misconception", "emotion",
    "acquired_knowledge", "remaining_unknowns",
}
STUDENT_MODEL_STATE_KEYS = STUDENT_STATE_KEYS - {"acquired_knowledge"}
STUDENT_EMOTIONS = [
    "engaged", "curious", "neutral", "confused", "frustrated", "anxious",
    "bored", "eureka", "relieved", "proud",
]
STUDENT_RESPONSE_STAGES = ["observation", "attempt", "help_seeking", "answer"]
EMOTION_TRANSITIONS = {
    "engaged": {"curious", "confused", "eureka"},
    "curious": {"engaged", "confused", "eureka"},
    "neutral": {"curious", "engaged", "confused", "anxious"},
    "confused": {"engaged", "frustrated", "anxious", "eureka"},
    "frustrated": {"confused", "bored"},
    "bored": {"frustrated", "neutral", "engaged"},
    "anxious": {"confused", "engaged", "frustrated", "relieved"},
    "eureka": {"engaged", "proud", "relieved"},
    "relieved": {"neutral", "engaged", "proud"},
    "proud": {"neutral", "engaged"},
}
STUDENT_LEAK_MARKERS = (
    '"state_before"', '"state_after"', '"latest_teacher_utterance"',
    '"recent_dialogue"', '"problem"', '"utterance"', '<analysis>', '<final>',
)


@dataclass(frozen=True)
class Config:
    teacher_base_url: str
    teacher_model: str
    teacher_serving_mode: str
    teacher_revision: str | None
    student_base_url: str
    student_model: str
    student_revision: str
    max_turns: int
    student_temperature: float
    student_top_p: float
    student_top_k: int
    student_min_p: float
    student_max_tokens: int
    teacher_temperature: float
    phase2_temperature: float
    seed: int
    transfer_mode: str


STUDENT_TURN_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "v4_test_student_turn",
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
                            "enum": STUDENT_EMOTIONS,
                        },
                        "remaining_unknowns": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "understanding_level", "confidence", "active_misconception",
                        "emotion", "remaining_unknowns",
                    ],
                    "additionalProperties": False,
                },
                "newly_acquired_knowledge": {
                    "type": "array", "items": {"type": "string"},
                },
                "response_stage": {"type": "string", "enum": STUDENT_RESPONSE_STAGES},
                "knowledge_used": {"type": "array", "items": {"type": "string"}},
                "state_update_reason": {"type": "string"},
                "utterance": {"type": "string"},
            },
            "required": [
                "state_after", "newly_acquired_knowledge", "response_stage", "knowledge_used",
                "state_update_reason", "utterance",
            ],
            "additionalProperties": False,
        },
    },
}

PHASE2_TRANSFER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "v4_phase2_transfer_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "knowledge_sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_type": {
                                "type": "string",
                                "enum": ["prior_knowledge", "phase1_teacher"],
                            },
                            "source_text": {"type": "string"},
                        },
                        "required": ["source_type", "source_text"],
                        "additionalProperties": False,
                    },
                },
                "application_summary": {"type": "string"},
            },
            "required": ["answer", "knowledge_sources", "application_summary"],
            "additionalProperties": False,
        },
    },
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", default=os.getenv("TEACHER_MODEL_NAME"), required=os.getenv("TEACHER_MODEL_NAME") is None)
    parser.add_argument("--teacher-base-url", default=os.getenv("TEACHER_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument(
        "--student-model",
        default=os.getenv("STUDENT_MODEL_NAME", DEFAULT_STUDENT_MODEL),
        help=f"固定生徒モデル（既定: {DEFAULT_STUDENT_MODEL}）",
    )
    parser.add_argument("--student-base-url", default=os.getenv("STUDENT_BASE_URL", "http://localhost:8001/v1"))
    parser.add_argument(
        "--student-api-provider", choices=("vllm", "openai"), default="vllm",
        help="生徒APIの種類。OpenAI APIではvLLM固有sampling引数を送信しない。",
    )
    parser.add_argument(
        "--student-reasoning-effort", default="none",
        help="OpenAI生徒モデルへ渡すreasoning_effort。",
    )
    parser.add_argument("--student-request-timeout", type=float, default=180.0)
    parser.add_argument("--questions", type=Path, default=SHARED_DIR / "questions" / "test_math_questions.jsonl")
    parser.add_argument("--similar-questions", type=Path, default=SHARED_DIR / "questions" / "similar_test_math_questions.jsonl")
    parser.add_argument(
        "--excluded-question-ids", type=Path,
        default=SHARED_DIR / "questions" / "excluded_test_question_ids.json",
        help="評価条件間で共通除外する問題IDのJSON",
    )
    parser.add_argument("--profiles", type=Path, default=BASE_DIR / "prompts" / "student_profiles.json")
    parser.add_argument(
        "--problem-profile-assignments", type=Path,
        default=BASE_DIR / "prompts" / "problem_profile_assignments.jsonl",
        help="事前生成した問題・E2/E3プロフィール・初期感情対応表",
    )
    parser.add_argument(
        "--problem-selection", type=Path,
        default=BASE_DIR / "prompts" / "test_60_primary_selection.json",
        help="一次60問（各scope 15件）。確認評価ではtest_60_confirmation_selection.jsonを指定",
    )
    parser.add_argument(
        "--initial-emotions", type=Path,
        default=BASE_DIR / "prompts" / "initial_emotions.json",
    )
    parser.add_argument(
        "--teacher-system-prompt", type=Path,
        default=BASE_DIR / "prompts" / "teacher_system.txt",
        help="v4 SFT形式と同じ教師プロンプト",
    )
    parser.add_argument("--output", type=Path, default=BASE_DIR / "data" / "dialogues.jsonl")
    parser.add_argument("--limit", type=int, help="先頭から実行する問題数。パイロットでは20を推奨")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--student-temperature", type=float, default=0.6)
    parser.add_argument("--student-top-p", type=float, default=0.95)
    parser.add_argument("--student-top-k", type=int, default=20)
    parser.add_argument("--student-min-p", type=float, default=0.0)
    parser.add_argument("--student-max-tokens", type=int, default=4096)
    parser.add_argument("--teacher-temperature", type=float, default=0.2)
    parser.add_argument("--phase2-temperature", type=float, default=0.6)
    parser.add_argument("--response-retries", type=int, default=3)
    parser.add_argument("--teacher-checkpoint", help="実際にロードした教師base checkpoint/HF ID")
    parser.add_argument(
        "--teacher-revision", default=os.getenv("TEACHER_MODEL_REVISION"),
        help="教師vLLM起動時に固定したbase model revision",
    )
    parser.add_argument(
        "--teacher-adapter",
        help="vLLMの--lora-modulesで実際にロードしたadapterの絶対パス（記録だけには使用不可）",
    )
    parser.add_argument(
        "--teacher-serving-mode", choices=("base", "lora", "merged"),
        help="教師vLLMの配信形態。省略時は--teacher-adapterがあればlora、それ以外はbase",
    )
    parser.add_argument("--student-checkpoint", help="実際にロードした生徒checkpoint/HF ID")
    parser.add_argument("--student-revision", default=DEFAULT_STUDENT_REVISION)
    parser.add_argument("--overwrite", action="store_true", help="既存出力を消して最初から実行")
    return parser.parse_args()


def read_text(name: str, shared: bool = False) -> str:
    prompt_dir = SHARED_DIR / "prompts" if shared else BASE_DIR / "prompts"
    return (prompt_dir / name).read_text(encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_excluded_ids(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(source_id) for source_id in data.get("excluded_source_ids", [])}


def normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        if normalized and normalized[-1]["role"] == message["role"]:
            normalized[-1]["content"] += "\n\n" + message["content"]
        else:
            normalized.append(dict(message))
    return normalized


def build_call_kwargs(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int = 512,
    response_format: dict[str, Any] | None = None,
    seed: int | None = None,
    extra_body: dict[str, Any] | None = None,
    *,
    api_provider: str = "vllm",
    reasoning_effort: str = "none",
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": normalize_messages(messages),
        "temperature": temperature,
    }
    if api_provider == "openai":
        kwargs["max_completion_tokens"] = max_tokens
        kwargs["reasoning_effort"] = reasoning_effort
        standard_sampling = dict(extra_body or {})
        unsupported = [
            key for key in ("top_k", "min_p")
            if key in standard_sampling
        ]
        if unsupported:
            raise ValueError(
                "OpenAI APIへvLLM固有sampling引数を送信できません: "
                + ", ".join(unsupported)
            )
        kwargs.update(standard_sampling)
    elif api_provider == "vllm":
        kwargs["max_tokens"] = max_tokens
        if seed is not None:
            kwargs["seed"] = seed
        if extra_body:
            kwargs["extra_body"] = extra_body
    else:
        raise ValueError(f"unsupported API provider: {api_provider}")
    if response_format:
        kwargs["response_format"] = response_format
    return kwargs


def call_model(client: OpenAI, model: str, messages: list[dict[str, str]], temperature: float,
               max_tokens: int = 512, response_format: dict[str, Any] | None = None,
               seed: int | None = None, extra_body: dict[str, Any] | None = None,
               *, api_provider: str = "vllm", reasoning_effort: str = "none") -> str:
    kwargs = build_call_kwargs(
        model, messages, temperature, max_tokens, response_format, seed, extra_body,
        api_provider=api_provider, reasoning_effort=reasoning_effort,
    )
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    if not content:
        raise ValueError(f"{model} returned empty content")
    return content.strip()


def validate_openai_student_model(client: OpenAI, expected_model: str) -> dict[str, str]:
    """OpenAIが返すsnapshot IDを固定値と照合し、比較条件のずれを防ぐ。"""
    retrieved = client.models.retrieve(expected_model)
    actual_model = str(getattr(retrieved, "id", ""))
    if actual_model != expected_model:
        raise RuntimeError(
            f"OpenAI生徒モデルが固定snapshotと一致しません: "
            f"expected={expected_model}, actual={actual_model or 'unknown'}"
        )
    return {"id": actual_model, "provider": "openai", "snapshot": actual_model}


def parse_json_response(raw: str) -> dict[str, Any]:
    def loads_lenient(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', value))

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


def validate_student_state(
    state: Any,
    previous_state: dict[str, Any],
    *,
    allow_emotion_change: bool = True,
    newly_acquired_knowledge: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """v4コーパス生成と同じ状態更新制約を検証する。"""
    if not isinstance(state, dict) or set(state) not in (
        STUDENT_MODEL_STATE_KEYS, STUDENT_STATE_KEYS,
    ):
        raise ValueError("state_after is missing or has unexpected fields")
    if newly_acquired_knowledge is None:
        reported = state.get("acquired_knowledge", previous_state["acquired_knowledge"])
        newly_acquired_knowledge = [
            item for item in reported if item not in previous_state["acquired_knowledge"]
        ]
    if (
        not isinstance(newly_acquired_knowledge, list)
        or any(not isinstance(item, str) or not item.strip() for item in newly_acquired_knowledge)
        or len(newly_acquired_knowledge) != len(set(newly_acquired_knowledge))
    ):
        raise ValueError("newly_acquired_knowledge is invalid")
    accumulated_knowledge = list(previous_state["acquired_knowledge"])
    accumulated_knowledge.extend(
        item for item in newly_acquired_knowledge if item not in accumulated_knowledge
    )
    state = {**state, "acquired_knowledge": accumulated_knowledge}
    state = {key: state[key] for key in STUDENT_STATE_KEYS}
    level, confidence = state["understanding_level"], state["confidence"]
    if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 4:
        raise ValueError("understanding_level is invalid")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("confidence is invalid")
    if abs(level - int(previous_state["understanding_level"])) > 1:
        raise ValueError("understanding_level changed by more than one step")
    if abs(float(confidence) - float(previous_state["confidence"])) > 0.25:
        raise ValueError("confidence changed by more than 0.25")
    misconception = state["active_misconception"]
    if not isinstance(misconception, str) or not misconception.strip():
        raise ValueError("active_misconception is empty")
    state["active_misconception"] = misconception.strip()
    if state["emotion"] not in STUDENT_EMOTIONS:
        raise ValueError("emotion is invalid")
    previous_emotion = str(previous_state["emotion"])
    next_emotion = str(state["emotion"])
    if not allow_emotion_change and next_emotion != previous_emotion:
        raise ValueError("initial emotion changed before teacher intervention")
    if not allow_emotion_change:
        if level != int(previous_state["understanding_level"]):
            raise ValueError("initial understanding changed before teacher intervention")
        if list(state["acquired_knowledge"]) != list(previous_state["acquired_knowledge"]):
            raise ValueError("initial acquired knowledge changed before teacher intervention")
        if list(state["remaining_unknowns"]) != list(previous_state["remaining_unknowns"]):
            raise ValueError("initial remaining unknowns changed before teacher intervention")
        if state["active_misconception"] != previous_state["active_misconception"]:
            raise ValueError("initial misconception changed before teacher intervention")
        if abs(float(confidence) - float(previous_state["confidence"])) > 0.1:
            raise ValueError("initial confidence changed by more than 0.1")
    if (
        allow_emotion_change
        and next_emotion != previous_emotion
        and next_emotion not in EMOTION_TRANSITIONS[previous_emotion]
    ):
        raise ValueError("emotion skipped the permitted cycle")
    for field in ("acquired_knowledge", "remaining_unknowns"):
        if not isinstance(state[field], list) or not all(
            isinstance(item, str) and item.strip() for item in state[field]
        ):
            raise ValueError(f"{field} is invalid")
    return state, []


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
    if not utterance or len(utterance) > 500:
        raise ValueError("utterance is empty or too long")
    if utterance.startswith(("{", "[")) or any(marker in utterance.lower() for marker in STUDENT_LEAK_MARKERS):
        raise ValueError("utterance contains JSON, state, or hidden-tag leakage")
    return utterance


def parse_student_turn(
    raw: str,
    previous_state: dict[str, Any],
    *,
    allow_emotion_change: bool = True,
    allowed_knowledge: list[str] | None = None,
    expected_response_mode: str = "follow_latest_teacher_step_only",
    latest_teacher_utterance: str = "",
    required_initial_disclosure: str = "",
) -> dict[str, Any]:
    parsed = parse_json_response(raw)
    required = {
        "state_after", "response_stage", "knowledge_used",
        "state_update_reason", "utterance", "newly_acquired_knowledge",
    }
    if not required.issubset(parsed):
        raise ValueError(f"student response is missing required fields; returned keys={sorted(parsed)}")
    normalized = {key: parsed[key] for key in required}
    normalized["state_after"], notes = validate_student_state(
        normalized["state_after"], previous_state,
        allow_emotion_change=allow_emotion_change,
        newly_acquired_knowledge=normalized["newly_acquired_knowledge"],
    )
    newly_acquired = set(normalized["newly_acquired_knowledge"])
    if any(item not in latest_teacher_utterance for item in newly_acquired):
        raise ValueError("new knowledge was not copied from the latest teacher utterance")
    normalized["_state_normalizations"] = notes
    normalized["utterance"] = validate_student_utterance(normalized["utterance"])
    if (
        required_initial_disclosure
        and not normalized["utterance"].startswith(required_initial_disclosure)
    ):
        raise ValueError("far_beyondの初回発話に2回の試行履歴が明示されていません")
    if normalized["response_stage"] not in STUDENT_RESPONSE_STAGES:
        raise ValueError("response_stage is invalid")
    stages = {
        "plausible_incorrect": {"attempt"},
        "partial_reasoning": {"observation", "attempt"},
        "correct_but_uncertain": {"attempt", "answer"},
        "scope_limited_help_seeking": {"observation", "help_seeking"},
        "natural_profile_consistent": set(STUDENT_RESPONSE_STAGES),
        "follow_latest_teacher_step_only": set(STUDENT_RESPONSE_STAGES),
    }
    if normalized["response_stage"] not in stages[expected_response_mode]:
        raise ValueError("response_stage does not match the response condition")
    knowledge_used = normalized["knowledge_used"]
    if not isinstance(knowledge_used, list) or any(
        not isinstance(item, str) or not item.strip() for item in knowledge_used
    ):
        raise ValueError("knowledge_used is invalid")
    if len(knowledge_used) != len(set(knowledge_used)):
        raise ValueError("knowledge_used contains duplicates")
    allowed = set(allowed_knowledge or []) | newly_acquired
    if allowed_knowledge is not None and any(item not in allowed for item in knowledge_used):
        raise ValueError("knowledge_used is outside the profile boundary")
    if not isinstance(normalized["state_update_reason"], str) or not normalized["state_update_reason"].strip():
        raise ValueError("state_update_reason is invalid")
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


def initial_state(
    profile: dict[str, Any], emotion: str,
    epistemic_assignment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    confidence = 0.35 if profile.get("confidence_bias") == "underconfident" else 0.55
    unknown_knowledge = profile["unknown_knowledge"]
    if isinstance(unknown_knowledge, str):
        unknown_knowledge = [unknown_knowledge]
    misconception = profile["target_misconception"]
    if epistemic_assignment is not None:
        misconception = epistemic_assignment["misconception_model"]["label"]
    return {
        "understanding_level": max(0, min(4, int(profile["ability_level"]) - 1)),
        "confidence": confidence,
        "active_misconception": misconception,
        "emotion": emotion,
        "acquired_knowledge": [],
        "remaining_unknowns": list(unknown_knowledge),
    }


def problem_level(row: dict[str, Any]) -> int:
    value = row.get("level")
    if isinstance(value, bool):
        raise ValueError("problem level must be an integer from 1 to 5")
    try:
        level = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("problem level must be an integer from 1 to 5") from exc
    if not 1 <= level <= 5:
        raise ValueError("problem level must be an integer from 1 to 5")
    return level


def initial_response_condition(epistemic_assignment: dict[str, Any]) -> str:
    if epistemic_assignment["scope_relation"] in {"one_step_beyond", "far_beyond"}:
        return "scope_limited_help_seeking"
    return str(epistemic_assignment["initial_response_mode"])


def required_initial_disclosure(epistemic_assignment: dict[str, Any]) -> str:
    if epistemic_assignment.get("scope_relation") != "far_beyond":
        return ""
    history = epistemic_assignment.get("prior_attempt_history")
    if not isinstance(history, dict):
        return ""
    disclosure = str(history.get("required_initial_disclosure", "")).strip()
    return "" if disclosure == "なし" else disclosure


def load_epistemic_assignments(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    assignments = {str(row.get("source_id")): row for row in rows}
    if len(assignments) != len(rows) or "" in assignments:
        raise ValueError("問題・プロフィール対応表に欠損または重複があります")
    for source_id, row in assignments.items():
        attempt_history = row.get("prior_attempt_history")
        if not isinstance(attempt_history, dict):
            raise ValueError(f"事前試行履歴がありません: {source_id}")
        attempts = attempt_history.get("attempts")
        if not isinstance(attempts, list):
            raise ValueError(f"事前試行一覧が不正です: {source_id}")
        if row.get("scope_relation") == "far_beyond":
            if (
                row.get("initial_emotion") != "frustrated"
                or int(attempt_history.get("attempt_count", 0)) != 2
                or len(attempts) != 2
                or [attempt.get("attempt_number") for attempt in attempts] != [1, 2]
                or any(
                    attempt.get("stopped_at") != attempt_history.get("repeated_stuck_point")
                    for attempt in attempts
                )
                or attempt_history.get("required_initial_disclosure") in {None, "", "なし"}
            ):
                raise ValueError(f"far_beyondに明示的な2回の事前試行履歴がありません: {source_id}")
        elif (
            int(attempt_history.get("attempt_count", 0)) != 0
            or attempts
            or attempt_history.get("required_initial_disclosure") != "なし"
        ):
            raise ValueError(f"far_beyond以外に事前試行履歴があります: {source_id}")
    return assignments


def load_problem_selection(
    path: Path, assignments: dict[str, dict[str, Any]], excluded_ids: set[str],
) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    records = value.get("records") if isinstance(value, dict) else None
    selected_count = int(value.get("selected_count", 0)) if isinstance(value, dict) else 0
    per_scope = int(value.get("per_scope_relation", 0)) if isinstance(value, dict) else 0
    if selected_count not in {60, 120} or per_scope not in {15, 30}:
        raise ValueError("テスト問題選択表は各scope 15件の60問または各30件の120問にしてください")
    if not isinstance(records, list) or len(records) != selected_count:
        raise ValueError("テスト問題選択表のselected_countとrecords件数が一致しません")
    if value.get("source_partition") != {"start": 800, "end_exclusive": 1000}:
        raise ValueError("テスト問題は後半200件から選択してください")
    expected_digest = hashlib.sha256(
        json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if value.get("selection_sha256") != expected_digest:
        raise ValueError("テスト問題選択表のselection_sha256が一致しません")
    source_ids: list[str] = []
    counts: dict[str, int] = {relation: 0 for relation in (
        "mastered", "frontier", "one_step_beyond", "far_beyond",
    )}
    for record in records:
        source_id = str(record.get("source_id", ""))
        assignment = assignments.get(source_id)
        if not source_id or source_id in source_ids or assignment is None:
            raise ValueError(f"テスト問題選択表に欠損・重複があります: {source_id}")
        if source_id in excluded_ids:
            raise ValueError(f"除外問題がテスト選択表に含まれています: {source_id}")
        if int(assignment.get("order_index", -1)) < 800:
            raise ValueError(f"先頭800件の問題がテスト選択表に含まれています: {source_id}")
        if bool(assignment["curriculum_annotation"].get("requires_human_review")):
            raise ValueError(f"人手確認対象の問題がテスト選択表に含まれています: {source_id}")
        knowledge_audit = assignment.get("knowledge_boundary_audit") or {}
        if not bool(knowledge_audit.get("relation_consistent")):
            raise ValueError(f"テスト問題とプロフィールの知識境界が不整合です: {source_id}")
        if (
            assignment.get("scope_relation") == "mastered"
            and knowledge_audit.get("not_in_prior_knowledge")
        ):
            raise ValueError(f"未習概念をmasteredへ割り当てています: {source_id}")
        for key in ("order_index", "scope_relation", "profile_id", "question_sha256"):
            if record.get(key) != assignment.get(key):
                raise ValueError(f"テスト選択表と対応表が一致しません: {source_id}/{key}")
        counts[str(assignment["scope_relation"])] += 1
        source_ids.append(source_id)
    if set(counts.values()) != {per_scope}:
        raise ValueError(f"テスト問題の範囲関係が不均衡です: {counts}")
    return source_ids


def build_phase2_input(
    problem: str,
    dialogue: list[dict[str, Any]],
    similar_question: str,
) -> dict[str, Any]:
    return {
        "original_problem": problem,
        "phase1_dialogue": [
            {"role": item["role"], "content": item["content"]}
            for item in dialogue
        ],
        "new_problem": similar_question,
    }


def teacher_user_input(
    problem: str, initial_emotion: str, utterance: str, turn: int,
) -> str:
    if turn > 0:
        return utterance
    return (
        f"問題: {problem}\n\n"
        f"初期感情ラベル: {initial_emotion}\n\n"
        f"生徒発話: {utterance}"
    )


def validate_phase2_transfer(
    value: dict[str, Any], profile: dict[str, Any], dialogue: list[dict[str, Any]],
) -> dict[str, Any]:
    if set(value) != {"answer", "knowledge_sources", "application_summary"}:
        raise ValueError("phase2 response keys are invalid")
    answer = str(value["answer"]).strip()
    if (
        not answer.startswith(r"\boxed{")
        or not answer.endswith("}")
        or answer.count(r"\boxed{") != 1
        or "\n" in answer
    ):
        raise ValueError("phase2 answer must be one boxed answer")
    sources = value["knowledge_sources"]
    if not isinstance(sources, list):
        raise ValueError("phase2 knowledge_sources must be a list")
    teacher_texts = [
        str(turn["content"]) for turn in dialogue if turn.get("role") == "teacher"
    ]
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"source_type", "source_text"}:
            raise ValueError("phase2 knowledge source is invalid")
        source_text = str(source["source_text"]).strip()
        if source["source_type"] == "prior_knowledge":
            if source_text not in profile["prior_knowledge"]:
                raise ValueError("phase2 prior knowledge source is outside the profile")
        elif source["source_type"] == "phase1_teacher":
            if not source_text or not any(source_text in text for text in teacher_texts):
                raise ValueError("phase2 teacher source is not an exact dialogue quote")
        else:
            raise ValueError("phase2 source_type is invalid")
    if answer != r"\boxed{わからない}" and not sources:
        raise ValueError("a phase2 answer requires at least one permitted knowledge source")
    summary = str(value["application_summary"]).strip()
    if not summary or len(summary) > 300:
        raise ValueError("phase2 application_summary is invalid")
    return {
        "answer": answer,
        "knowledge_sources": sources,
        "application_summary": summary,
    }


def generation_succeeded(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("run_id"))
        and not row.get("generation_error")
        and int(row.get("phase1_turns", 0)) > 0
        and bool(str(row.get("phase2_student_answer", "")).strip())
    )


def _model_card_dict(card: Any) -> dict[str, Any]:
    if isinstance(card, dict):
        return dict(card)
    if hasattr(card, "model_dump"):
        return dict(card.model_dump())
    return {
        key: getattr(card, key, None)
        for key in ("id", "root", "parent", "owned_by")
    }


def _same_adapter_path(served_root: str, expected_adapter: str) -> bool:
    served = Path(served_root).expanduser()
    expected = Path(expected_adapter).expanduser()
    try:
        if served.exists() and expected.exists():
            return served.samefile(expected)
    except OSError:
        pass
    return served.resolve(strict=False) == expected.resolve(strict=False)


def validate_teacher_serving_metadata(
    cards: list[Any], teacher_model: str, serving_mode: str,
    teacher_adapter: str | None, teacher_checkpoint: str | None = None,
) -> dict[str, Any]:
    models = [_model_card_dict(card) for card in cards]
    matches = [card for card in models if str(card.get("id")) == teacher_model]
    if len(matches) != 1:
        available = [str(card.get("id")) for card in models]
        raise RuntimeError(
            f"教師modelがvLLM /modelsに一意にありません: {teacher_model}; available={available}"
        )
    card = matches[0]
    parent = str(card.get("parent") or "").strip()
    root = str(card.get("root") or "").strip()
    if serving_mode == "lora":
        if not teacher_adapter:
            raise RuntimeError("LoRA評価には--teacher-adapterが必要です")
        if not parent:
            raise RuntimeError(
                "指定したteacher-modelはLoRA model cardではありません。"
                "vLLMを--enable-loraとbase_model_name付き--lora-modulesで起動してください。"
            )
        if teacher_checkpoint and parent != teacher_checkpoint:
            raise RuntimeError(
                f"LoRAのparentと--teacher-checkpointが一致しません: "
                f"served={parent!r}, expected={teacher_checkpoint!r}"
            )
        if not root or not _same_adapter_path(root, teacher_adapter):
            raise RuntimeError(
                f"vLLMが配信するLoRA rootと--teacher-adapterが一致しません: "
                f"served={root!r}, expected={teacher_adapter!r}"
            )
    elif teacher_adapter:
        raise RuntimeError(
            "--teacher-adapterは--teacher-serving-mode loraでのみ指定できます。"
            "merged評価ではマージ済みcheckpointを--teacher-checkpointへ指定してください。"
        )
    elif parent:
        raise RuntimeError(
            f"{serving_mode}評価なのにLoRA子モデルが配信されています: parent={parent}"
        )
    return {
        "id": str(card.get("id")),
        "root": root or None,
        "parent": parent or None,
        "serving_mode": serving_mode,
    }


def call_and_parse_student(client: OpenAI, model: str, messages: list[dict[str, str]], temperature: float,
                           previous_state: dict[str, Any], seed: int, retries: int,
                           *, allow_emotion_change: bool, top_p: float, top_k: int,
                           min_p: float, max_tokens: int,
                            api_provider: str = "vllm",
                            reasoning_effort: str = "none",
                            allowed_knowledge: list[str] | None = None,
                            expected_response_mode: str = "follow_latest_teacher_step_only",
                            latest_teacher_utterance: str = "",
                            required_initial_disclosure: str = "") -> tuple[dict[str, Any], int]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            retry_messages = list(messages)
            if attempt:
                disclosure_retry = (
                    " utteranceの先頭を次の文字列と完全一致させてください: "
                    f"{required_initial_disclosure}"
                    if required_initial_disclosure else ""
                )
                retry_messages.append({"role": "user", "content": (
                    "前回の出力形式が不正でした。problem、turn、state_before等をコピーせず、"
                    "state_after、response_stage、knowledge_used、state_update_reason、"
                    "utteranceの5キーだけを持つJSONを返してください。"
                    f"{disclosure_retry}"
                )})
            sampling = {"top_p": top_p}
            if api_provider == "vllm":
                sampling.update({"top_k": top_k, "min_p": min_p})
            raw = call_model(
                client, model, retry_messages, temperature, max_tokens,
                STUDENT_TURN_SCHEMA, seed + attempt,
                extra_body=sampling,
                api_provider=api_provider,
                reasoning_effort=reasoning_effort,
            )
            return parse_student_turn(
                raw, previous_state, allow_emotion_change=allow_emotion_change,
                allowed_knowledge=allowed_knowledge,
                expected_response_mode=expected_response_mode,
                latest_teacher_utterance=latest_teacher_utterance,
                required_initial_disclosure=required_initial_disclosure,
            ), attempt
        except Exception as exc:
            last_error = exc
            time.sleep(0.2 * (attempt + 1))
    raise ValueError(f"student structured response failed after {retries} attempts: {last_error}")


def call_and_parse_teacher(client: OpenAI, model: str, messages: list[dict[str, str]], temperature: float,
                           seed: int, retries: int) -> tuple[str, str | None, bool, str, int]:
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def question_content_hash(row: dict[str, Any]) -> str:
    problem = str(row.get("translated_question") or row.get("problem") or "").strip()
    solution = str(row.get("translated_solution") or row.get("solution") or "").strip()
    encoded = json.dumps(
        {"problem": problem, "reference_solution": solution},
        ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_fingerprint(args: argparse.Namespace, config: Config) -> tuple[str, dict[str, str]]:
    source_paths = {
        "questions": args.questions,
        "similar_questions": args.similar_questions,
        "excluded_question_ids": args.excluded_question_ids,
        "profiles": args.profiles,
        "problem_profile_assignments": args.problem_profile_assignments,
        "problem_selection": args.problem_selection,
        "initial_emotions": args.initial_emotions,
        "teacher_system_prompt": args.teacher_system_prompt,
        "student_system_prompt": BASE_DIR / "prompts" / "student_system.txt",
        "phase2_system_prompt": BASE_DIR / "prompts" / "phase2_in_context_student_system.txt",
    }
    hashes = {name: sha256(path) for name, path in source_paths.items()}
    payload = {
        "config": asdict(config),
        "source_sha256": hashes,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "limit": args.limit,
        "teacher_checkpoint": args.teacher_checkpoint or args.teacher_model,
        "teacher_adapter": args.teacher_adapter,
        "teacher_serving_mode": config.teacher_serving_mode,
        "teacher_revision": config.teacher_revision,
        "student_checkpoint": args.student_checkpoint or args.student_model,
        "student_api_provider": args.student_api_provider,
        "student_reasoning_effort": args.student_reasoning_effort,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), hashes


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--num-shardsと--shard-indexの組み合わせが不正です。")
    if args.response_retries < 1:
        raise SystemExit("--response-retriesは1以上にしてください。")
    if not 0 <= args.student_top_p <= 1 or args.student_top_k < 0 or not 0 <= args.student_min_p <= 1:
        raise SystemExit("生徒のtop-p/top-k/min-p設定が不正です。")
    if args.student_max_tokens < 1:
        raise SystemExit("--student-max-tokensは1以上にしてください。")
    if args.student_request_timeout <= 0:
        raise SystemExit("--student-request-timeoutは正の値にしてください。")
    if args.student_api_provider == "openai" and (
        args.student_top_k != 0 or args.student_min_p != 0
    ):
        raise SystemExit(
            "OpenAI生徒モデルでは--student-top-k 0と--student-min-p 0を指定してください。"
        )
    teacher_serving_mode = args.teacher_serving_mode or (
        "lora" if args.teacher_adapter else "base"
    )
    if teacher_serving_mode == "lora":
        if not args.teacher_adapter or not args.teacher_checkpoint:
            raise SystemExit(
                "LoRA評価には--teacher-adapterと--teacher-checkpointを両方指定してください。"
            )
        adapter_path = Path(args.teacher_adapter).expanduser()
        if not adapter_path.is_dir():
            raise SystemExit(f"教師LoRA adapterディレクトリがありません: {adapter_path}")
        args.teacher_adapter = str(adapter_path.resolve())
    teacher_checkpoint = args.teacher_checkpoint or args.teacher_model
    if (
        teacher_serving_mode in {"base", "lora"}
        and not Path(teacher_checkpoint).expanduser().exists()
        and not args.teacher_revision
    ):
        raise SystemExit(
            "Hugging Face上の教師baseを評価する場合は--teacher-revisionを固定してください。"
        )
    config = Config(
        args.teacher_base_url, args.teacher_model, teacher_serving_mode,
        args.teacher_revision,
        args.student_base_url, args.student_model,
        args.student_revision, args.max_turns, args.student_temperature,
        args.student_top_p, args.student_top_k, args.student_min_p, args.student_max_tokens,
        args.teacher_temperature,
        args.phase2_temperature, args.seed, TRANSFER_MODE,
    )
    if config.teacher_base_url == config.student_base_url and config.teacher_model == config.student_model:
        raise SystemExit("教師と生徒が同じURL・モデルです。比較の交絡を避けるため別モデルを指定してください。")

    teacher_client = OpenAI(api_key="EMPTY", base_url=config.teacher_base_url)
    if args.student_api_provider == "openai":
        student_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY")
        if not student_api_key:
            raise SystemExit("OpenAI生徒モデルにはOPENAI_API_KEY（またはGPT_API_KEY）が必要です。")
        student_client = OpenAI(
            api_key=student_api_key,
            base_url=config.student_base_url,
            timeout=args.student_request_timeout,
        )
        try:
            student_serving_evidence = validate_openai_student_model(
                student_client, config.student_model,
            )
        except Exception as exc:
            raise SystemExit(
                f"OpenAI生徒モデルのsnapshot検証に失敗しました: {type(exc).__name__}: {exc}"
            ) from exc
    else:
        student_client = OpenAI(
            api_key="EMPTY",
            base_url=config.student_base_url,
            timeout=args.student_request_timeout,
        )
        student_serving_evidence = {
            "id": config.student_model,
            "provider": "vllm",
            "revision": config.student_revision,
        }
    try:
        teacher_serving_evidence = validate_teacher_serving_metadata(
            list(teacher_client.models.list().data), config.teacher_model,
            config.teacher_serving_mode, args.teacher_adapter, args.teacher_checkpoint,
        )
    except Exception as exc:
        raise SystemExit(
            f"教師vLLMの配信モデル検証に失敗しました: {type(exc).__name__}: {exc}"
        ) from exc

    teacher_system = args.teacher_system_prompt.read_text(encoding="utf-8").strip()
    student_template = read_text("student_system.txt")
    phase2_template = read_text("phase2_in_context_student_system.txt")
    profiles = json.loads(args.profiles.read_text(encoding="utf-8"))
    emotion_config = json.loads(args.initial_emotions.read_text(encoding="utf-8"))
    emotion_rows = emotion_config["emotions"]
    emotions = [row["name"] for row in emotion_rows]
    emotion_by_name = {row["name"]: row for row in emotion_rows}
    expected_initial_emotions = {
        "neutral", "engaged", "curious", "confused", "frustrated", "anxious",
    }
    profile_ids = [str(profile.get("id")) for profile in profiles]
    if (
        len(profiles) != 8
        or len(set(profile_ids)) != 8
        or set(emotions) != expected_initial_emotions
        or len(emotions) != 6
    ):
        raise ValueError("v4テストは8プロフィールと6初期感情を前提とします。")

    excluded_ids = read_excluded_ids(args.excluded_question_ids)
    originals = [
        row for row in read_jsonl(args.questions)
        if str(row.get("id") or row.get("source_id")) not in excluded_ids
    ]
    similar_by_id = {
        str(row.get("source_id") or row.get("id")): row
        for row in read_jsonl(args.similar_questions)
        if str(row.get("source_id") or row.get("id")) not in excluded_ids
    }
    epistemic_by_id = load_epistemic_assignments(args.problem_profile_assignments)
    selected_source_ids = load_problem_selection(
        args.problem_selection, epistemic_by_id, excluded_ids,
    )
    original_by_id = {
        str(row.get("id") or row.get("source_id")): row for row in originals
    }
    missing_selected = [
        source_id for source_id in selected_source_ids
        if source_id not in original_by_id or source_id not in similar_by_id
    ]
    if missing_selected:
        raise ValueError(f"選択済みテスト問題または類似問題がありません: {missing_selected[:5]}")
    all_pairs = [
        (original_by_id[source_id], similar_by_id[source_id])
        for source_id in selected_source_ids
    ]
    random.Random(config.seed).shuffle(all_pairs)
    profiles_by_id = {str(profile["id"]): profile for profile in profiles}
    missing_assignments = [
        str(original.get("id") or original.get("source_id"))
        for original, _ in all_pairs
        if str(original.get("id") or original.get("source_id")) not in epistemic_by_id
    ]
    if missing_assignments:
        raise ValueError(f"問題・プロフィール対応表に未登録です: {missing_assignments[:5]}")
    changed_questions = [
        str(original.get("id") or original.get("source_id"))
        for original, _ in all_pairs
        if epistemic_by_id[str(original.get("id") or original.get("source_id"))].get(
            "question_sha256"
        ) != question_content_hash(original)
    ]
    if changed_questions:
        raise ValueError(f"対応表作成後に問題内容が変わっています: {changed_questions[:5]}")
    pairs = [
        (global_index, original, similar)
        for global_index, (original, similar) in enumerate(all_pairs)
        if global_index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        pairs = pairs[:args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        args.output.write_text("", encoding="utf-8")
    existing_rows = read_jsonl(args.output) if args.output.exists() else []
    existing_by_id = {str(row["run_id"]): row for row in existing_rows if row.get("run_id")}
    done = {run_id for run_id, row in existing_by_id.items() if generation_succeeded(row)}
    manifest_path = args.output.with_suffix(".manifest.json")
    fingerprint, source_hashes = run_fingerprint(args, config)
    if existing_rows and not manifest_path.exists() and not args.overwrite:
        raise RuntimeError("既存対話にmanifestがないため安全に再開できません。別の--outputを使用してください。")
    if manifest_path.exists() and not args.overwrite:
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("run_fingerprint") != fingerprint:
            raise RuntimeError(
                "既存出力とモデル・prompt・問題・seed・sampling設定が一致しません。"
                "別の--outputを使うか、意図的に再生成する場合だけ--overwriteを指定してください。"
            )
    manifest_path.write_text(json.dumps({
        "config": asdict(config),
        "questions": str(args.questions), "similar_questions": str(args.similar_questions),
        "problem_profile_assignments": str(args.problem_profile_assignments),
        "problem_selection": str(args.problem_selection),
        "excluded_question_ids": str(args.excluded_question_ids),
        "excluded_question_count": len(excluded_ids),
        "profiles": str(args.profiles), "initial_emotions": str(args.initial_emotions),
        "teacher_system_prompt": str(args.teacher_system_prompt),
        "source_sha256": source_hashes, "run_fingerprint": fingerprint,
        "phase": "generation", "planned_runs": len(pairs),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "planned_assignments": [
            {
                "global_pair_index": global_index,
                "source_id": str(original.get("id") or original.get("source_id")),
                "profile_id": epistemic_by_id[
                    str(original.get("id") or original.get("source_id"))
                ]["profile_id"],
                "initial_emotion": epistemic_by_id[
                    str(original.get("id") or original.get("source_id"))
                ]["initial_emotion"],
            }
            for global_index, original, _ in pairs
        ],
        "loaded_models": {
            "teacher_checkpoint": args.teacher_checkpoint or args.teacher_model,
            "teacher_adapter": args.teacher_adapter,
            "teacher_revision": args.teacher_revision,
            "student_checkpoint": args.student_checkpoint or args.student_model,
            "student_revision": args.student_revision,
            "student_api_provider": args.student_api_provider,
            "student_serving_evidence": student_serving_evidence,
            "teacher_serving_evidence": teacher_serving_evidence,
        },
        "response_retries": args.response_retries,
        "resume": {"existing_records": len(existing_by_id), "successful_records_skipped": len(done),
                   "failed_records_planned_for_retry": sum(not generation_succeeded(row) for row in existing_by_id.values()),
                   "overwrite": args.overwrite},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    for global_index, original, similar in tqdm(pairs, desc="test v4 generation"):
        source_id = str(original.get("id") or original.get("source_id"))
        run_id = f"{source_id}:{config.transfer_mode}:seed-{config.seed}"
        if run_id in done:
            continue
        previous_record = existing_by_id.get(run_id, {})
        previous_attempt = int(previous_record.get("generation_attempt", 1 if previous_record else 0))
        generation_attempt = previous_attempt + 1
        epistemic_assignment = epistemic_by_id[source_id]
        base_profile = profiles_by_id[str(epistemic_assignment["profile_id"])]
        profile = json.loads(json.dumps(base_profile, ensure_ascii=False))
        profile["problem_epistemic_state"] = {
            "curriculum_annotation": epistemic_assignment["curriculum_annotation"],
            "scope_relation": epistemic_assignment["scope_relation"],
            "misconception_model": epistemic_assignment["misconception_model"],
            "prior_attempt_history": epistemic_assignment["prior_attempt_history"],
            "initial_response_constraint": epistemic_assignment["initial_response_constraint"],
        }
        initial_emotion = str(epistemic_assignment["initial_emotion"])
        formatted_profile = profile_text(profile)
        state = initial_state(profile, initial_emotion, epistemic_assignment)
        initial_student_state = dict(state)
        problem = original["translated_question"]
        level = problem_level(original)
        first_response_condition = initial_response_condition(epistemic_assignment)
        dialogue: list[dict[str, Any]] = []
        teacher_history = [{"role": "system", "content": teacher_system}]
        last_teacher = ""
        is_completed = False
        generation_error: str | None = None
        validation_retries = {"student": 0, "teacher": 0, "phase2": 0}
        run_seed = config.seed + global_index * 100 + (generation_attempt - 1) * 1_000_000

        for turn in range(config.max_turns):
            student_input = {
                "problem": problem,
                "turn": turn,
                "state_before": state,
                "initial_emotion": initial_emotion,
                "initial_emotion_condition": emotion_by_name[initial_emotion],
                "initial_response_condition": (
                    first_response_condition
                    if turn == 0 else "follow_latest_teacher_step_only"
                ),
                "problem_level": level,
                "epistemic_state_specification": epistemic_assignment,
                "knowledge_boundary": {
                    "prior_knowledge": profile["prior_knowledge"],
                    "acquired_knowledge": state["acquired_knowledge"],
                    "unknown_knowledge": profile["unknown_knowledge"],
                    "max_independent_math_level": profile["max_independent_math_level"],
                },
                "latest_teacher_utterance": last_teacher,
                "recent_dialogue": [
                    {"role": item["role"], "content": item["content"]}
                    for item in dialogue[-6:]
                ],
                "instruction": (
                    "問題を解き始めてください"
                    if turn == 0
                    else "教師の最新発話へ生徒として応答してください"
                ),
            }
            try:
                previous_state = state
                student_turn, retry_count = call_and_parse_student(
                    student_client, config.student_model,
                    [{"role": "system", "content": student_template.replace("{STUDENT_PROFILE}", formatted_profile)},
                     {"role": "user", "content": json.dumps(student_input, ensure_ascii=False)}],
                    config.student_temperature, state, run_seed + turn * 20, args.response_retries,
                    allow_emotion_change=turn > 0,
                    top_p=config.student_top_p, top_k=config.student_top_k,
                    min_p=config.student_min_p, max_tokens=config.student_max_tokens,
                    api_provider=args.student_api_provider,
                    reasoning_effort=args.student_reasoning_effort,
                    allowed_knowledge=[
                        *profile["prior_knowledge"], *state["acquired_knowledge"],
                    ],
                    expected_response_mode=(
                        first_response_condition
                        if turn == 0 else "follow_latest_teacher_step_only"
                    ),
                    latest_teacher_utterance=last_teacher,
                    required_initial_disclosure=(
                        required_initial_disclosure(epistemic_assignment)
                        if turn == 0 else ""
                    ),
                )
                validation_retries["student"] += retry_count
                state_normalizations = student_turn.pop("_state_normalizations", [])
                state = student_turn["state_after"]
                utterance = student_turn["utterance"].strip()
                dialogue.append({
                    "role": "student",
                    "content": utterance,
                    "response_stage": student_turn["response_stage"],
                    "knowledge_used": student_turn["knowledge_used"],
                    "newly_acquired_knowledge": student_turn["newly_acquired_knowledge"],
                    "state_after": state,
                    "state_update_validated": True,
                    "state_changed": state != previous_state,
                    "state_normalizations": state_normalizations,
                    "state_update_reason": student_turn.get(
                        "state_update_reason",
                        "生徒モデルの応答で更新理由が省略されました。",
                    ),
                })
                teacher_user_content = teacher_user_input(
                    problem, initial_emotion, utterance, turn,
                )
                teacher_history.append({"role": "user", "content": teacher_user_content})

                last_teacher, teacher_analysis, is_completed, teacher_raw, retry_count = call_and_parse_teacher(
                    teacher_client, config.teacher_model, teacher_history,
                    config.teacher_temperature, run_seed + turn * 20 + 10, args.response_retries,
                )
                validation_retries["teacher"] += retry_count
                teacher_log = {
                    "role": "teacher", "content": last_teacher,
                    "is_completed": is_completed,
                }
                if teacher_analysis is not None:
                    teacher_log["analysis"] = teacher_analysis
                dialogue.append(teacher_log)
                teacher_history.append({"role": "assistant", "content": teacher_raw})
                if is_completed:
                    break
            except Exception as exc:
                generation_error = f"{type(exc).__name__}: {exc}"
                break

        phase2_answer = ""
        phase2_student_trace: dict[str, Any] | None = None
        if generation_error is None and dialogue:
            phase2_input = build_phase2_input(
                problem, dialogue, similar["similar_question"],
            )
            try:
                phase2_messages = [
                    {"role": "system", "content": phase2_template.replace(
                        "{STUDENT_PROFILE}", formatted_profile,
                    )},
                    {"role": "user", "content": json.dumps(
                        phase2_input, ensure_ascii=False,
                    )},
                ]
                phase2_error: Exception | None = None
                for attempt in range(args.response_retries):
                    retry_messages = list(phase2_messages)
                    if phase2_error is not None:
                        retry_messages.append({"role": "user", "content": (
                            f"前回出力は検証エラーでした: {phase2_error}。"
                            "知識源を完全一致で引用し、指定JSONだけを再生成してください。"
                        )})
                    try:
                        phase2_sampling = {"top_p": config.student_top_p}
                        if args.student_api_provider == "vllm":
                            phase2_sampling.update({
                                "top_k": config.student_top_k,
                                "min_p": config.student_min_p,
                            })
                        raw_phase2 = call_model(
                            student_client, config.student_model, retry_messages,
                            config.phase2_temperature, config.student_max_tokens,
                            response_format=PHASE2_TRANSFER_SCHEMA,
                            seed=run_seed + 90 + attempt,
                            extra_body=phase2_sampling,
                            api_provider=args.student_api_provider,
                            reasoning_effort=args.student_reasoning_effort,
                        )
                        phase2_student_trace = validate_phase2_transfer(
                            parse_json_response(raw_phase2), profile, dialogue,
                        )
                        validation_retries["phase2"] = attempt
                        break
                    except Exception as exc:
                        phase2_error = exc
                else:
                    raise ValueError(
                        f"phase2 structured response failed after retries: {phase2_error}"
                    ) from phase2_error
                phase2_answer = phase2_student_trace["answer"]
            except Exception as exc:
                generation_error = f"Phase2 {type(exc).__name__}: {exc}"

        record = {
            "run_id": run_id, "source_id": source_id, "seed": config.seed,
            "generation_attempt": generation_attempt,
            "global_pair_index": global_index,
            "transfer_mode": config.transfer_mode,
            "teacher_model": config.teacher_model, "student_model": config.student_model,
            "student_profile_used": profile, "initial_emotion": initial_emotion,
            "initial_student_state": initial_student_state,
            "generation_condition": {
                "problem_level": level,
                "initial_response_condition": first_response_condition,
                "knowledge_gate_active": first_response_condition == "scope_limited_help_seeking",
                "problem_profile_assignment": epistemic_assignment,
            },
            "final_student_state": state, "phase1_turns": sum(item["role"] == "student" for item in dialogue),
            "phase1_is_completed": is_completed, "phase2_student_answer": phase2_answer,
            "phase2_student_trace": phase2_student_trace,
            "similar_question": similar["similar_question"],
            "similar_solution": similar["similar_solution"],
            "dialogue_log": dialogue, "generation_error": generation_error,
            "validation_retries": validation_retries,
            "loaded_models": {
                "teacher_checkpoint": args.teacher_checkpoint or args.teacher_model,
                "teacher_adapter": args.teacher_adapter,
                "teacher_revision": args.teacher_revision,
                "student_checkpoint": args.student_checkpoint or args.student_model,
                "student_revision": args.student_revision,
                "student_api_provider": args.student_api_provider,
                "student_serving_evidence": student_serving_evidence,
                "teacher_serving_evidence": teacher_serving_evidence,
            },
        }
        existing_by_id[run_id] = record
        write_jsonl(args.output, [existing_by_id[key] for key in sorted(existing_by_id)])

    print(f"完了: {args.output}")
    print(f"設定: {manifest_path}")


if __name__ == "__main__":
    main()
