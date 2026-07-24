"""ABCIへコピーして実行できるv4コーパス生成パイプライン。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
PROMPT_DIR = BASE_DIR / "prompts"
DEFAULT_CONFIG = BASE_DIR / "config.json"

STUDENT_STATE_KEYS = {
    "understanding_level", "confidence", "active_misconception", "emotion",
    "acquired_knowledge", "remaining_unknowns",
}
STUDENT_EMOTIONS = [
    "engaged", "curious", "neutral", "confused", "frustrated", "anxious",
    "bored", "eureka", "relieved", "proud",
]
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
TEACHER_EMOTIONS = [
    "Engaged", "Curious", "Neutral", "Confusion", "Frustrated",
    "Bored", "Anxious", "Eureka", "Proud", "Relieved",
]
SCORE_FIELDS = [
    "mathematical_accuracy_score", "error_diagnosis_recovery_score",
    "cognitive_empathy_score", "emotional_support_score",
    "adaptive_scaffolding_score", "verification_completion_score",
]

STUDENT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "v4_student_turn", "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "state_after": {
                    "type": "object",
                    "properties": {
                        "understanding_level": {"type": "integer", "minimum": 0, "maximum": 4},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "active_misconception": {"type": "string"},
                        "emotion": {"type": "string", "enum": STUDENT_EMOTIONS},
                        "acquired_knowledge": {"type": "array", "items": {"type": "string"}},
                        "remaining_unknowns": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": sorted(STUDENT_STATE_KEYS),
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

MATHEMATICAL_ASSESSMENT_PROPERTIES = {
    "status": {"type": "string", "enum": ["correct", "partially_correct", "incorrect", "unclear"]},
    "verification": {"type": "string"},
    "correct_part": {"type": "string"},
    "error_part": {"type": "string"},
}
LEARNER_STATE_PROPERTIES = {
    "cognitive_state": {"type": "string"},
    "emotion": {"type": "string", "enum": TEACHER_EMOTIONS},
    "evidence": {"type": "string"},
}
SUPPORT_DECISION_PROPERTIES = {
    "next_support": {"type": "string"},
    "change_reason": {"type": "string"},
}
TEACHER_PROPERTIES = {
    "mathematical_assessment": {
        "type": "object", "properties": MATHEMATICAL_ASSESSMENT_PROPERTIES,
        "required": list(MATHEMATICAL_ASSESSMENT_PROPERTIES), "additionalProperties": False,
    },
    "learner_state": {
        "type": "object", "properties": LEARNER_STATE_PROPERTIES,
        "required": list(LEARNER_STATE_PROPERTIES), "additionalProperties": False,
    },
    "support_decision": {
        "type": "object", "properties": SUPPORT_DECISION_PROPERTIES,
        "required": list(SUPPORT_DECISION_PROPERTIES), "additionalProperties": False,
    },
    "is_completed": {"type": "boolean"},
    "teacher_utterance": {"type": "string"},
}
TEACHER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "v4_teacher_turn", "strict": True,
        "schema": {
            "type": "object", "properties": TEACHER_PROPERTIES,
            "required": list(TEACHER_PROPERTIES), "additionalProperties": False,
        },
    },
}

AUDIT_PROPERTIES = {
    **{name: {"type": "integer", "minimum": 0, "maximum": 10} for name in SCORE_FIELDS},
    "mathematically_correct": {"type": "boolean"},
    "student_answer_assessed_correctly": {"type": "boolean"},
    "cognitive_state_grounded": {"type": "boolean"},
    "emotion_grounded": {"type": "boolean"},
    "analysis_reflected_in_utterance": {"type": "boolean"},
    "student_profile_consistent": {"type": "boolean"},
    "student_role_consistent": {"type": "boolean"},
    "student_state_update_plausible": {"type": "boolean"},
    "initial_emotion_utterance_consistent": {"type": "boolean"},
    "false_affirmation": {"type": "boolean"},
    "direct_answer_without_need": {"type": "boolean"},
    "completion_decision_appropriate": {"type": "boolean"},
    "critical_failure": {"type": "boolean"},
    "context_repairable": {"type": "boolean"},
    "mathematical_verification": {"type": "string"},
    "issues": {"type": "array", "items": {"type": "string"}},
    "repair_instructions": {"type": "array", "items": {"type": "string"}},
    "reason": {"type": "string"},
}

REPAIR_TURN_PROPERTIES = {
    "teacher_index": {"type": "integer", "minimum": 0},
    **TEACHER_PROPERTIES,
}
REPAIR_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "v4_contextual_dialogue_repair", "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string"},
                "repaired_teacher_turns": {
                    "type": "array",
                    "items": {
                        "type": "object", "properties": REPAIR_TURN_PROPERTIES,
                        "required": list(REPAIR_TURN_PROPERTIES), "additionalProperties": False,
                    },
                },
                "context_consistency_check": {"type": "string"},
            },
            "required": ["candidate_id", "repaired_teacher_turns", "context_consistency_check"],
            "additionalProperties": False,
        },
    },
}
AUDIT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "v4_turn_audit", "strict": True,
        "schema": {
            "type": "object", "properties": AUDIT_PROPERTIES,
            "required": list(AUDIT_PROPERTIES), "additionalProperties": False,
        },
    },
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "generate", "submit-audit", "collect-audit", "submit-repair",
            "collect-repair", "submit-reaudit", "collect-reaudit", "finalize", "status",
        ],
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルがありません: {path}。config.example.jsonをconfig.jsonへコピーしてください。")
    config = read_json(path)
    required = {
        "target_dialogues", "max_candidates", "max_turns", "seed", "teacher_model",
        "teacher_reasoning_effort",
        "student_model", "student_model_revision", "vllm_version", "student_temperature",
        "student_top_p", "student_top_k", "student_min_p", "student_max_tokens",
        "judge_model", "judge_reasoning_effort", "repair_model",
        "repair_reasoning_effort", "questions", "output_dir",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"設定不足: {sorted(missing)}")
    if int(config["target_dialogues"]) > int(config["max_candidates"]):
        raise ValueError("target_dialoguesはmax_candidates以下にしてください")
    for name in ("target_dialogues", "max_candidates", "max_turns", "student_max_tokens"):
        if int(config[name]) <= 0:
            raise ValueError(f"{name}は正の整数にしてください")
    if not 0 <= float(config["student_temperature"]) <= 2:
        raise ValueError("student_temperatureは0以上2以下にしてください")
    if not 0 <= float(config["student_top_p"]) <= 1:
        raise ValueError("student_top_pは0以上1以下にしてください")
    if int(config["student_top_k"]) < 0:
        raise ValueError("student_top_kは0以上にしてください")
    if not 0 <= float(config["student_min_p"]) <= 1:
        raise ValueError("student_min_pは0以上1以下にしてください")
    config["questions"] = str(resolve_path(config["questions"], path))
    config["output_dir"] = str(resolve_path(config["output_dir"], path))
    if not Path(config["questions"]).is_file():
        raise FileNotFoundError(f"問題JSONLがありません: {config['questions']}")
    return config


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def paths(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["output_dir"])
    return {
        "root": root,
        "dialogues": root / "candidate_dialogues.jsonl",
        "generation_errors": root / "generation_errors.jsonl",
        "audit_input": root / "batches" / "audit_input.jsonl",
        "audits": root / "turn_audits.jsonl",
        "repair_input": root / "batches" / "repair_input.jsonl",
        "repairs": root / "dialogue_repairs.jsonl",
        "reaudit_input": root / "batches" / "reaudit_input.jsonl",
        "reaudits": root / "turn_reaudits.jsonl",
        "corpus": root / "v4_corpus.jsonl",
        "sft": root / "v4_sft.jsonl",
        "manifest": root / "manifest.json",
        "report": root / "corpus_report.md",
    }


PROMPT_FILES = [
    "teacher_system.txt", "student_system.txt", "student_profiles.json",
    "initial_emotions.json", "turn_quality_judge_system.txt", "sft_teacher_system.txt",
    "turn_repair_system.txt",
]
MUTABLE_LIMIT_KEYS = {"target_dialogues", "max_candidates"}


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def prompt_hashes() -> dict[str, str]:
    return {name: sha256(PROMPT_DIR / name) for name in PROMPT_FILES}


def immutable_run_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in sorted(config) if key not in MUTABLE_LIMIT_KEYS}


def source_hashes(config: dict[str, Any]) -> dict[str, str]:
    question_path = Path(config["questions"])
    return {
        "questions": sha256(question_path) if question_path.exists() else "missing",
    }


def runtime_environment(config: dict[str, Any]) -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "openai": package_version("openai"),
        "tqdm": package_version("tqdm"),
        "installed_vllm": package_version("vllm"),
        "configured_vllm": str(config["vllm_version"]),
        "student_model_revision": str(config["student_model_revision"]),
    }


def reproducibility_fingerprint(
    config: dict[str, Any], hashes: dict[str, str], sources: dict[str, str],
    environment: dict[str, str],
) -> str:
    payload = {
        "run_config": immutable_run_config(config),
        "prompt_sha256": hashes,
        "source_sha256": sources,
        "environment": environment,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def default_manifest(config: dict[str, Any]) -> dict[str, Any]:
    hashes = prompt_hashes()
    sources = source_hashes(config)
    environment = runtime_environment(config)
    return {
        "version": "v4",
        "selection_policy": "keep_or_contextual_repair_with_full_dialogue_reaudit",
        "run_config": immutable_run_config(config),
        "current_limits": {key: int(config[key]) for key in sorted(MUTABLE_LIMIT_KEYS)},
        "limit_history": [{key: int(config[key]) for key in sorted(MUTABLE_LIMIT_KEYS)}],
        "prompt_sha256": hashes,
        "source_sha256": sources,
        "run_fingerprint": reproducibility_fingerprint(config, hashes, sources, environment),
        "environment": environment,
        "batch_jobs": {},
    }


def load_manifest(config: dict[str, Any], file_paths: dict[str, Path]) -> dict[str, Any]:
    current = default_manifest(config)
    if not file_paths["manifest"].exists():
        return current
    manifest = read_json(file_paths["manifest"])
    if manifest.get("run_fingerprint") != current["run_fingerprint"]:
        raise RuntimeError(
            "既存runと現在のモデル・seed・prompt・入力問題・実行環境・設定が一致しません。"
            "別のoutput_dirを使うか、generate --overwriteで新規runを開始してください。"
        )
    limits = current["current_limits"]
    previous_limits = manifest.get("current_limits", {})
    decreased = [
        key for key, value in limits.items()
        if key in previous_limits and int(value) < int(previous_limits[key])
    ]
    if decreased:
        raise RuntimeError(
            f"既存runでは上限を減らせません: {decreased}。別のoutput_dirを使用してください。"
        )
    manifest["current_limits"] = limits
    history = manifest.setdefault("limit_history", [])
    if not history or history[-1] != limits:
        history.append(limits)
    return manifest


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_json_response(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(raw[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("response is not an object")
    return value


def profile_text(profile: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in profile.items():
        if isinstance(value, list):
            lines.append(f"- {key}:")
            lines.extend(f"  - {item}" for item in value)
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def initial_state(profile: dict[str, Any], emotion: str) -> dict[str, Any]:
    confidence = 0.35 if profile.get("confidence_bias") == "underconfident" else 0.55
    return {
        "understanding_level": max(0, min(4, int(profile["ability_level"]) - 1)),
        "confidence": confidence,
        "active_misconception": profile["target_misconception"],
        "emotion": emotion,
        "acquired_knowledge": [],
        "remaining_unknowns": list(profile["unknown_knowledge"]),
    }


def validate_student_turn(
    value: dict[str, Any], previous: dict[str, Any], *, allow_emotion_change: bool = True,
) -> dict[str, Any]:
    if set(value) != {"state_after", "state_update_reason", "utterance"}:
        raise ValueError("student response keys are invalid")
    state = value["state_after"]
    if not isinstance(state, dict) or set(state) != STUDENT_STATE_KEYS:
        raise ValueError("student state is incomplete")
    if abs(int(state["understanding_level"]) - int(previous["understanding_level"])) > 1:
        raise ValueError("understanding changed by more than one level")
    if state["emotion"] not in STUDENT_EMOTIONS:
        raise ValueError("student emotion is invalid")
    confidence = float(state["confidence"])
    previous_confidence = float(previous["confidence"])
    if not 0 <= confidence <= 1:
        raise ValueError("student confidence is outside 0..1")
    if abs(confidence - previous_confidence) > 0.25:
        raise ValueError("student confidence changed by more than 0.25")
    if not isinstance(state["acquired_knowledge"], list) or not isinstance(state["remaining_unknowns"], list):
        raise ValueError("student knowledge fields must be lists")
    if not set(previous["acquired_knowledge"]).issubset(state["acquired_knowledge"]):
        raise ValueError("previously acquired knowledge was removed")
    if not str(state["active_misconception"]).strip():
        raise ValueError("active misconception is empty")
    previous_emotion = str(previous["emotion"])
    next_emotion = str(state["emotion"])
    if not allow_emotion_change and next_emotion != previous_emotion:
        raise ValueError("initial emotion changed before teacher intervention")
    if (
        allow_emotion_change and next_emotion != previous_emotion
        and next_emotion not in EMOTION_TRANSITIONS[previous_emotion]
    ):
        raise ValueError("student emotion skipped the permitted cycle")
    utterance = str(value["utterance"]).strip()
    if not utterance or len(utterance) > 500 or utterance.startswith(("{", "[")):
        raise ValueError("student utterance is invalid")
    if any(marker in utterance.lower() for marker in ["<analysis>", "state_after", "state_update_reason"]):
        raise ValueError("student utterance leaks internal state")
    reason = str(value["state_update_reason"]).strip()
    if not reason or len(reason) > 1000:
        raise ValueError("student state update reason is invalid")
    value["utterance"] = utterance
    value["state_update_reason"] = reason
    value["state_after"] = {key: state[key] for key in STUDENT_STATE_KEYS}
    return value


def validate_teacher_turn(value: dict[str, Any]) -> dict[str, Any]:
    if set(value) != set(TEACHER_PROPERTIES):
        raise ValueError("teacher response keys are invalid")
    assessment = value["mathematical_assessment"]
    learner = value["learner_state"]
    support = value["support_decision"]
    if not isinstance(assessment, dict) or set(assessment) != set(MATHEMATICAL_ASSESSMENT_PROPERTIES):
        raise ValueError("mathematical assessment is invalid")
    if assessment["status"] not in {"correct", "partially_correct", "incorrect", "unclear"}:
        raise ValueError("mathematical assessment status is invalid")
    if any(not str(assessment[name]).strip() for name in ("verification", "correct_part", "error_part")):
        raise ValueError("mathematical assessment contains an empty field")
    if not isinstance(learner, dict) or set(learner) != set(LEARNER_STATE_PROPERTIES):
        raise ValueError("learner state is invalid")
    if learner["emotion"] not in TEACHER_EMOTIONS:
        raise ValueError("teacher emotion label is invalid")
    if any(not str(learner[name]).strip() for name in ("cognitive_state", "evidence")):
        raise ValueError("learner state contains an empty field")
    if not isinstance(support, dict) or set(support) != set(SUPPORT_DECISION_PROPERTIES):
        raise ValueError("support decision is invalid")
    if any(not str(support[name]).strip() for name in SUPPORT_DECISION_PROPERTIES):
        raise ValueError("support decision contains an empty field")
    if not isinstance(value["is_completed"], bool):
        raise ValueError("is_completed is not boolean")
    if value["is_completed"] and assessment["status"] != "correct":
        raise ValueError("instruction cannot be completed when the latest answer is not correct")
    utterance = str(value["teacher_utterance"]).strip()
    if not utterance or len(utterance) > 2000:
        raise ValueError("teacher utterance is invalid")
    if "[指導完了]" in utterance or "<analysis>" in utterance or "<final>" in utterance:
        raise ValueError("teacher utterance contains a reserved SFT marker")
    value["teacher_utterance"] = utterance
    return value


def chat_call(
    client: OpenAI, model: str, messages: list[dict[str, str]], schema: dict[str, Any],
    *, reasoning_effort: str | None = None, temperature: float | None = None,
    seed: int | None = None, retries: int = 3, max_completion_tokens: int | None = None,
    extra_body: dict[str, Any] | None = None, use_max_tokens: bool = False,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            kwargs: dict[str, Any] = {
                "model": model, "messages": messages, "response_format": schema,
            }
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
            if temperature is not None:
                kwargs["temperature"] = temperature
            if seed is not None:
                kwargs["seed"] = seed + attempt
            if max_completion_tokens is not None:
                token_key = "max_tokens" if use_max_tokens else "max_completion_tokens"
                kwargs[token_key] = max_completion_tokens
            if extra_body:
                kwargs["extra_body"] = extra_body
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if not content:
                raise ValueError("empty response")
            return parse_json_response(content)
        except Exception as exc:
            last_error = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"API call failed: {last_error}") from last_error


def question_fields(row: dict[str, Any]) -> tuple[str, str, str, dict[str, Any]]:
    source_id = str(row.get("id") or row.get("source_id") or "unknown")
    problem = str(row.get("translated_question") or row.get("problem") or row.get("question") or "").strip()
    solution = str(row.get("translated_solution") or row.get("solution") or "").strip()
    if not problem:
        raise ValueError(f"question is empty: {source_id}")
    text_fields = {
        "translated_question", "problem", "question",
        "translated_solution", "solution",
    }
    metadata = {key: value for key, value in row.items() if key not in text_fields}
    return source_id, problem, solution, metadata


def stratified_assignments(config: dict[str, Any], profiles: list[dict[str, Any]], emotions: list[str]) -> list[tuple[dict[str, Any], str]]:
    size = int(config["max_candidates"])
    combinations = [(profile, emotion) for profile in profiles for emotion in emotions]
    assignments: list[tuple[dict[str, Any], str]] = []
    for block_index in range((size + len(combinations) - 1) // len(combinations)):
        block = list(combinations)
        random.Random(int(config["seed"]) + block_index).shuffle(block)
        assignments.extend(block)
    return assignments[:size]


def generate(config: dict[str, Any], file_paths: dict[str, Path], overwrite: bool) -> None:
    load_env_file(BASE_DIR / ".env")
    if overwrite:
        for key in (
            "dialogues", "generation_errors", "audit_input", "audits",
            "repair_input", "repairs", "reaudit_input", "reaudits",
            "corpus", "sft", "manifest", "report",
        ):
            file_paths[key].unlink(missing_ok=True)
    existing = {row["candidate_id"] for row in read_jsonl(file_paths["dialogues"])}
    profiles = read_json(PROMPT_DIR / "student_profiles.json")
    emotion_config = read_json(PROMPT_DIR / "initial_emotions.json")
    emotions = [row["name"] for row in emotion_config["emotions"]]
    questions = read_jsonl(Path(config["questions"]))
    rng = random.Random(int(config["seed"]))
    rng.shuffle(questions)
    if len(questions) < int(config["max_candidates"]):
        raise ValueError("問題数がmax_candidatesより少ないです")
    assignments = stratified_assignments(config, profiles, emotions)

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEYを設定してください")
    teacher_client = OpenAI(api_key=api_key)
    student_client = OpenAI(
        api_key=os.getenv("STUDENT_API_KEY", "EMPTY"),
        base_url=os.getenv("STUDENT_BASE_URL", "http://localhost:8001/v1"),
    )
    teacher_system = (PROMPT_DIR / "teacher_system.txt").read_text(encoding="utf-8")
    student_template = (PROMPT_DIR / "student_system.txt").read_text(encoding="utf-8")
    manifest = load_manifest(config, file_paths)

    for index in tqdm(range(int(config["max_candidates"])), desc="generate"):
        candidate_id = f"v4-{index:04d}"
        if candidate_id in existing:
            continue
        profile, emotion = assignments[index]
        source_id, problem, solution, source_metadata = question_fields(questions[index])
        state = initial_state(profile, emotion)
        initial = dict(state)
        student_system = student_template.replace("{STUDENT_PROFILE}", profile_text(profile))
        teacher_history: list[dict[str, str]] = [{"role": "system", "content": teacher_system}]
        recent_dialogue: list[dict[str, str]] = []
        conversation: list[dict[str, Any]] = []
        generation_error = None
        try:
            for turn_index in range(int(config["max_turns"])):
                student_payload = {
                    "problem": problem,
                    "turn": turn_index,
                    "state_before": state,
                    "initial_emotion": emotion,
                    "latest_teacher_utterance": recent_dialogue[-1]["content"] if recent_dialogue else "",
                    "recent_dialogue": recent_dialogue[-6:],
                    "instruction": "問題を解き始めてください" if turn_index == 0 else "教師の最新発話へ生徒として応答してください",
                }
                student_value = chat_call(
                    student_client, config["student_model"],
                    [{"role": "system", "content": student_system}, {"role": "user", "content": json.dumps(student_payload, ensure_ascii=False)}],
                    STUDENT_SCHEMA, temperature=float(config["student_temperature"]),
                    seed=int(config["seed"]) + index * 100 + turn_index,
                    max_completion_tokens=int(config["student_max_tokens"]),
                    use_max_tokens=True,
                    extra_body={
                        "top_p": float(config.get("student_top_p", 0.95)),
                        "top_k": int(config.get("student_top_k", 20)),
                        "min_p": float(config.get("student_min_p", 0)),
                    },
                )
                student_value = validate_student_turn(
                    student_value, state, allow_emotion_change=turn_index > 0,
                )
                state = student_value["state_after"]
                student_turn = {
                    "turn": turn_index, "role": "student", "content": student_value["utterance"],
                    "state_after": state, "state_update_reason": student_value["state_update_reason"],
                }
                conversation.append(student_turn)
                recent_dialogue.append({"role": "student", "content": student_value["utterance"]})
                teacher_user_content = student_value["utterance"]
                if turn_index == 0:
                    teacher_user_content = (
                        f"問題: {problem}\n\n内部検算用の参照解答: {solution}\n"
                        "参照解答は検算にだけ使い、生徒へそのまま提示しないでください。\n\n"
                        f"生徒発話: {student_value['utterance']}"
                    )
                teacher_history.append({"role": "user", "content": teacher_user_content})

                teacher_value = chat_call(
                    teacher_client, config["teacher_model"], teacher_history, TEACHER_SCHEMA,
                    reasoning_effort=config.get("teacher_reasoning_effort"),
                    max_completion_tokens=3000,
                )
                teacher_value = validate_teacher_turn(teacher_value)
                teacher_turn = {"turn": turn_index, "role": "teacher", **teacher_value}
                conversation.append(teacher_turn)
                recent_dialogue.append({"role": "teacher", "content": teacher_value["teacher_utterance"]})
                teacher_history.append({
                    "role": "assistant",
                    "content": json.dumps(teacher_value, ensure_ascii=False),
                })
                if teacher_value["is_completed"]:
                    break
        except Exception as exc:
            generation_error = f"{type(exc).__name__}: {exc}"

        row = {
            "candidate_id": candidate_id, "source_id": source_id,
            "source_metadata": source_metadata, "problem": problem,
            "reference_solution": solution, "student_profile": profile,
            "initial_emotion": emotion, "initial_student_state": initial,
            "final_student_state": state, "conversation": conversation,
            "is_completed": bool(conversation and conversation[-1].get("is_completed")),
            "generation_error": generation_error,
            "models": {
                "teacher": config["teacher_model"],
                "student": config["student_model"],
                "student_revision": config["student_model_revision"],
            },
        }
        if generation_error:
            append_jsonl(file_paths["generation_errors"], row)
        else:
            append_jsonl(file_paths["dialogues"], row)
            existing.add(candidate_id)
        manifest["generated_candidates"] = len(existing)
        save_manifest(file_paths["manifest"], manifest)


def teacher_turn_records(dialogues: list[dict[str, Any]]) -> Iterable[tuple[str, dict[str, Any], int, int, dict[str, Any]]]:
    for dialogue in dialogues:
        teacher_index = 0
        for conversation_index, turn in enumerate(dialogue["conversation"]):
            if turn.get("role") != "teacher":
                continue
            key = f"{dialogue['candidate_id']}:teacher:{teacher_index}"
            yield key, dialogue, teacher_index, conversation_index, turn
            teacher_index += 1


def audit_payload(dialogue: dict[str, Any], conversation_index: int, teacher_turn: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": dialogue["candidate_id"], "problem": dialogue["problem"],
        "reference_solution": dialogue["reference_solution"],
        "student_profile": dialogue["student_profile"], "initial_emotion": dialogue["initial_emotion"],
        "dialogue_before": dialogue["conversation"][:conversation_index],
        "teacher_turn": teacher_turn,
        "next_student_turn": dialogue["conversation"][conversation_index + 1] if conversation_index + 1 < len(dialogue["conversation"]) else None,
        "dialogue_after": dialogue["conversation"][conversation_index + 1:],
    }


def batch_body(model: str, reasoning_effort: str, system: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "response_format": schema,
        "reasoning_effort": reasoning_effort,
    }


def classify_audit(audit: dict[str, Any]) -> str:
    minimum = min(int(audit[name]) for name in SCORE_FIELDS)
    required_true = (
        "mathematically_correct",
        "student_answer_assessed_correctly",
        "cognitive_state_grounded",
        "emotion_grounded",
        "analysis_reflected_in_utterance",
        "student_profile_consistent",
        "student_role_consistent",
        "student_state_update_plausible",
        "initial_emotion_utterance_consistent",
        "completion_decision_appropriate",
    )
    required_false = (
        "false_affirmation",
        "direct_answer_without_need",
        "critical_failure",
    )
    keep = (
        minimum >= 8
        and all(bool(audit[name]) for name in required_true)
        and all(not bool(audit[name]) for name in required_false)
        and not audit["issues"]
        and not audit["repair_instructions"]
    )
    if keep:
        return "Keep"
    immutable_context_is_valid = all(bool(audit[name]) for name in (
        "student_profile_consistent",
        "student_role_consistent",
        "student_state_update_plausible",
        "initial_emotion_utterance_consistent",
    ))
    if bool(audit["context_repairable"]) and bool(audit["repair_instructions"]) and immutable_context_is_valid:
        return "Repair"
    return "Reject"


def invalidate_batch_stage(
    manifest: dict[str, Any], file_paths: dict[str, Path], stage: str,
) -> None:
    downstream = {
        "audit": ("audit", "repair", "reaudit"),
        "repair": ("repair", "reaudit"),
        "reaudit": ("reaudit",),
    }[stage]
    output_keys = {
        "audit": ("audits",),
        "repair": ("repairs",),
        "reaudit": ("reaudits",),
    }
    jobs = manifest.setdefault("batch_jobs", {})
    for name in downstream:
        jobs.pop(name, None)
        for key in output_keys[name]:
            file_paths[key].unlink(missing_ok=True)
    for key in ("corpus", "sft", "report"):
        file_paths[key].unlink(missing_ok=True)
    manifest.pop("final", None)


def submit_batch(
    config: dict[str, Any], file_paths: dict[str, Path], stage: str,
    input_path: Path, requests: list[dict[str, Any]], overwrite: bool,
) -> None:
    load_env_file(BASE_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEYを設定してください")
    manifest = load_manifest(config, file_paths)
    existing = manifest.get("batch_jobs", {}).get(stage)
    if existing and not overwrite:
        print(f"{stage}は処理済みです: {existing.get('batch_id', existing.get('status', 'unknown'))}")
        return
    if overwrite:
        invalidate_batch_stage(manifest, file_paths, stage)
    write_jsonl(input_path, requests)
    if not requests:
        manifest.setdefault("batch_jobs", {})[stage] = {
            "status": "skipped", "request_count": 0, "collected": False,
        }
        save_manifest(file_paths["manifest"], manifest)
        print(f"{stage}: 対象リクエストがありません")
        return
    client = OpenAI(api_key=api_key)
    with input_path.open("rb") as stream:
        uploaded = client.files.create(file=stream, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id, endpoint="/v1/chat/completions", completion_window="24h",
        metadata={"pipeline": "v4_corpus", "stage": stage},
    )
    manifest.setdefault("batch_jobs", {})[stage] = {
        "batch_id": batch.id, "input_file_id": uploaded.id, "status": batch.status,
        "request_count": len(requests),
    }
    save_manifest(file_paths["manifest"], manifest)
    print(f"submitted {stage}: {batch.id} ({len(requests)} requests)")


def submit_audit(config: dict[str, Any], file_paths: dict[str, Path], overwrite: bool) -> None:
    system = (PROMPT_DIR / "turn_quality_judge_system.txt").read_text(encoding="utf-8")
    requests = []
    for key, dialogue, _, conversation_index, turn in teacher_turn_records(read_jsonl(file_paths["dialogues"])):
        requests.append({
            "custom_id": key, "method": "POST", "url": "/v1/chat/completions",
            "body": batch_body(config["judge_model"], config["judge_reasoning_effort"], system, audit_payload(dialogue, conversation_index, turn), AUDIT_SCHEMA),
        })
    submit_batch(config, file_paths, "audit", file_paths["audit_input"], requests, overwrite)


def collect_batch(
    config: dict[str, Any], file_paths: dict[str, Path], stage: str,
    output_path: Path, transform,
) -> None:
    load_env_file(BASE_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY")
    manifest = load_manifest(config, file_paths)
    job = manifest.get("batch_jobs", {}).get(stage)
    if not job:
        raise RuntimeError(f"{stage} batchが未送信です")
    if job.get("collected") and output_path.exists():
        print(f"{stage}: 収集済みです ({output_path})")
        return
    if job.get("status") == "skipped":
        write_jsonl(output_path, [])
        job["collected"] = True
        save_manifest(file_paths["manifest"], manifest)
        print(f"{stage}: 対象0件のため空の結果を保存しました")
        return
    client = OpenAI(api_key=api_key)
    batch = client.batches.retrieve(job["batch_id"])
    job["status"] = batch.status
    if batch.status != "completed":
        save_manifest(file_paths["manifest"], manifest)
        print(f"{stage}: {batch.status}。完了後に同じcollectコマンドを再実行してください。")
        return
    if not batch.output_file_id:
        raise RuntimeError(f"{stage}: output_file_idがありません")
    content = client.files.content(batch.output_file_id).text
    rows = []
    for line in content.splitlines():
        if not line.strip():
            continue
        key: str | None = None
        try:
            item = json.loads(line)
            key = str(item.get("custom_id") or "unknown")
            response = item.get("response") or {}
            body = response.get("body") or {}
            if int(response.get("status_code", 0)) != 200:
                raise RuntimeError(str(item.get("error") or body))
            raw = body["choices"][0]["message"]["content"]
            value = parse_json_response(raw)
            rows.append(transform(key, value))
        except Exception as exc:
            rows.append({
                "turn_key": key,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })
    write_jsonl(output_path, rows)
    job["output_file_id"] = batch.output_file_id
    job["collected"] = True
    save_manifest(file_paths["manifest"], manifest)
    print(f"collected {stage}: {len(rows)} records")


def collect_audit(config: dict[str, Any], file_paths: dict[str, Path]) -> None:
    def transform(key: str, value: dict[str, Any]) -> dict[str, Any]:
        return {"turn_key": key, "status": "completed", "classification": classify_audit(value), "total_score": sum(value[n] for n in SCORE_FIELDS), "audit": value}
    collect_batch(config, file_paths, "audit", file_paths["audits"], transform)


def audits_for_dialogue(
    dialogue: dict[str, Any], audits: dict[str, dict[str, Any]],
) -> list[dict[str, Any] | None]:
    return [
        audits.get(key)
        for key, _, _, _, _ in teacher_turn_records([dialogue])
    ]


def submit_repair(config: dict[str, Any], file_paths: dict[str, Path], overwrite: bool) -> None:
    audits = {
        row["turn_key"]: row for row in read_jsonl(file_paths["audits"])
        if row.get("status") == "completed"
    }
    system = (PROMPT_DIR / "turn_repair_system.txt").read_text(encoding="utf-8")
    requests = []
    for dialogue in read_jsonl(file_paths["dialogues"]):
        dialogue_audits = audits_for_dialogue(dialogue, audits)
        if not dialogue_audits or any(row is None for row in dialogue_audits):
            continue
        classifications = [str(row["classification"]) for row in dialogue_audits if row]
        if "Reject" in classifications or "Repair" not in classifications:
            continue
        targets = [
            {
                "teacher_index": index,
                "turn_key": row["turn_key"],
                "audit": row,
            }
            for index, row in enumerate(dialogue_audits)
            if row and row["classification"] == "Repair"
        ]
        payload = {
            "candidate_id": dialogue["candidate_id"],
            "problem": dialogue["problem"],
            "reference_solution": dialogue["reference_solution"],
            "student_profile": dialogue["student_profile"],
            "initial_emotion": dialogue["initial_emotion"],
            "immutable_conversation": dialogue["conversation"],
            "repair_targets": targets,
            "instruction": (
                "repair_targetsの教師ターンだけを、前後の生徒発話と対話全体に整合するよう"
                "まとめて修正してください。生徒発話と非対象ターンは変更しないでください。"
            ),
        }
        requests.append({
            "custom_id": dialogue["candidate_id"], "method": "POST",
            "url": "/v1/chat/completions",
            "body": batch_body(
                config["repair_model"], config["repair_reasoning_effort"],
                system, payload, REPAIR_SCHEMA,
            ),
        })
    submit_batch(config, file_paths, "repair", file_paths["repair_input"], requests, overwrite)


def collect_repair(config: dict[str, Any], file_paths: dict[str, Path]) -> None:
    audits = {
        row["turn_key"]: row for row in read_jsonl(file_paths["audits"])
        if row.get("status") == "completed"
    }
    dialogues = {row["candidate_id"]: row for row in read_jsonl(file_paths["dialogues"])}

    def transform(candidate_id: str, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("candidate_id") != candidate_id or candidate_id not in dialogues:
            raise ValueError("repair candidate_id does not match the request")
        expected = {
            index for index, row in enumerate(audits_for_dialogue(dialogues[candidate_id], audits))
            if row and row["classification"] == "Repair"
        }
        repaired_turns = value.get("repaired_teacher_turns")
        if not isinstance(repaired_turns, list):
            raise ValueError("repaired_teacher_turns is not a list")
        actual = {int(turn["teacher_index"]) for turn in repaired_turns}
        if actual != expected or len(actual) != len(repaired_turns):
            raise ValueError(f"repair indices mismatch: expected={sorted(expected)}, actual={sorted(actual)}")
        normalized = []
        for item in repaired_turns:
            teacher_index = int(item["teacher_index"])
            teacher_value = validate_teacher_turn({
                key: item[key] for key in TEACHER_PROPERTIES
            })
            normalized.append({"teacher_index": teacher_index, **teacher_value})
        value["repaired_teacher_turns"] = normalized
        return {
            "candidate_id": candidate_id,
            "status": "completed",
            "repair": value,
        }

    collect_batch(config, file_paths, "repair", file_paths["repairs"], transform)


def build_repaired_dialogue(
    dialogue: dict[str, Any], repair_row: dict[str, Any],
) -> dict[str, Any]:
    rebuilt = json.loads(json.dumps(dialogue, ensure_ascii=False))
    replacements = {
        int(item["teacher_index"]): item
        for item in repair_row["repair"]["repaired_teacher_turns"]
    }
    teacher_index = 0
    for conversation_index, turn in enumerate(rebuilt["conversation"]):
        if turn.get("role") != "teacher":
            continue
        if teacher_index in replacements:
            replacement = replacements[teacher_index]
            rebuilt["conversation"][conversation_index] = {
                "turn": turn["turn"],
                "role": "teacher",
                **{key: replacement[key] for key in TEACHER_PROPERTIES},
                "repaired": True,
            }
        teacher_index += 1
    if set(replacements) - set(range(teacher_index)):
        raise ValueError("repair references an unknown teacher turn")
    return rebuilt


def submit_reaudit(config: dict[str, Any], file_paths: dict[str, Path], overwrite: bool) -> None:
    dialogues = {row["candidate_id"]: row for row in read_jsonl(file_paths["dialogues"])}
    repairs = {
        row["candidate_id"]: row for row in read_jsonl(file_paths["repairs"])
        if row.get("status") == "completed"
    }
    system = (PROMPT_DIR / "turn_quality_judge_system.txt").read_text(encoding="utf-8")
    requests = []
    for candidate_id, repair_row in repairs.items():
        if candidate_id not in dialogues:
            continue
        rebuilt = build_repaired_dialogue(dialogues[candidate_id], repair_row)
        for key, _, _, conversation_index, turn in teacher_turn_records([rebuilt]):
            requests.append({
                "custom_id": key, "method": "POST", "url": "/v1/chat/completions",
                "body": batch_body(
                    config["judge_model"], config["judge_reasoning_effort"], system,
                    audit_payload(rebuilt, conversation_index, turn), AUDIT_SCHEMA,
                ),
            })
    submit_batch(config, file_paths, "reaudit", file_paths["reaudit_input"], requests, overwrite)


def collect_reaudit(config: dict[str, Any], file_paths: dict[str, Path]) -> None:
    def transform(key: str, value: dict[str, Any]) -> dict[str, Any]:
        return {
            "turn_key": key, "status": "completed",
            "classification": classify_audit(value),
            "total_score": sum(value[name] for name in SCORE_FIELDS),
            "audit": value,
        }
    collect_batch(config, file_paths, "reaudit", file_paths["reaudits"], transform)


def sft_assistant_content(turn: dict[str, Any]) -> str:
    assessment = turn["mathematical_assessment"]
    learner = turn["learner_state"]
    support = turn["support_decision"]

    def compact(value: Any) -> str:
        return " ".join(str(value).split())

    analysis = (
        f"【数学的評価】{assessment['status']}; 検証={compact(assessment['verification'])}; "
        f"正しい部分={compact(assessment['correct_part'])}; 修正点={compact(assessment['error_part'])}\n"
        f"【生徒状態】{compact(learner['cognitive_state'])}; 感情={learner['emotion']}; "
        f"根拠={compact(learner['evidence'])}\n"
        f"【支援判断】{compact(support['next_support'])}; "
        f"変更理由={compact(support['change_reason'])}"
    )
    final = compact(turn["teacher_utterance"]) + ("\n[指導完了]" if turn["is_completed"] else "")
    return f"<analysis>\n{analysis}\n</analysis>\n<final>\n{final}\n</final>"


def build_sft_messages(dialogue: dict[str, Any], system_prompt: str) -> list[dict[str, str]]:
    """問題と最初の生徒発話を結合し、roleが必ず交互になるSFT messagesを作る。"""
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt.strip()}]
    first_student = True
    for turn in dialogue["conversation"]:
        if turn["role"] == "student":
            content = str(turn["content"]).strip()
            if first_student:
                content = f"問題: {dialogue['problem']}\n\n生徒発話: {content}"
                first_student = False
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "assistant", "content": sft_assistant_content(turn)})

    if first_student:
        raise ValueError(f"生徒発話がない対話はSFT化できません: {dialogue.get('candidate_id', 'unknown')}")
    expected = ["system"] + ["user" if index % 2 == 0 else "assistant" for index in range(len(messages) - 1)]
    actual = [message["role"] for message in messages]
    if actual != expected:
        raise ValueError(
            f"SFT messagesのroleが交互ではありません: "
            f"{dialogue.get('candidate_id', 'unknown')} {actual}"
        )
    return messages


def incomplete_reason(dialogue: dict[str, Any]) -> str | None:
    if dialogue.get("is_completed"):
        return None
    final_state = dialogue.get("final_student_state") or {}
    if final_state.get("emotion") == "bored":
        return "student_disengagement"
    initial_misconception = (dialogue.get("initial_student_state") or {}).get("active_misconception")
    if initial_misconception and final_state.get("active_misconception") == initial_misconception:
        return "persistent_misconception"
    return "max_turns"


def support_change_count(dialogues: list[dict[str, Any]]) -> int:
    no_change = {"", "なし", "変更なし", "not_applicable", "none"}
    return sum(
        str(turn.get("support_decision", {}).get("change_reason", "")).strip().lower() not in no_change
        for dialogue in dialogues
        for turn in dialogue.get("conversation", [])
        if turn.get("role") == "teacher"
    )


def finalize(config: dict[str, Any], file_paths: dict[str, Path]) -> None:
    dialogues = read_jsonl(file_paths["dialogues"])
    audit_rows = read_jsonl(file_paths["audits"])
    audits = {row["turn_key"]: row for row in audit_rows if row.get("status") == "completed"}
    repairs = {
        row["candidate_id"]: row for row in read_jsonl(file_paths["repairs"])
        if row.get("status") == "completed"
    }
    reaudit_rows = read_jsonl(file_paths["reaudits"])
    reaudits = {
        row["turn_key"]: row for row in reaudit_rows
        if row.get("status") == "completed"
    }
    accepted: list[dict[str, Any]] = []
    rejected = Counter()
    accepted_repaired_dialogues = 0
    accepted_repaired_turns = 0
    for dialogue in dialogues:
        rebuilt = json.loads(json.dumps(dialogue, ensure_ascii=False))
        valid = True
        teacher_index = 0
        initial_classifications: list[str] = []
        for turn in rebuilt["conversation"]:
            if turn.get("role") != "teacher":
                continue
            key = f"{dialogue['candidate_id']}:teacher:{teacher_index}"
            audit = audits.get(key)
            teacher_index += 1
            if not audit:
                valid = False; rejected["missing_audit"] += 1; break
            initial_classifications.append(str(audit["classification"]))
            if audit["classification"] == "Reject":
                valid = False
                rejected["initial_audit_reject"] += 1
                break
        if teacher_index == 0:
            valid = False
            rejected["no_teacher_turn"] += 1
        requires_repair = valid and "Repair" in initial_classifications
        if requires_repair:
            repair_row = repairs.get(dialogue["candidate_id"])
            if not repair_row:
                valid = False
                rejected["missing_repair"] += 1
            else:
                try:
                    rebuilt = build_repaired_dialogue(dialogue, repair_row)
                except (KeyError, TypeError, ValueError) as exc:
                    valid = False
                    rejected[f"invalid_repair_{type(exc).__name__.lower()}"] += 1
        if valid and requires_repair:
            for key, _, _, _, _ in teacher_turn_records([rebuilt]):
                reaudit = reaudits.get(key)
                if not reaudit:
                    valid = False
                    rejected["missing_reaudit"] += 1
                    break
                if reaudit["classification"] != "Keep":
                    valid = False
                    rejected["reaudit_not_keep"] += 1
                    break
        if valid:
            teacher_turns = [t for t in rebuilt["conversation"] if t.get("role") == "teacher"]
            rebuilt["is_completed"] = bool(teacher_turns and teacher_turns[-1].get("is_completed"))
            rebuilt["incomplete_reason"] = incomplete_reason(rebuilt)
            rebuilt["selection_path"] = "repair_then_full_reaudit" if requires_repair else "initial_keep"
            accepted.append(rebuilt)
            if requires_repair:
                accepted_repaired_dialogues += 1
                accepted_repaired_turns += sum(bool(turn.get("repaired")) for turn in teacher_turns)
        if len(accepted) >= int(config["target_dialogues"]):
            break

    write_jsonl(file_paths["corpus"], accepted)
    sft_system = (PROMPT_DIR / "sft_teacher_system.txt").read_text(encoding="utf-8")
    sft_rows = []
    for dialogue in accepted:
        sft_rows.append({
            "id": dialogue["candidate_id"],
            "messages": build_sft_messages(dialogue, sft_system),
        })
    write_jsonl(file_paths["sft"], sft_rows)

    record_lengths = [
        sum(len(message["content"]) for message in row["messages"])
        for row in sft_rows
    ]
    assistant_lengths = [
        len(message["content"])
        for row in sft_rows for message in row["messages"]
        if message["role"] == "assistant"
    ]

    manifest = load_manifest(config, file_paths)
    manifest["final"] = {
        "accepted_dialogues": len(accepted), "target_dialogues": int(config["target_dialogues"]),
        "completed_dialogues": sum(bool(row["is_completed"]) for row in accepted),
        "incomplete_dialogues": sum(not bool(row["is_completed"]) for row in accepted),
        "incomplete_reason_counts": dict(Counter(
            row["incomplete_reason"] for row in accepted if row["incomplete_reason"]
        )),
        "support_change_turns": support_change_count(accepted),
        "accepted_initial_keep_dialogues": len(accepted) - accepted_repaired_dialogues,
        "accepted_repaired_dialogues": accepted_repaired_dialogues,
        "accepted_repaired_turns": accepted_repaired_turns,
        "rejected_reasons": dict(rejected),
        "profile_counts": dict(Counter(row["student_profile"]["id"] for row in accepted)),
        "initial_emotion_counts": dict(Counter(row["initial_emotion"] for row in accepted)),
        "profile_initial_emotion_counts": dict(Counter(f"{row['student_profile']['id']}::{row['initial_emotion']}" for row in accepted)),
        "teacher_emotion_counts": dict(Counter(
            turn["learner_state"]["emotion"] for row in accepted for turn in row["conversation"]
            if turn.get("role") == "teacher"
        )),
        "sft_format": {
            "records": len(sft_rows),
            "assistant_targets": len(assistant_lengths),
            "average_record_characters": (
                round(sum(record_lengths) / len(record_lengths), 1) if record_lengths else 0
            ),
            "maximum_record_characters": max(record_lengths, default=0),
            "average_assistant_characters": (
                round(sum(assistant_lengths) / len(assistant_lengths), 1) if assistant_lengths else 0
            ),
            "maximum_assistant_characters": max(assistant_lengths, default=0),
            "tokenizer_length_audit_required": True,
            "truncation_policy": "Do not truncate an assistant target; split overlength records at teacher-turn boundaries.",
        },
        "initial_audit_classification_counts": dict(Counter(
            row.get("classification", "error") if row.get("status") == "completed" else "error"
            for row in audit_rows
        )),
        "reaudit_classification_counts": dict(Counter(
            row.get("classification", "error") if row.get("status") == "completed" else "error"
            for row in reaudit_rows
        )),
    }
    save_manifest(file_paths["manifest"], manifest)
    report = [
        "# v4コーパス作成結果", "",
        f"- 採択対話: {len(accepted)} / 目標 {config['target_dialogues']}",
        f"- 完了対話: {manifest['final']['completed_dialogues']}",
        f"- 未完了対話: {manifest['final']['incomplete_dialogues']}",
        f"- 初回Keep採択: {manifest['final']['accepted_initial_keep_dialogues']}",
        f"- Repair後の全対話再監査で採択: {manifest['final']['accepted_repaired_dialogues']}",
        "",
        "## プロフィール", "",
    ]
    report.extend(f"- {key}: {value}" for key, value in sorted(manifest["final"]["profile_counts"].items()))
    report.extend(["", "## 初期感情", ""])
    report.extend(f"- {key}: {value}" for key, value in sorted(manifest["final"]["initial_emotion_counts"].items()))
    file_paths["report"].write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"accepted: {len(accepted)}; sft: {len(sft_rows)}")
    if len(accepted) < int(config["target_dialogues"]):
        print("警告: 目標件数未達です。max_candidatesを増やして追加生成してください。", file=sys.stderr)


def status(config: dict[str, Any], file_paths: dict[str, Path]) -> None:
    manifest = load_manifest(config, file_paths)
    summary = {
        "candidate_dialogues": len(read_jsonl(file_paths["dialogues"])),
        "generation_errors": len(read_jsonl(file_paths["generation_errors"])),
        "audits": len(read_jsonl(file_paths["audits"])),
        "repairs": len(read_jsonl(file_paths["repairs"])),
        "reaudits": len(read_jsonl(file_paths["reaudits"])),
        "accepted": len(read_jsonl(file_paths["corpus"])),
        "batch_jobs": manifest.get("batch_jobs", {}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    file_paths = paths(config)
    file_paths["root"].mkdir(parents=True, exist_ok=True)
    commands = {
        "generate": lambda: generate(config, file_paths, args.overwrite),
        "submit-audit": lambda: submit_audit(config, file_paths, args.overwrite),
        "collect-audit": lambda: collect_audit(config, file_paths),
        "submit-repair": lambda: submit_repair(config, file_paths, args.overwrite),
        "collect-repair": lambda: collect_repair(config, file_paths),
        "submit-reaudit": lambda: submit_reaudit(config, file_paths, args.overwrite),
        "collect-reaudit": lambda: collect_reaudit(config, file_paths),
        "finalize": lambda: finalize(config, file_paths),
        "status": lambda: status(config, file_paths),
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
