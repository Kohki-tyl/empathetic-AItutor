"""ABCIへコピーして実行できるv4コーパス生成パイプライン。"""

from __future__ import annotations

import argparse
import copy
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import platform
import re
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
STUDENT_MODEL_STATE_KEYS = STUDENT_STATE_KEYS - {"acquired_knowledge"}
STUDENT_EMOTIONS = [
    "engaged", "curious", "neutral", "confused", "frustrated", "anxious",
    "bored", "eureka", "relieved", "proud",
]
INITIAL_RESPONSE_MODES = [
    "plausible_incorrect", "partial_reasoning", "help_seeking",
    "correct_but_uncertain",
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
                        "remaining_unknowns": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": sorted(STUDENT_MODEL_STATE_KEYS),
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


def student_schema_for_turn(
    previous: dict[str, Any], expected_response_mode: str | None,
    *, allow_emotion_change: bool,
) -> dict[str, Any]:
    """構造化出力の段階で、感情遷移と応答段階を現在ターンへ制約する。"""
    schema = copy.deepcopy(STUDENT_SCHEMA)
    properties = schema["json_schema"]["schema"]["properties"]
    previous_emotion = str(previous["emotion"])
    next_emotions = (
        {previous_emotion, *EMOTION_TRANSITIONS[previous_emotion]}
        if allow_emotion_change else {previous_emotion}
    )
    properties["state_after"]["properties"]["emotion"]["enum"] = sorted(next_emotions)
    allowed_stages = {
        "plausible_incorrect": ["attempt"],
        "partial_reasoning": ["observation", "attempt"],
        "help_seeking": ["help_seeking"],
        "correct_but_uncertain": ["attempt", "answer"],
        "scope_limited_help_seeking": ["observation", "help_seeking"],
    }
    if expected_response_mode in allowed_stages:
        properties["response_stage"]["enum"] = allowed_stages[expected_response_mode]
    return schema

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

DIALOGUE_AUDIT_PROPERTIES = {
    **AUDIT_PROPERTIES,
    "metadata_warnings": {"type": "array", "items": {"type": "string"}},
    "acceptable_incompleteness": {"type": "array", "items": {"type": "string"}},
}
DIALOGUE_AUDIT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "v4_dialogue_sft_audit", "strict": True,
        "schema": {
            "type": "object", "properties": DIALOGUE_AUDIT_PROPERTIES,
            "required": list(DIALOGUE_AUDIT_PROPERTIES), "additionalProperties": False,
        },
    },
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "preflight", "generate", "submit-audit", "collect-audit", "submit-repair",
            "audit-sync", "audit-dialogues-sync", "collect-repair", "submit-reaudit",
            "collect-reaudit", "finalize", "status",
        ],
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--limit", type=int,
        help="generateで今回処理する先頭候補数（選択表や本番上限は変更しない）",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="generateで処理を開始する候補index（並列区間生成用）",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="対話全体同期監査の並列数",
    )
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
        "student_model", "student_max_tokens",
        "judge_model", "judge_reasoning_effort", "repair_model",
        "repair_reasoning_effort", "questions", "problem_profile_assignments",
        "problem_selection",
        "output_dir",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"設定不足: {sorted(missing)}")
    if int(config["target_dialogues"]) > int(config["max_candidates"]):
        raise ValueError("target_dialoguesはmax_candidates以下にしてください")
    for name in ("target_dialogues", "max_candidates", "max_turns", "student_max_tokens"):
        if int(config[name]) <= 0:
            raise ValueError(f"{name}は正の整数にしてください")
    config.setdefault("student_provider", "vllm")
    if config["student_provider"] not in {"openai", "vllm"}:
        raise ValueError("student_providerはopenaiまたはvllmにしてください")
    if config["student_provider"] == "vllm":
        for name in (
            "student_model_revision", "vllm_version", "student_temperature",
            "student_top_p", "student_top_k", "student_min_p",
        ):
            if name not in config:
                raise ValueError(f"vLLM生徒に必要な設定がありません: {name}")
        if not 0 <= float(config["student_temperature"]) <= 2:
            raise ValueError("student_temperatureは0以上2以下にしてください")
        if not 0 <= float(config["student_top_p"]) <= 1:
            raise ValueError("student_top_pは0以上1以下にしてください")
        if int(config["student_top_k"]) < 0:
            raise ValueError("student_top_kは0以上にしてください")
        if not 0 <= float(config["student_min_p"]) <= 1:
            raise ValueError("student_min_pは0以上1以下にしてください")
    config.setdefault("generation_validation_attempts", 3)
    if int(config["generation_validation_attempts"]) < 1:
        raise ValueError("generation_validation_attemptsは1以上にしてください")
    config["questions"] = str(resolve_path(config["questions"], path))
    config["problem_profile_assignments"] = str(
        resolve_path(config["problem_profile_assignments"], path)
    )
    config["problem_selection"] = str(resolve_path(config["problem_selection"], path))
    config["output_dir"] = str(resolve_path(config["output_dir"], path))
    if not Path(config["questions"]).is_file():
        raise FileNotFoundError(f"問題JSONLがありません: {config['questions']}")
    if not Path(config["problem_profile_assignments"]).is_file():
        raise FileNotFoundError(
            f"問題・プロフィール対応表がありません: {config['problem_profile_assignments']}"
        )
    if not Path(config["problem_selection"]).is_file():
        raise FileNotFoundError(f"問題選択表がありません: {config['problem_selection']}")
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
        "dialogue_audits": root / "dialogue_audits.jsonl",
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
MUTABLE_LIMIT_KEYS = {"target_dialogues"}


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
    assignment_value = config.get("problem_profile_assignments")
    assignment_path = Path(assignment_value) if assignment_value else None
    selection_value = config.get("problem_selection")
    selection_path = Path(selection_value) if selection_value else None
    return {
        "questions": sha256(question_path) if question_path.exists() else "missing",
        "problem_profile_assignments": (
            sha256(assignment_path)
            if assignment_path is not None and assignment_path.exists()
            else "missing"
        ),
        "problem_selection": (
            sha256(selection_path)
            if selection_path is not None and selection_path.exists()
            else "missing"
        ),
    }


def runtime_environment(config: dict[str, Any]) -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "openai": package_version("openai"),
        "tqdm": package_version("tqdm"),
        "installed_vllm": package_version("vllm"),
        "student_provider": str(config.get("student_provider", "vllm")),
        "configured_vllm": str(config.get("vllm_version", "not-used")),
        "student_model_revision": str(config.get("student_model_revision", "alias")),
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


def initial_state(
    profile: dict[str, Any], emotion: str,
    epistemic_assignment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    confidence = 0.35 if profile.get("confidence_bias") == "underconfident" else 0.55
    misconception = profile["target_misconception"]
    if epistemic_assignment is not None:
        misconception = epistemic_assignment["misconception_model"]["label"]
    return {
        "understanding_level": max(0, min(4, int(profile["ability_level"]) - 1)),
        "confidence": confidence,
        "active_misconception": misconception,
        "emotion": emotion,
        "acquired_knowledge": [],
        "remaining_unknowns": list(profile["unknown_knowledge"]),
    }


def teacher_turn_input(
    *, problem: str, reference_solution: str, student_utterance: str,
    profile: dict[str, Any], epistemic_assignment: dict[str, Any],
    initial_emotion: str, turn_index: int,
) -> str:
    """初回だけ固定条件を渡し、以後は生徒発話だけを教師へ渡す。"""
    if turn_index > 0:
        return student_utterance
    learner_context = {
        "scope_relation": epistemic_assignment["scope_relation"],
        "prior_knowledge": profile["prior_knowledge"],
        "unknown_knowledge": profile["unknown_knowledge"],
        "max_independent_math_level": profile["max_independent_math_level"],
        "initial_emotion": initial_emotion,
        "prior_attempt_history": epistemic_assignment["prior_attempt_history"],
    }
    sections = [
        f"問題: {problem}",
        (
            f"内部検算用の参照解答: {reference_solution}\n"
            "参照解答は検算にだけ使い、生徒へそのまま提示しないでください。"
        ),
    ]
    sections.extend([
        (
            "学習者条件（問い・例・記号・公式も使用可能知識と照合してください）:\n"
            + json.dumps(learner_context, ensure_ascii=False)
        ),
        f"生徒発話: {student_utterance}",
    ])
    return "\n\n".join(sections)


def validate_student_turn(
    value: dict[str, Any], previous: dict[str, Any], *,
    allow_emotion_change: bool = True,
    allowed_knowledge: Iterable[str] = (),
    expected_response_mode: str | None = None,
    latest_teacher_utterance: str = "",
    required_initial_disclosure: str = "",
) -> dict[str, Any]:
    normalizations: list[str] = []
    expected_keys = {
        "state_after", "response_stage", "knowledge_used",
        "state_update_reason", "utterance",
    }
    schema_keys = expected_keys | {"newly_acquired_knowledge"}
    if set(value) not in (expected_keys, schema_keys):
        raise ValueError("student response keys are invalid")
    response_stage = str(value["response_stage"])
    if response_stage not in STUDENT_RESPONSE_STAGES:
        raise ValueError("student response stage is invalid")
    knowledge_used = value["knowledge_used"]
    if not isinstance(knowledge_used, list) or any(
        not isinstance(item, str) or not item.strip() for item in knowledge_used
    ):
        raise ValueError("knowledge_used must be a list of non-empty strings")
    if len(knowledge_used) != len(set(knowledge_used)):
        raise ValueError("knowledge_used contains duplicates")
    allowed = set(allowed_knowledge)
    allowed_stages = {
        "plausible_incorrect": {"attempt"},
        "partial_reasoning": {"observation", "attempt"},
        "help_seeking": {"help_seeking"},
        "correct_but_uncertain": {"attempt", "answer"},
        "scope_limited_help_seeking": {"observation", "help_seeking"},
        "natural_profile_consistent": set(STUDENT_RESPONSE_STAGES),
        "follow_latest_teacher_step_only": set(STUDENT_RESPONSE_STAGES),
    }
    if expected_response_mode and response_stage not in allowed_stages[expected_response_mode]:
        raise ValueError(
            f"response_stage={response_stage} does not match {expected_response_mode}"
        )
    state = value["state_after"]
    if not isinstance(state, dict) or set(state) not in (
        STUDENT_MODEL_STATE_KEYS, STUDENT_STATE_KEYS,
    ):
        raise ValueError("student state is incomplete")
    # 新形式ではモデルは差分だけを返し、累積リストはPython側を正本として管理する。
    # 旧形式の入力も検証関数単体では受理し、既存成果物の再開互換性を保つ。
    if "newly_acquired_knowledge" in value:
        newly_acquired_list = value["newly_acquired_knowledge"]
    else:
        reported = state.get("acquired_knowledge", previous["acquired_knowledge"])
        newly_acquired_list = [
            item for item in reported if item not in previous["acquired_knowledge"]
        ]
    if (
        not isinstance(newly_acquired_list, list)
        or any(not isinstance(item, str) or not item.strip() for item in newly_acquired_list)
        or len(newly_acquired_list) != len(set(newly_acquired_list))
    ):
        raise ValueError("newly_acquired_knowledge must contain unique non-empty strings")
    accumulated_knowledge = list(previous["acquired_knowledge"])
    accumulated_knowledge.extend(
        item for item in newly_acquired_list if item not in accumulated_knowledge
    )
    state = {**state, "acquired_knowledge": accumulated_knowledge}
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
    if not str(state["active_misconception"]).strip():
        state["active_misconception"] = previous["active_misconception"]
        normalizations.append("blank_active_misconception_preserved")
    previous_emotion = str(previous["emotion"])
    next_emotion = str(state["emotion"])
    if not allow_emotion_change and next_emotion != previous_emotion:
        raise ValueError("initial emotion changed before teacher intervention")
    if not allow_emotion_change:
        if int(state["understanding_level"]) != int(previous["understanding_level"]):
            raise ValueError("initial understanding changed before teacher intervention")
        if list(state["acquired_knowledge"]) != list(previous["acquired_knowledge"]):
            raise ValueError("initial acquired knowledge changed before teacher intervention")
        if list(state["remaining_unknowns"]) != list(previous["remaining_unknowns"]):
            raise ValueError("initial remaining unknowns changed before teacher intervention")
        if str(state["active_misconception"]) != str(previous["active_misconception"]):
            raise ValueError("initial misconception changed before teacher intervention")
        if abs(confidence - previous_confidence) > 0.1:
            raise ValueError("initial confidence changed by more than 0.1")
    newly_acquired = set(newly_acquired_list)
    if any(item not in latest_teacher_utterance for item in newly_acquired):
        raise ValueError("new knowledge was not copied from the latest teacher utterance")
    allowed.update(newly_acquired)
    if any(item not in allowed for item in knowledge_used):
        normalizations.append("knowledge_used_outside_boundary_retained_for_audit")
    if (
        allow_emotion_change and next_emotion != previous_emotion
        and next_emotion not in EMOTION_TRANSITIONS[previous_emotion]
    ):
        raise ValueError("student emotion skipped the permitted cycle")
    utterance = str(value["utterance"]).strip()
    if required_initial_disclosure and not utterance.startswith(required_initial_disclosure):
        utterance = required_initial_disclosure + utterance
        normalizations.append("required_attempt_history_prefixed")
    if not utterance or len(utterance) > 500 or utterance.startswith(("{", "[")):
        raise ValueError("student utterance is invalid")
    if any(marker in utterance.lower() for marker in ["<analysis>", "state_after", "state_update_reason"]):
        raise ValueError("student utterance leaks internal state")
    reason = str(value["state_update_reason"]).strip()
    if not reason or len(reason) > 1000:
        raise ValueError("student state update reason is invalid")
    value["utterance"] = utterance
    value["response_stage"] = response_stage
    value["knowledge_used"] = knowledge_used
    value["state_update_reason"] = reason
    value["state_after"] = {key: state[key] for key in STUDENT_STATE_KEYS}
    value["newly_acquired_knowledge"] = newly_acquired_list
    value["state_normalizations"] = normalizations
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
    if value["is_completed"]:
        if support["next_support"].strip() != "なし" or support["change_reason"].strip() != "なし":
            raise ValueError("completed instruction cannot contain next support")
        follow_up_markers = (
            "?", "？", "確認してみましょう", "考えてみましょう",
            "説明できますか", "いくつになりますか", "求めてみましょう",
        )
        if any(marker in utterance for marker in follow_up_markers):
            raise ValueError("completed teacher utterance cannot ask a follow-up question")
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


MATH_TRAIN_ID = re.compile(r"^math_train_(\d+)$")


def ordered_math_questions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """math_train_0以降を辞書順ではなく数値順に返す。"""
    numbered: list[tuple[int, dict[str, Any]]] = []
    seen: set[int] = set()
    for row in rows:
        source_id = str(row.get("id") or row.get("source_id") or "")
        match = MATH_TRAIN_ID.fullmatch(source_id)
        if not match:
            continue
        number = int(match.group(1))
        if number in seen:
            raise ValueError(f"問題IDの数値部分が重複しています: {source_id}")
        seen.add(number)
        numbered.append((number, row))
    numbered.sort(key=lambda item: item[0])
    return [row for _, row in numbered]


def problem_level(metadata: dict[str, Any]) -> int:
    value = metadata.get("level")
    if isinstance(value, bool):
        raise ValueError("問題levelは1以上5以下の整数にしてください")
    try:
        level = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("問題levelは1以上5以下の整数にしてください") from exc
    if not 1 <= level <= 5:
        raise ValueError("問題levelは1以上5以下の整数にしてください")
    return level


def effective_initial_response_mode(
    configured_mode: str, epistemic_assignment: dict[str, Any],
) -> str:
    if epistemic_assignment["scope_relation"] in {"one_step_beyond", "far_beyond"}:
        return "scope_limited_help_seeking"
    return configured_mode


def required_initial_disclosure(epistemic_assignment: dict[str, Any]) -> str:
    if epistemic_assignment.get("scope_relation") != "far_beyond":
        return ""
    history = epistemic_assignment.get("prior_attempt_history")
    if not isinstance(history, dict):
        return ""
    disclosure = str(history.get("required_initial_disclosure", "")).strip()
    return "" if disclosure == "なし" else disclosure


def question_content_hash(problem: str, solution: str) -> str:
    encoded = json.dumps(
        {"problem": problem, "reference_solution": solution},
        ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_problem_profile_assignments(
    path: Path, questions: list[dict[str, Any]], profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != len(questions):
        raise ValueError("問題・プロフィール対応表の件数が問題数と一致しません")
    profile_ids = {str(profile["id"]) for profile in profiles}
    validated: list[dict[str, Any]] = []
    for index, (row, question) in enumerate(zip(rows, questions)):
        source_id, problem, solution, _ = question_fields(question)
        if row.get("source_id") != source_id or row.get("order_index") != index:
            raise ValueError(f"問題・プロフィール対応表の順序が不正です: {source_id}")
        if row.get("question_sha256") != question_content_hash(problem, solution):
            raise ValueError(f"対応表作成後に問題内容が変わっています: {source_id}")
        if row.get("profile_id") not in profile_ids:
            raise ValueError(f"対応表のprofile_idが不正です: {source_id}")
        if row.get("initial_emotion") not in STUDENT_EMOTIONS:
            raise ValueError(f"対応表のinitial_emotionが不正です: {source_id}")
        attempt_history = row.get("prior_attempt_history")
        required_attempt_keys = {
            "attempt_count", "attempts", "repeated_stuck_point",
            "received_help", "required_initial_disclosure",
        }
        if not isinstance(attempt_history, dict) or set(attempt_history) != required_attempt_keys:
            raise ValueError(f"対応表のprior_attempt_historyが不正です: {source_id}")
        relation = row.get("scope_relation")
        if relation not in {
            "mastered", "frontier", "one_step_beyond", "far_beyond",
        }:
            raise ValueError(f"対応表のscope_relationが不正です: {source_id}")
        attempts = attempt_history["attempts"]
        if not isinstance(attempts, list):
            raise ValueError(f"対応表の試行一覧が不正です: {source_id}")
        if relation == "far_beyond":
            required_trial_keys = {
                "attempt_number", "strategy", "stopped_at", "outcome",
            }
            if (
                row.get("initial_emotion") != "frustrated"
                or int(attempt_history["attempt_count"]) != 2
                or len(attempts) != 2
                or [attempt.get("attempt_number") for attempt in attempts] != [1, 2]
                or any(not isinstance(attempt, dict) or set(attempt) != required_trial_keys for attempt in attempts)
                or any(attempt.get("stopped_at") != attempt_history["repeated_stuck_point"] for attempt in attempts)
                or any(attempt.get("outcome") != "未解決" for attempt in attempts)
                or attempt_history["repeated_stuck_point"] == "なし"
                or attempt_history["received_help"] is not False
                or attempt_history["required_initial_disclosure"] == "なし"
            ):
                raise ValueError(f"far_beyondに明示的な2回の事前試行履歴がありません: {source_id}")
        elif (
            int(attempt_history["attempt_count"]) != 0
            or attempts
            or attempt_history["required_initial_disclosure"] != "なし"
        ):
            raise ValueError(f"far_beyond以外に事前試行履歴があります: {source_id}")
        if row.get("initial_response_mode") not in {
            *INITIAL_RESPONSE_MODES, "scope_limited_help_seeking",
        }:
            raise ValueError(f"対応表のinitial_response_modeが不正です: {source_id}")
        misconception = row.get("misconception_model")
        required_misconception_keys = {
            "id", "label", "trigger", "faulty_procedure",
            "observable_signature", "repair_criterion",
        }
        if not isinstance(misconception, dict) or set(misconception) != required_misconception_keys:
            raise ValueError(f"対応表のmisconception_modelが不正です: {source_id}")
        validated.append(row)
    return validated


def load_problem_selection(
    path: Path, assignments: list[dict[str, Any]], expected_count: int,
) -> list[dict[str, Any]]:
    value = read_json(path)
    records = value.get("records") if isinstance(value, dict) else None
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError(f"問題選択表は{expected_count}件必要です")
    if value.get("source_partition") != {"start": 0, "end_exclusive": 800}:
        raise ValueError("コーパス問題は先頭800件から選択してください")
    if int(value.get("per_scope_relation", 0)) != 30 or expected_count != 120:
        raise ValueError("コーパス問題は4範囲関係から各30件、計120件にしてください")
    assignment_by_id = {str(row["source_id"]): row for row in assignments}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        source_id = str(record.get("source_id", ""))
        assignment = assignment_by_id.get(source_id)
        if not source_id or source_id in seen or assignment is None:
            raise ValueError(f"問題選択表に欠損・重複があります: {source_id}")
        if int(assignment["order_index"]) >= 800:
            raise ValueError(f"テスト用問題がコーパス選択表に含まれています: {source_id}")
        if bool(assignment["curriculum_annotation"].get("requires_human_review")):
            raise ValueError(f"人手確認対象の問題はコーパス生成できません: {source_id}")
        knowledge_audit = assignment.get("knowledge_boundary_audit") or {}
        if not bool(knowledge_audit.get("relation_consistent")):
            raise ValueError(f"問題とプロフィールの知識境界が不整合です: {source_id}")
        if (
            assignment.get("scope_relation") == "mastered"
            and knowledge_audit.get("not_in_prior_knowledge")
        ):
            raise ValueError(f"未習概念をmasteredへ割り当てています: {source_id}")
        for key in ("order_index", "scope_relation", "profile_id", "question_sha256"):
            if record.get(key) != assignment.get(key):
                raise ValueError(f"問題選択表と対応表が一致しません: {source_id}/{key}")
        seen.add(source_id)
        selected.append(assignment)
    counts = Counter(row["scope_relation"] for row in selected)
    if counts != Counter({relation: 30 for relation in (
        "mastered", "frontier", "one_step_beyond", "far_beyond",
    )}):
        raise ValueError(f"問題選択表の範囲関係が不均衡です: {dict(counts)}")
    return selected


def preflight(config: dict[str, Any]) -> None:
    """高価なモデル起動前に、ABCI実行資産と主要な版を読み取り専用で検証する。"""
    load_env_file(BASE_DIR / ".env")
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY")):
        raise RuntimeError("v4/.envにOPENAI_API_KEYを設定してください")
    provider = str(config.get("student_provider", "vllm"))
    installed_vllm = package_version("vllm")
    torch_cuda = "not-used"
    if provider == "vllm":
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError(
                f"Python 3.12が必要です: detected={platform.python_version()}"
            )
        if installed_vllm != str(config["vllm_version"]):
            raise RuntimeError(
                f"vLLM version mismatch: installed={installed_vllm}, "
                f"config={config['vllm_version']}"
            )
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorchがインストールされていません") from exc
        torch_cuda = str(torch.version.cuda)
        if torch_cuda != "13.0":
            raise RuntimeError(
                f"PyTorch CUDA 13.0が必要です: detected={torch_cuda}"
            )

    profiles = read_json(PROMPT_DIR / "student_profiles.json")
    questions = ordered_math_questions(read_jsonl(Path(config["questions"])))
    assignments = load_problem_profile_assignments(
        Path(config["problem_profile_assignments"]), questions, profiles,
    )
    selected = load_problem_selection(
        Path(config["problem_selection"]), assignments,
        int(config["max_candidates"]),
    )
    test_selection_path = BASE_DIR / "assignments" / "test_120_selection.json"
    test_selection = read_json(test_selection_path)
    test_records = test_selection.get("records", [])
    if (
        test_selection.get("source_partition")
        != {"start": 800, "end_exclusive": 1000}
        or len(test_records) != 120
        or len({str(row.get("source_id")) for row in test_records}) != 120
        or Counter(str(row.get("scope_relation")) for row in test_records)
        != Counter({relation: 30 for relation in (
            "mastered", "frontier", "one_step_beyond", "far_beyond",
        )})
    ):
        raise RuntimeError("同梱test-v4選択表が不正です")
    assignment_by_id = {str(row["source_id"]): row for row in assignments}
    for record in test_records:
        source_id = str(record.get("source_id", ""))
        assignment = assignment_by_id.get(source_id)
        if assignment is None or int(assignment["order_index"]) < 800:
            raise RuntimeError(f"test-v4選択表と対応表が不一致です: {source_id}")
        for key in ("order_index", "scope_relation", "profile_id", "question_sha256"):
            if record.get(key) != assignment.get(key):
                raise RuntimeError(f"test-v4選択表が不一致です: {source_id}/{key}")
    print(json.dumps({
        "status": "ready",
        "student_provider": provider,
        "python": platform.python_version(),
        "torch_cuda": torch_cuda,
        "vllm": installed_vllm,
        "questions": len(questions),
        "corpus_selection": len(selected),
        "test_selection": len(test_records),
    }, ensure_ascii=False, indent=2))


def generate(
    config: dict[str, Any], file_paths: dict[str, Path], overwrite: bool,
    limit: int | None = None, start: int = 0,
) -> None:
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
    emotion_rows = emotion_config["emotions"]
    emotion_by_name = {row["name"]: row for row in emotion_rows}
    all_questions = ordered_math_questions(read_jsonl(Path(config["questions"])))
    epistemic_assignment_pool = load_problem_profile_assignments(
        Path(config["problem_profile_assignments"]), all_questions, profiles,
    )
    epistemic_assignments = load_problem_selection(
        Path(config["problem_selection"]), epistemic_assignment_pool,
        int(config["max_candidates"]),
    )
    question_by_id = {
        question_fields(question)[0]: question for question in all_questions
    }
    questions = [question_by_id[str(row["source_id"])] for row in epistemic_assignments]
    profiles_by_id = {str(profile["id"]): profile for profile in profiles}

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEYを設定してください")
    provider = str(config.get("student_provider", "vllm"))
    teacher_client = OpenAI(api_key=api_key)
    if provider == "openai":
        student_client = OpenAI(api_key=api_key)
    else:
        student_base_url = os.getenv("STUDENT_BASE_URL")
        if not student_base_url:
            raise RuntimeError(
                "STUDENT_BASE_URLが未設定です。PBSを使うか、起動したvLLMのURLを設定してください"
            )
        student_client = OpenAI(
            api_key=os.getenv("STUDENT_API_KEY", "EMPTY"),
            base_url=student_base_url,
        )
    teacher_system = (PROMPT_DIR / "teacher_system.txt").read_text(encoding="utf-8")
    student_template = (PROMPT_DIR / "student_system.txt").read_text(encoding="utf-8")
    manifest = load_manifest(config, file_paths)

    generation_count = int(config["max_candidates"])
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limitは1以上にしてください")
        generation_count = min(generation_count, limit)
    if start < 0 or start >= generation_count:
        raise ValueError("--startは0以上かつ生成終了index未満にしてください")
    for index in tqdm(range(start, generation_count), desc="generate"):
        candidate_id = f"v4-{index:04d}"
        if candidate_id in existing:
            continue
        epistemic_assignment = epistemic_assignments[index]
        base_profile = profiles_by_id[str(epistemic_assignment["profile_id"])]
        profile = json.loads(json.dumps(base_profile, ensure_ascii=False))
        profile["problem_epistemic_state"] = {
            "curriculum_annotation": epistemic_assignment["curriculum_annotation"],
            "scope_relation": epistemic_assignment["scope_relation"],
            "misconception_model": epistemic_assignment["misconception_model"],
            "prior_attempt_history": epistemic_assignment["prior_attempt_history"],
            "initial_response_constraint": epistemic_assignment["initial_response_constraint"],
        }
        emotion = str(epistemic_assignment["initial_emotion"])
        source_id, problem, solution, source_metadata = question_fields(questions[index])
        level = problem_level(source_metadata)
        requested_response_mode = str(epistemic_assignment["initial_response_mode"])
        response_mode = effective_initial_response_mode(
            requested_response_mode, epistemic_assignment,
        )
        state = initial_state(profile, emotion, epistemic_assignment)
        initial = dict(state)
        student_system = student_template.replace("{STUDENT_PROFILE}", profile_text(profile))
        teacher_history: list[dict[str, str]] = [{"role": "system", "content": teacher_system}]
        recent_dialogue: list[dict[str, str]] = []
        conversation: list[dict[str, Any]] = []
        generation_diagnostics: list[dict[str, Any]] = []
        generation_error = None
        try:
            for turn_index in range(int(config["max_turns"])):
                student_payload = {
                    "problem": problem,
                    "turn": turn_index,
                    "state_before": state,
                    "initial_emotion": emotion,
                    "initial_emotion_condition": emotion_by_name[emotion],
                    "initial_response_condition": response_mode if turn_index == 0 else "follow_latest_teacher_step_only",
                    "problem_level": level,
                    "epistemic_state_specification": epistemic_assignment,
                    "knowledge_boundary": {
                        "prior_knowledge": profile["prior_knowledge"],
                        "acquired_knowledge": state["acquired_knowledge"],
                        "unknown_knowledge": profile["unknown_knowledge"],
                        "max_independent_math_level": profile["max_independent_math_level"],
                    },
                    "latest_teacher_utterance": recent_dialogue[-1]["content"] if recent_dialogue else "",
                    "recent_dialogue": recent_dialogue[-6:],
                    "instruction": "問題を解き始めてください" if turn_index == 0 else "教師の最新発話へ生徒として応答してください",
                }
                student_messages = [
                    {"role": "system", "content": student_system},
                    {"role": "user", "content": json.dumps(student_payload, ensure_ascii=False)},
                ]
                last_error = None
                for validation_attempt in range(int(config["generation_validation_attempts"])):
                    retry_messages = list(student_messages)
                    if last_error is not None:
                        disclosure_retry = (
                            " utteranceの先頭を次の文字列と完全一致させてください: "
                            f"{required_initial_disclosure(epistemic_assignment)}"
                            if turn_index == 0 and required_initial_disclosure(epistemic_assignment)
                            else ""
                        )
                        retry_messages.append({
                            "role": "user",
                            "content": (
                                f"前回出力は検証エラーでした: {last_error}。"
                                "プロフィール、初期状態、感情表出を変えず、JSONを再生成してください。"
                                f"{disclosure_retry}"
                            ),
                        })
                    turn_schema = student_schema_for_turn(
                        state,
                        response_mode if turn_index == 0 else "follow_latest_teacher_step_only",
                        allow_emotion_change=turn_index > 0,
                    )
                    if provider == "openai":
                        raw_student = chat_call(
                            student_client, config["student_model"], retry_messages,
                            turn_schema,
                            reasoning_effort=config.get("student_reasoning_effort", "none"),
                            max_completion_tokens=int(config["student_max_tokens"]),
                        )
                    else:
                        raw_student = chat_call(
                            student_client, config["student_model"], retry_messages,
                            turn_schema, temperature=float(config["student_temperature"]),
                            seed=(
                                int(config["seed"]) + index * 100 + turn_index
                                + validation_attempt * 10_000
                            ),
                            max_completion_tokens=int(config["student_max_tokens"]),
                            use_max_tokens=True,
                            extra_body={
                                "top_p": float(config.get("student_top_p", 0.95)),
                                "top_k": int(config.get("student_top_k", 20)),
                                "min_p": float(config.get("student_min_p", 0)),
                            },
                        )
                    try:
                        student_value = validate_student_turn(
                            raw_student, state,
                            allow_emotion_change=turn_index > 0,
                            allowed_knowledge=[
                                *profile["prior_knowledge"],
                                *state["acquired_knowledge"],
                            ],
                            expected_response_mode=(
                                response_mode if turn_index == 0
                                else "follow_latest_teacher_step_only"
                            ),
                            latest_teacher_utterance=(
                                recent_dialogue[-1]["content"] if recent_dialogue else ""
                            ),
                            required_initial_disclosure=(
                                required_initial_disclosure(epistemic_assignment)
                                if turn_index == 0 else ""
                            ),
                        )
                        break
                    except ValueError as exc:
                        last_error = exc
                        generation_diagnostics.append({
                            "role": "student", "turn": turn_index,
                            "attempt": validation_attempt + 1,
                            "validation_error": str(exc),
                            "invalid_output": raw_student,
                        })
                else:
                    raise RuntimeError(
                        f"student validation failed after retries: {last_error}"
                    ) from last_error
                state = student_value["state_after"]
                student_turn = {
                    "turn": turn_index, "role": "student", "content": student_value["utterance"],
                    "response_stage": student_value["response_stage"],
                    "knowledge_used": student_value["knowledge_used"],
                    "newly_acquired_knowledge": student_value["newly_acquired_knowledge"],
                    "state_after": state, "state_update_reason": student_value["state_update_reason"],
                    "state_normalizations": student_value.get("state_normalizations", []),
                }
                conversation.append(student_turn)
                recent_dialogue.append({"role": "student", "content": student_value["utterance"]})
                teacher_user_content = teacher_turn_input(
                    problem=problem,
                    reference_solution=solution,
                    student_utterance=student_value["utterance"],
                    profile=profile,
                    epistemic_assignment=epistemic_assignment,
                    initial_emotion=emotion,
                    turn_index=turn_index,
                )
                teacher_history.append({"role": "user", "content": teacher_user_content})

                last_error = None
                for validation_attempt in range(int(config["generation_validation_attempts"])):
                    retry_history = list(teacher_history)
                    if last_error is not None:
                        retry_history.append({
                            "role": "user",
                            "content": (
                                f"前回出力は検証エラーでした: {last_error}。"
                                "数学的検算、完了判定、次の支援を整合させてJSONを再生成してください。"
                            ),
                        })
                    raw_teacher = chat_call(
                        teacher_client, config["teacher_model"], retry_history,
                        TEACHER_SCHEMA,
                        reasoning_effort=config.get("teacher_reasoning_effort"),
                        max_completion_tokens=3000,
                    )
                    try:
                        teacher_value = validate_teacher_turn(raw_teacher)
                        break
                    except ValueError as exc:
                        last_error = exc
                        generation_diagnostics.append({
                            "role": "teacher", "turn": turn_index,
                            "attempt": validation_attempt + 1,
                            "validation_error": str(exc),
                            "invalid_output": raw_teacher,
                        })
                else:
                    raise RuntimeError(
                        f"teacher validation failed after retries: {last_error}"
                    ) from last_error
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
            "generation_condition": {
                "requested_initial_response_mode": requested_response_mode,
                "effective_initial_response_mode": response_mode,
                "problem_level": level,
                "knowledge_gate_active": response_mode == "scope_limited_help_seeking",
                "problem_profile_assignment": epistemic_assignment,
            },
            "generation_diagnostics": generation_diagnostics,
            "models": {
                "teacher": config["teacher_model"],
                "student": config["student_model"],
                "student_provider": provider,
                "student_revision": config.get("student_model_revision", "alias"),
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
        "generation_condition": dialogue.get("generation_condition", {}),
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


def classify_dialogue_audit(audit: dict[str, Any]) -> str:
    """教師SFT適格性を判定し、生徒の内部メタデータ警告は相殺要因にしない。"""
    minimum = min(int(audit[name]) for name in SCORE_FIELDS)
    required_true = (
        "mathematically_correct", "student_answer_assessed_correctly",
        "cognitive_state_grounded", "emotion_grounded",
        "analysis_reflected_in_utterance", "student_profile_consistent",
        "student_role_consistent", "student_state_update_plausible",
        "initial_emotion_utterance_consistent", "completion_decision_appropriate",
    )
    required_false = (
        "false_affirmation", "direct_answer_without_need", "critical_failure",
    )
    acceptable_incompleteness = audit.get("acceptable_incompleteness", [])
    if (
        minimum >= 8
        and all(bool(audit[name]) for name in required_true)
        and all(not bool(audit[name]) for name in required_false)
        and not audit["issues"]
        and not audit["repair_instructions"]
    ):
        return "Keep"
    # 高難度問題を正確かつ共感的に支援している途中でmax_turnsへ達した例は、
    # 最終解・検算・完了確認の欠如や足場の細かさだけでは除外しない。
    if acceptable_incompleteness:
        core_scores = (
            "mathematical_accuracy_score", "error_diagnosis_recovery_score",
            "cognitive_empathy_score", "emotional_support_score",
        )
        observable_context_valid = all(bool(audit[name]) for name in (
            "mathematically_correct", "student_answer_assessed_correctly",
            "cognitive_state_grounded", "emotion_grounded",
            "analysis_reflected_in_utterance", "student_profile_consistent",
            "student_role_consistent", "student_state_update_plausible",
            "initial_emotion_utterance_consistent",
        ))
        if (
            all(int(audit[name]) >= 8 for name in core_scores)
            and observable_context_valid
            and not any(bool(audit[name]) for name in (
                "false_affirmation", "direct_answer_without_need", "critical_failure",
            ))
            and not audit["issues"]
            and not audit["repair_instructions"]
        ):
            return "Keep"
    if bool(audit["context_repairable"]) and bool(audit["repair_instructions"]):
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


def audit_sync(
    config: dict[str, Any], file_paths: dict[str, Path], overwrite: bool,
) -> None:
    """Batchのファイル経路が利用できない場合の再開可能な同期監査。"""
    load_env_file(BASE_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEYを設定してください")
    manifest = load_manifest(config, file_paths)
    if overwrite:
        invalidate_batch_stage(manifest, file_paths, "audit")
        save_manifest(file_paths["manifest"], manifest)
    existing = {
        str(row["turn_key"]) for row in read_jsonl(file_paths["audits"])
        if row.get("status") == "completed"
    }
    records = list(teacher_turn_records(read_jsonl(file_paths["dialogues"])))
    system = (PROMPT_DIR / "turn_quality_judge_system.txt").read_text(encoding="utf-8")
    client = OpenAI(api_key=api_key)
    for key, dialogue, _, conversation_index, turn in tqdm(records, desc="audit-sync"):
        if key in existing:
            continue
        value = chat_call(
            client, config["judge_model"],
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(
                    audit_payload(dialogue, conversation_index, turn), ensure_ascii=False,
                )},
            ],
            AUDIT_SCHEMA,
            reasoning_effort=config.get("judge_reasoning_effort"),
            max_completion_tokens=4000,
        )
        append_jsonl(file_paths["audits"], {
            "turn_key": key,
            "status": "completed",
            "classification": classify_audit(value),
            "total_score": sum(value[name] for name in SCORE_FIELDS),
            "audit": value,
        })
        existing.add(key)
    manifest = load_manifest(config, file_paths)
    manifest.setdefault("batch_jobs", {})["audit"] = {
        "status": "completed_sync",
        "request_count": len(records),
        "collected": True,
    }
    save_manifest(file_paths["manifest"], manifest)
    print(f"audited synchronously: {len(existing)} records")


def audit_dialogues_sync(
    config: dict[str, Any], file_paths: dict[str, Path], overwrite: bool,
    workers: int = 1,
) -> None:
    """1対話を1リクエストとして監査し、教師SFT適格性を保存する。"""
    load_env_file(BASE_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEYを設定してください")
    if overwrite:
        file_paths["dialogue_audits"].unlink(missing_ok=True)
    existing = {
        str(row["candidate_id"]) for row in read_jsonl(file_paths["dialogue_audits"])
        if row.get("status") == "completed"
    }
    dialogues = read_jsonl(file_paths["dialogues"])
    system = (PROMPT_DIR / "dialogue_quality_judge_system.txt").read_text(encoding="utf-8")
    if workers < 1:
        raise ValueError("--workersは1以上にしてください")
    pending = [d for d in dialogues if str(d["candidate_id"]) not in existing]

    def audit_one(dialogue: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        value = chat_call(
            client, config["judge_model"],
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(dialogue, ensure_ascii=False)},
            ],
            DIALOGUE_AUDIT_SCHEMA,
            reasoning_effort=config.get("judge_reasoning_effort"),
            max_completion_tokens=5000,
        )
        return str(dialogue["candidate_id"]), value

    client = OpenAI(api_key=api_key)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(audit_one, dialogue): dialogue for dialogue in pending}
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures),
            desc="audit-dialogues-sync",
        ):
            dialogue = futures[future]
            candidate_id = str(dialogue["candidate_id"])
            try:
                _, value = future.result()
                row = {
                    "candidate_id": candidate_id, "status": "completed",
                    "classification": classify_dialogue_audit(value),
                    "total_score": sum(value[name] for name in SCORE_FIELDS),
                    "audit": value,
                }
                existing.add(candidate_id)
            except Exception as exc:
                row = {
                    "candidate_id": candidate_id, "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            append_jsonl(file_paths["dialogue_audits"], row)
    manifest = load_manifest(config, file_paths)
    manifest.setdefault("batch_jobs", {})["dialogue_audit"] = {
        "status": "completed_sync", "request_count": len(dialogues), "collected": True,
        "prompt_sha256": sha256(PROMPT_DIR / "dialogue_quality_judge_system.txt"),
    }
    save_manifest(file_paths["manifest"], manifest)
    print(f"audited dialogues synchronously: {len(existing)} records")


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


def validate_dialogue_knowledge_contract(dialogue: dict[str, Any]) -> None:
    profile = dialogue["student_profile"]
    previous = dialogue["initial_student_state"]
    latest_teacher_utterance = ""
    for turn in dialogue["conversation"]:
        if turn.get("role") == "teacher":
            latest_teacher_utterance = str(
                turn.get("teacher_utterance") or turn.get("content") or ""
            )
            continue
        if turn.get("role") != "student":
            continue
        state = turn.get("state_after")
        knowledge_used = turn.get("knowledge_used")
        if not isinstance(state, dict) or not isinstance(knowledge_used, list):
            raise ValueError("student knowledge metadata is missing")
        newly_acquired = set(state["acquired_knowledge"]) - set(
            previous["acquired_knowledge"]
        )
        if any(item not in latest_teacher_utterance for item in newly_acquired):
            raise ValueError("dialogue contains knowledge not introduced by the latest teacher")
        allowed = (
            set(profile["prior_knowledge"])
            | set(previous["acquired_knowledge"])
            | newly_acquired
        )
        if any(item not in allowed for item in knowledge_used):
            raise ValueError("dialogue contains out-of-bound knowledge_used")
        previous = state


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
    validate_dialogue_knowledge_contract(rebuilt)
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
    initial_emotion = str(dialogue.get("initial_emotion", "")).strip()
    if initial_emotion not in STUDENT_EMOTIONS:
        raise ValueError(
            f"SFT化に有効な初期感情ラベルがありません: "
            f"{dialogue.get('candidate_id', 'unknown')}"
        )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt.strip()}]
    first_student = True
    for turn in dialogue["conversation"]:
        if turn["role"] == "student":
            content = str(turn["content"]).strip()
            if first_student:
                content = (
                    f"問題: {dialogue['problem']}\n\n"
                    f"初期感情ラベル: {initial_emotion}\n\n"
                    f"生徒発話: {content}"
                )
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
        "initial_response_mode_counts": dict(Counter(
            row.get("generation_condition", {}).get(
                "effective_initial_response_mode", "legacy_or_unknown"
            )
            for row in accepted
        )),
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
    report.extend(["", "## 初回応答条件", ""])
    report.extend(
        f"- {key}: {value}"
        for key, value in sorted(manifest["final"]["initial_response_mode_counts"].items())
    )
    file_paths["report"].write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"accepted: {len(accepted)}; sft: {len(sft_rows)}")
    if len(accepted) < int(config["target_dialogues"]):
        print(
            "警告: 固定120候補では目標件数未達です。採択基準または候補設計を再検討し、"
            "新しい選択表とoutput_dirで再実行してください。",
            file=sys.stderr,
        )


def status(config: dict[str, Any], file_paths: dict[str, Path]) -> None:
    manifest = load_manifest(config, file_paths)
    summary = {
        "candidate_dialogues": len(read_jsonl(file_paths["dialogues"])),
        "generation_errors": len(read_jsonl(file_paths["generation_errors"])),
        "audits": len(read_jsonl(file_paths["audits"])),
        "dialogue_audits": len(read_jsonl(file_paths["dialogue_audits"])),
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
    commands = {
        "preflight": lambda: preflight(config),
        "generate": lambda: generate(
            config, file_paths, args.overwrite, args.limit, args.start,
        ),
        "submit-audit": lambda: submit_audit(config, file_paths, args.overwrite),
        "collect-audit": lambda: collect_audit(config, file_paths),
        "audit-sync": lambda: audit_sync(config, file_paths, args.overwrite),
        "audit-dialogues-sync": lambda: audit_dialogues_sync(
            config, file_paths, args.overwrite, args.workers,
        ),
        "submit-repair": lambda: submit_repair(config, file_paths, args.overwrite),
        "collect-repair": lambda: collect_repair(config, file_paths),
        "submit-reaudit": lambda: submit_reaudit(config, file_paths, args.overwrite),
        "collect-reaudit": lambda: collect_reaudit(config, file_paths),
        "finalize": lambda: finalize(config, file_paths),
        "status": lambda: status(config, file_paths),
    }
    if args.command != "preflight":
        file_paths["root"].mkdir(parents=True, exist_ok=True)
    commands[args.command]()


if __name__ == "__main__":
    main()
