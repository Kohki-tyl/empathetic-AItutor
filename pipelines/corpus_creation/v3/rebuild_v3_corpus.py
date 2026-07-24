"""既存コーパスを教師ターン単位で監査・修正し、v3コーパスを構築する。"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]
DEFAULT_INPUT = REPO_ROOT / "pipelines" / "corpus_creation" / "500_empathetic_dialogues.jsonl"
DEFAULT_AUDIT = BASE_DIR / "data" / "turn_audits.jsonl"
DEFAULT_REPAIRS = BASE_DIR / "data" / "turn_repairs.jsonl"
DEFAULT_OUTPUT = BASE_DIR / "data" / "v3_rebuilt_corpus.jsonl"
DEFAULT_MANIFEST = BASE_DIR / "data" / "v3_rebuilt_corpus_manifest.json"
JUDGE_PROMPT = BASE_DIR / "prompts" / "turn_quality_judge_system.txt"
REPAIR_PROMPT = BASE_DIR / "prompts" / "turn_repair_system.txt"
ENV_FILE = REPO_ROOT / ".env"

EMOTIONS = [
    "Engaged", "Curious", "Neutral", "Confusion", "Frustrated",
    "Bored", "Anxious", "Eureka", "Proud", "Relieved",
]

AUDIT_SCORE_FIELDS = [
    "mathematical_accuracy_score",
    "student_assessment_score",
    "cognitive_empathy_score",
    "emotional_alignment_score",
    "scaffolding_score",
    "dialogue_control_score",
    "verification_completion_score",
]

AUDIT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "teacher_turn_quality_audit",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "mathematical_accuracy_score": {"type": "integer", "minimum": 0, "maximum": 4},
                "student_assessment_score": {"type": "integer", "minimum": 0, "maximum": 3},
                "cognitive_empathy_score": {"type": "integer", "minimum": 0, "maximum": 3},
                "emotional_alignment_score": {"type": "integer", "minimum": 0, "maximum": 2},
                "scaffolding_score": {"type": "integer", "minimum": 0, "maximum": 3},
                "dialogue_control_score": {"type": "integer", "minimum": 0, "maximum": 2},
                "verification_completion_score": {"type": "integer", "minimum": 0, "maximum": 3},
                "mathematically_correct": {"type": "boolean"},
                "mathematical_verification": {"type": "string"},
                "student_answer_assessed_correctly": {"type": "boolean"},
                "cognitive_state_grounded": {"type": "boolean"},
                "emotion_grounded": {"type": "boolean"},
                "emotion_math_separated": {"type": "boolean"},
                "scaffolding_appropriate": {"type": "boolean"},
                "non_repetitive": {"type": "boolean"},
                "cognitive_load_appropriate": {"type": "boolean"},
                "understanding_verified": {"type": "boolean"},
                "false_affirmation": {"type": "boolean"},
                "direct_answer_without_need": {"type": "boolean"},
                "critical_failure": {"type": "boolean"},
                "context_repairable": {"type": "boolean"},
                "unrepairable_reason": {"type": "string"},
                "issues": {"type": "array", "items": {"type": "string"}},
                "repair_instructions": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": [
                "mathematical_accuracy_score", "student_assessment_score",
                "cognitive_empathy_score", "emotional_alignment_score",
                "scaffolding_score", "dialogue_control_score",
                "verification_completion_score",
                "mathematically_correct", "mathematical_verification",
                "student_answer_assessed_correctly",
                "cognitive_state_grounded", "emotion_grounded", "emotion_math_separated",
                "scaffolding_appropriate", "non_repetitive", "cognitive_load_appropriate",
                "understanding_verified", "false_affirmation", "direct_answer_without_need",
                "critical_failure", "context_repairable", "unrepairable_reason",
                "issues", "repair_instructions", "reason",
            ],
            "additionalProperties": False,
        },
    },
}

REPAIR_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "repaired_teacher_turn",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "thought_process": {"type": "string"},
                "student_emotion": {"type": "string", "enum": EMOTIONS},
                "roadmap_breakdown": {"type": "string"},
                "next_step_plan": {"type": "string"},
                "content": {"type": "string"},
                "is_completed": {"type": "boolean"},
                "change_summary": {"type": "string"},
            },
            "required": [
                "thought_process", "student_emotion", "roadmap_breakdown",
                "next_step_plan", "content", "is_completed", "change_summary",
            ],
            "additionalProperties": False,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="教師ターンを監査・修正してv3コーパスを構築する")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--repair-output", type=Path, default=DEFAULT_REPAIRS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--judge-model", default="gpt-5.4")
    parser.add_argument("--repair-model", default="gpt-5.4")
    parser.add_argument("--limit", type=int, help="先頭から処理する対話数")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8, help="同時API呼び出し数")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def turn_key(source_id: str, teacher_index: int) -> str:
    return f"{source_id}:teacher:{teacher_index}"


def completed_map(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in read_jsonl(path):
        if row.get("status") == "completed" and row.get("turn_key"):
            result[str(row["turn_key"])] = row
    return result


def teacher_turns(session: dict[str, Any]):
    teacher_index = 0
    for conversation_index, turn in enumerate(session.get("conversation", [])):
        if turn.get("role") == "teacher":
            yield teacher_index, conversation_index, turn
            teacher_index += 1


def context_payload(session: dict[str, Any], conversation_index: int) -> dict[str, Any]:
    conversation = session.get("conversation", [])
    current = conversation[conversation_index]
    latest_student = next(
        (turn for turn in reversed(conversation[:conversation_index]) if turn.get("role") == "student"),
        None,
    )
    next_student = next(
        (turn for turn in conversation[conversation_index + 1:] if turn.get("role") == "student"),
        None,
    )
    return {
        "source_id": session.get("source_id"),
        "problem": session.get("problem"),
        "student_profile": session.get("student_profile"),
        "previous_dialogue": conversation[:conversation_index],
        "latest_student_turn": latest_student,
        "teacher_turn": current,
        "next_student_turn": next_student,
        "is_final_teacher_turn": conversation_index == len(conversation) - 1,
        "session_is_completed": session.get("is_completed", False),
    }


def call_structured(
    client: OpenAI,
    model: str,
    system: str,
    payload: dict[str, Any],
    schema: dict[str, Any],
    max_retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format=schema,
                temperature=0.0,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as exc:  # API障害時に再開可能なエラー行を残す
            last_error = exc
            if getattr(exc, "status_code", None) in {401, 403}:
                raise
            if attempt + 1 < max_retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"{max_retries}回の試行に失敗しました: {last_error}") from last_error


def classify_audit(audit: dict[str, Any]) -> dict[str, Any]:
    """20点満点とハード条件からkeep / repair / rejectを決定する。"""
    result = dict(audit)
    total = sum(int(result[field]) for field in AUDIT_SCORE_FIELDS)
    if not 0 <= total <= 20:
        raise ValueError(f"監査合計点が範囲外です: {total}")
    keep_hard_conditions = (
        not result["critical_failure"]
        and not result["false_affirmation"]
        and result["mathematically_correct"]
        and result["student_answer_assessed_correctly"]
        and result["mathematical_accuracy_score"] >= 3
        and result["student_assessment_score"] >= 2
        and result["verification_completion_score"] >= 2
    )
    if not result["context_repairable"] or total <= 9:
        decision = "reject"
    elif total >= 17 and keep_hard_conditions:
        decision = "keep"
    else:
        decision = "repair"
    result["total_score"] = total
    result["decision"] = decision
    return result


def validate_audit(audit: dict[str, Any]) -> None:
    if audit["decision"] == "repair" and not audit["repair_instructions"]:
        raise ValueError("repairには修正指示が必要です。")
    if audit["decision"] == "reject" and not audit["issues"]:
        raise ValueError("rejectには問題点の記録が必要です。")


def audit_all(
    client: OpenAI,
    sessions: list[dict[str, Any]],
    args: argparse.Namespace,
    prompt: str,
) -> dict[str, dict[str, Any]]:
    done = completed_map(args.audit_output)
    pending = []
    relevant_keys = set()
    for session in sessions:
        source_id = str(session["source_id"])
        for teacher_index, conversation_index, _ in teacher_turns(session):
            key = turn_key(source_id, teacher_index)
            relevant_keys.add(key)
            if key not in done:
                pending.append((session, teacher_index, conversation_index, key))
    def run_one(item):
        session, teacher_index, conversation_index, key = item
        try:
            audit = call_structured(
                client, args.judge_model, prompt,
                context_payload(session, conversation_index), AUDIT_SCHEMA, args.max_retries,
            )
            audit = classify_audit(audit)
            validate_audit(audit)
            record = {
                "turn_key": key, "source_id": session["source_id"],
                "teacher_index": teacher_index, "status": "completed", "audit": audit,
            }
            done[key] = record
        except Exception as exc:
            record = {
                "turn_key": key, "source_id": session["source_id"],
                "teacher_index": teacher_index, "status": "error", "error": str(exc),
            }
        return key, record

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, item) for item in pending]
        for future in tqdm(as_completed(futures), total=len(futures), desc="v3 turn audit"):
            key, record = future.result()
            append_jsonl(args.audit_output, record)
            if record["status"] == "completed":
                done[key] = record
    return {key: done[key] for key in relevant_keys if key in done}


def print_audit_distribution(
    audits: dict[str, dict[str, Any]], expected_turns: int,
) -> None:
    counts = Counter(row["audit"]["decision"] for row in audits.values())
    completed = sum(counts.values())
    print("\n=== ターン監査分類（完了後に一度だけ表示） ===")
    print(f"対象: {expected_turns} / 監査成功: {completed} / 未完了: {expected_turns - completed}")
    for decision in ("keep", "repair", "reject"):
        count = counts[decision]
        ratio = count / completed * 100 if completed else 0.0
        print(f"{decision}: {count}件 ({ratio:.1f}%)")
    usable = counts["keep"] + counts["repair"]
    usable_ratio = usable / completed * 100 if completed else 0.0
    print(f"keep + repair: {usable}件 ({usable_ratio:.1f}%)")


def repair_all(
    client: OpenAI,
    sessions: list[dict[str, Any]],
    audits: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    prompt: str,
    audit_prompt: str,
) -> dict[str, dict[str, Any]]:
    done = completed_map(args.repair_output)
    pending = []
    for session in sessions:
        source_id = str(session["source_id"])
        for teacher_index, conversation_index, _ in teacher_turns(session):
            key = turn_key(source_id, teacher_index)
            audit_row = audits.get(key)
            if (
                audit_row
                and audit_row["audit"]["decision"] == "repair"
                and key not in done
            ):
                pending.append((session, teacher_index, conversation_index, key, audit_row["audit"]))
    def run_one(item):
        session, teacher_index, conversation_index, key, audit = item
        payload = context_payload(session, conversation_index)
        payload["audit"] = audit
        try:
            repair = call_structured(
                client, args.repair_model, prompt, payload, REPAIR_SCHEMA, args.max_retries,
            )
            if not all(str(repair[field]).strip() for field in (
                "thought_process", "roadmap_breakdown", "next_step_plan", "content"
            )):
                raise ValueError("修正ターンに空の必須フィールドがあります。")
            validation_payload = context_payload(session, conversation_index)
            validation_payload["teacher_turn"] = {
                "turn": validation_payload["teacher_turn"].get("turn"),
                "role": "teacher",
                "thought_process": repair["thought_process"],
                "student_emotion": repair["student_emotion"],
                "roadmap_breakdown": repair["roadmap_breakdown"],
                "next_step_plan": repair["next_step_plan"],
                "content": repair["content"],
            }
            validation_payload["session_is_completed"] = repair["is_completed"]
            validation_audit = call_structured(
                client, args.judge_model, audit_prompt,
                validation_payload, AUDIT_SCHEMA, args.max_retries,
            )
            validation_audit = classify_audit(validation_audit)
            validate_audit(validation_audit)
            if validation_audit["decision"] != "keep":
                raise ValueError(
                    "修正後の再監査に不合格: "
                    f"{validation_audit['decision']} / {validation_audit['reason']}"
                )
            record = {
                "turn_key": key, "source_id": session["source_id"],
                "teacher_index": teacher_index, "status": "completed", "repair": repair,
                "validation_audit": validation_audit,
            }
            done[key] = record
        except Exception as exc:
            record = {
                "turn_key": key, "source_id": session["source_id"],
                "teacher_index": teacher_index, "status": "error", "error": str(exc),
            }
        return key, record

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, item) for item in pending]
        for future in tqdm(as_completed(futures), total=len(futures), desc="v3 turn repair"):
            key, record = future.result()
            append_jsonl(args.repair_output, record)
            if record["status"] == "completed":
                done[key] = record
    return done


def rebuilt_session(
    session: dict[str, Any],
    audits: dict[str, dict[str, Any]],
    repairs: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str, Counter]:
    source_id = str(session["source_id"])
    rebuilt = json.loads(json.dumps(session, ensure_ascii=False))
    decisions: Counter = Counter()
    repaired_count = 0
    final_completion: bool | None = None
    for teacher_index, conversation_index, turn in teacher_turns(rebuilt):
        key = turn_key(source_id, teacher_index)
        audit_row = audits.get(key)
        if not audit_row:
            return None, f"audit_missing:{key}", decisions
        decision = audit_row["audit"]["decision"]
        turn["v3_audit"] = audit_row["audit"]
        decisions[decision] += 1
        if decision == "reject":
            return None, f"turn_rejected:{key}", decisions
        if decision == "repair":
            repair_row = repairs.get(key)
            if not repair_row:
                return None, f"repair_missing:{key}", decisions
            repair = repair_row["repair"]
            turn["v3_audit"] = repair_row["validation_audit"]
            turn.update({
                "thought_process": repair["thought_process"],
                "student_emotion": repair["student_emotion"],
                "roadmap_breakdown": repair["roadmap_breakdown"],
                "next_step_plan": repair["next_step_plan"],
                "content": repair["content"],
            })
            turn["v3_repair"] = {
                "original_audit": audit_row["audit"],
                "change_summary": repair["change_summary"],
            }
            repaired_count += 1
            if conversation_index == len(rebuilt["conversation"]) - 1:
                final_completion = bool(repair["is_completed"])
    if final_completion is not None:
        rebuilt["is_completed"] = final_completion
    rebuilt["v3_rebuild"] = {
        "source_corpus": "500_empathetic_dialogues.jsonl",
        "teacher_turns": sum(decisions.values()),
        "repaired_turns": repaired_count,
        "decisions": dict(decisions),
    }
    return rebuilt, "accepted", decisions


def write_rebuilt(
    sessions: list[dict[str, Any]],
    audits: dict[str, dict[str, Any]],
    repairs: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    accepted = []
    rejected = []
    decision_totals: Counter = Counter()
    for session in sessions:
        rebuilt, reason, decisions = rebuilt_session(session, audits, repairs)
        decision_totals.update(decisions)
        if rebuilt is None:
            rejected.append({"source_id": session.get("source_id"), "reason": reason})
        else:
            accepted.append(rebuilt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for session in accepted:
            stream.write(json.dumps(session, ensure_ascii=False) + "\n")
    manifest = {
        "dataset_name": "v3_turn_audited_rebuilt_corpus",
        "source": str(args.input),
        "judge_model": args.judge_model,
        "repair_model": args.repair_model,
        "input_dialogues": len(sessions),
        "accepted_dialogues": len(accepted),
        "rejected_dialogues": len(rejected),
        "completed_dialogues": sum(bool(row.get("is_completed")) for row in accepted),
        "turn_decisions": dict(decision_totals),
        "rejected": rejected,
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    load_env(ENV_FILE)
    if not os.getenv("GPT_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("GPT_API_KEYまたはOPENAI_API_KEYを設定してください。")
    sessions = read_jsonl(args.input)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limitには1以上を指定してください。")
        sessions = sessions[:args.limit]
    if args.workers <= 0:
        raise ValueError("--workersには1以上を指定してください。")
    for path in (JUDGE_PROMPT, REPAIR_PROMPT):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.overwrite:
        for path in (args.audit_output, args.repair_output):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
    client = OpenAI(api_key=os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY"))
    audits = audit_all(client, sessions, args, JUDGE_PROMPT.read_text(encoding="utf-8"))
    expected_turns = sum(1 for session in sessions for _ in teacher_turns(session))
    print_audit_distribution(audits, expected_turns)
    if args.audit_only:
        return
    repairs = repair_all(
        client, sessions, audits, args,
        REPAIR_PROMPT.read_text(encoding="utf-8"),
        JUDGE_PROMPT.read_text(encoding="utf-8"),
    )
    write_rebuilt(sessions, audits, repairs, args)
    print(f"v3 corpus: {args.output}")
    print(f"manifest: {args.manifest}")


if __name__ == "__main__":
    main()
