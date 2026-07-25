"""全Keepコーパスの表面制御文字・空発話を修正し、対話全体を再監査する。"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


V4_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4_DIR))

import run_v4 as pipeline  # noqa: E402
import repair_sft_policy_dialogues as contextual  # noqa: E402


SURFACE_REPAIR_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "v4_surface_text_repair",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string"},
                "replacements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "conversation_index": {"type": "integer", "minimum": 0},
                            "role": {"type": "string", "enum": ["student", "teacher"]},
                            "repaired_text": {"type": "string"},
                        },
                        "required": ["conversation_index", "role", "repaired_text"],
                        "additionalProperties": False,
                    },
                },
                "context_consistency_check": {"type": "string"},
            },
            "required": ["candidate_id", "replacements", "context_consistency_check"],
            "additionalProperties": False,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "repair", "reaudit", "repair-context", "reaudit-context", "all", "summary",
        ],
    )
    parser.add_argument(
        "--config", type=Path, default=V4_DIR / "configs" / "pilot10.openai.json"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def paths(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["output_dir"])
    return {
        "source": root / "v4_all_keep_corpus.jsonl",
        "repairs": root / "surface_text_repairs.jsonl",
        "repaired": root / "surface_repaired_dialogues.jsonl",
        "reaudits": root / "surface_reaudits.jsonl",
        "context_repairs": root / "surface_context_repairs.jsonl",
        "context_repaired": root / "surface_context_repaired_dialogues.jsonl",
        "context_reaudits": root / "surface_context_reaudits.jsonl",
        "final_dialogues": root / "surface_final_dialogues.jsonl",
        "final_audits": root / "surface_final_audits.jsonl",
        "report": root / "REPORT_SURFACE_REPAIRS.md",
    }


def api_client() -> OpenAI:
    pipeline.load_env_file(V4_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEYを設定してください")
    return OpenAI(api_key=api_key)


def has_control(text: str) -> bool:
    return any(ord(char) < 32 and char not in "\n\t\r" for char in text)


def visible_text(text: str) -> str:
    return "".join(
        char for char in text if ord(char) >= 32 or char in "\n\t\r"
    ).strip()


def target_text(turn: dict[str, Any]) -> str:
    return str(turn.get("content", "")) if turn.get("role") == "student" else str(
        turn.get("teacher_utterance", "")
    )


def repair_targets(dialogue: dict[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for index, turn in enumerate(dialogue["conversation"]):
        text = target_text(turn)
        reasons = []
        if has_control(text):
            reasons.append("制御文字を含む")
        if turn.get("role") == "student" and not visible_text(text):
            reasons.append("表示可能な発話内容がない")
        if reasons:
            targets.append({
                "conversation_index": index,
                "role": turn["role"],
                "broken_text": text,
                "reasons": reasons,
            })
    return targets


def latest_completed(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in pipeline.read_jsonl(path):
        if row.get("status") == "completed":
            latest[str(row["candidate_id"])] = row
    return latest


def normalize_repair(
    dialogue: dict[str, Any], targets: list[dict[str, Any]], value: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(dialogue["candidate_id"])
    if str(value.get("candidate_id")) != candidate_id:
        raise ValueError("candidate_id does not match")
    expected = {
        (int(row["conversation_index"]), str(row["role"])) for row in targets
    }
    replacements = value.get("replacements")
    if not isinstance(replacements, list) or len(replacements) != len(expected):
        raise ValueError("replacement count does not match repair targets")
    actual = {
        (int(row["conversation_index"]), str(row["role"])) for row in replacements
    }
    if actual != expected or len(actual) != len(replacements):
        raise ValueError(f"replacement targets mismatch: expected={expected}, actual={actual}")
    normalized = []
    for row in replacements:
        index = int(row["conversation_index"])
        role = str(row["role"])
        text = str(row["repaired_text"]).strip()
        if not text or has_control(text):
            raise ValueError(f"repaired text is empty or contains controls: {index}")
        if len(text) > 2000 or text.startswith(("{", "[")):
            raise ValueError(f"repaired text is invalid: {index}")
        lower = text.lower()
        if any(marker in lower for marker in ("<analysis>", "<final>", "state_after")):
            raise ValueError(f"repaired text leaks internal state: {index}")
        if role == "student" and len(text) > 800:
            raise ValueError(f"student repaired text is too long: {index}")
        normalized.append({
            "conversation_index": index, "role": role, "repaired_text": text,
        })
    check = str(value.get("context_consistency_check", "")).strip()
    if not check:
        raise ValueError("context_consistency_check is empty")
    return {
        "candidate_id": candidate_id,
        "replacements": sorted(normalized, key=lambda row: row["conversation_index"]),
        "context_consistency_check": check,
    }


def rebuild(dialogue: dict[str, Any], repair: dict[str, Any]) -> dict[str, Any]:
    rebuilt = json.loads(json.dumps(dialogue, ensure_ascii=False))
    replacements = {
        int(row["conversation_index"]): row for row in repair["replacements"]
    }
    for index, replacement in replacements.items():
        turn = rebuilt["conversation"][index]
        if turn.get("role") != replacement["role"]:
            raise ValueError(f"role changed at conversation index {index}")
        if turn["role"] == "student":
            turn["content"] = replacement["repaired_text"]
            turn["surface_repaired"] = True
        else:
            turn["teacher_utterance"] = replacement["repaired_text"]
            turn["surface_repaired"] = True
            pipeline.validate_teacher_turn({
                key: turn[key] for key in pipeline.TEACHER_PROPERTIES
            })
    remaining = repair_targets(rebuilt)
    if remaining:
        raise ValueError(f"surface corruption remains: {remaining}")
    rebuilt["surface_repair_metadata"] = {
        "repaired_conversation_indices": sorted(replacements),
        "context_consistency_check": repair["context_consistency_check"],
    }
    return rebuilt


def repair(
    config: dict[str, Any], file_paths: dict[str, Path], workers: int, overwrite: bool,
) -> None:
    if workers < 1:
        raise ValueError("--workersは1以上にしてください")
    if overwrite:
        file_paths["repairs"].unlink(missing_ok=True)
        file_paths["repaired"].unlink(missing_ok=True)
        file_paths["reaudits"].unlink(missing_ok=True)
    dialogues = {
        str(row["candidate_id"]): row for row in pipeline.read_jsonl(file_paths["source"])
        if repair_targets(row)
    }
    if len(dialogues) != 18:
        raise RuntimeError(f"表面修正対象は18件の想定ですが、{len(dialogues)}件です")
    completed = latest_completed(file_paths["repairs"])
    pending = [candidate_id for candidate_id in dialogues if candidate_id not in completed]
    system = (V4_DIR / "prompts" / "surface_text_repair_system.txt").read_text(
        encoding="utf-8"
    )
    client = api_client()

    def repair_one(candidate_id: str) -> tuple[str, dict[str, Any]]:
        dialogue = dialogues[candidate_id]
        targets = repair_targets(dialogue)
        payload = {
            "candidate_id": candidate_id,
            "problem": dialogue["problem"],
            "reference_solution": dialogue["reference_solution"],
            "student_profile": dialogue["student_profile"],
            "initial_emotion": dialogue["initial_emotion"],
            "immutable_conversation": dialogue["conversation"],
            "repair_targets": targets,
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        last_error: Exception | None = None
        for _ in range(3):
            value = pipeline.chat_call(
                client, config["repair_model"], messages, SURFACE_REPAIR_SCHEMA,
                reasoning_effort=config.get("repair_reasoning_effort"),
                max_completion_tokens=5000,
            )
            try:
                return candidate_id, normalize_repair(dialogue, targets, value)
            except (KeyError, TypeError, ValueError) as exc:
                last_error = exc
                messages.extend([
                    {"role": "assistant", "content": json.dumps(value, ensure_ascii=False)},
                    {"role": "user", "content": f"検証エラー: {exc}。全対象を修正して再出力してください。"},
                ])
        raise RuntimeError(f"surface repair validation failed: {last_error}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(repair_one, cid): cid for cid in pending}
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures),
            desc="surface-repair",
        ):
            candidate_id = futures[future]
            try:
                _, value = future.result()
                row = {"candidate_id": candidate_id, "status": "completed", "repair": value}
                completed[candidate_id] = row
            except Exception as exc:
                row = {
                    "candidate_id": candidate_id, "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            pipeline.append_jsonl(file_paths["repairs"], row)
    repaired = [
        rebuild(dialogues[candidate_id], completed[candidate_id]["repair"])
        for candidate_id in sorted(dialogues) if candidate_id in completed
    ]
    pipeline.write_jsonl(file_paths["repaired"], repaired)
    print(f"surface repaired: {len(repaired)} / {len(dialogues)}")


def reaudit(
    config: dict[str, Any], file_paths: dict[str, Path], workers: int, overwrite: bool,
) -> None:
    if workers < 1:
        raise ValueError("--workersは1以上にしてください")
    if overwrite:
        file_paths["reaudits"].unlink(missing_ok=True)
    dialogues = pipeline.read_jsonl(file_paths["repaired"])
    completed = latest_completed(file_paths["reaudits"])
    pending = [row for row in dialogues if str(row["candidate_id"]) not in completed]
    system = (V4_DIR / "prompts" / "dialogue_quality_judge_system.txt").read_text(
        encoding="utf-8"
    )
    client = api_client()

    def audit_one(dialogue: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        value = pipeline.chat_call(
            client, config["judge_model"],
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(dialogue, ensure_ascii=False)},
            ],
            pipeline.DIALOGUE_AUDIT_SCHEMA,
            reasoning_effort=config.get("judge_reasoning_effort"),
            max_completion_tokens=5000,
        )
        return str(dialogue["candidate_id"]), value

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(audit_one, row): row for row in pending}
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures),
            desc="surface-reaudit",
        ):
            dialogue = futures[future]
            candidate_id = str(dialogue["candidate_id"])
            try:
                _, value = future.result()
                row = {
                    "candidate_id": candidate_id, "status": "completed",
                    "classification": pipeline.classify_dialogue_audit(value),
                    "total_score": sum(int(value[field]) for field in pipeline.SCORE_FIELDS),
                    "audit": value,
                }
                completed[candidate_id] = row
            except Exception as exc:
                row = {
                    "candidate_id": candidate_id, "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            pipeline.append_jsonl(file_paths["reaudits"], row)
    print(f"surface reaudited: {len(completed)} / {len(dialogues)}")


def repair_context(
    config: dict[str, Any], file_paths: dict[str, Path], workers: int, overwrite: bool,
) -> None:
    if workers < 1:
        raise ValueError("--workersは1以上にしてください")
    if overwrite:
        file_paths["context_repairs"].unlink(missing_ok=True)
        file_paths["context_repaired"].unlink(missing_ok=True)
        file_paths["context_reaudits"].unlink(missing_ok=True)
    dialogues = {
        str(row["candidate_id"]): row
        for row in pipeline.read_jsonl(file_paths["repaired"])
    }
    audits = {
        candidate_id: row for candidate_id, row in latest_completed(file_paths["reaudits"]).items()
        if row.get("classification") == "Repair"
    }
    completed = latest_completed(file_paths["context_repairs"])
    pending = [candidate_id for candidate_id in audits if candidate_id not in completed]
    system = (V4_DIR / "prompts" / "dialogue_repair_system.txt").read_text(
        encoding="utf-8"
    )
    client = api_client()

    def repair_one(candidate_id: str) -> tuple[str, dict[str, Any]]:
        dialogue = dialogues[candidate_id]
        payload = {
            "candidate_id": candidate_id,
            "problem": dialogue["problem"],
            "reference_solution": dialogue["reference_solution"],
            "student_profile": dialogue["student_profile"],
            "initial_emotion": dialogue["initial_emotion"],
            "generation_condition": dialogue.get("generation_condition", {}),
            "immutable_conversation": dialogue["conversation"],
            "dialogue_audit": audits[candidate_id],
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        last_error: Exception | None = None
        for _ in range(3):
            value = pipeline.chat_call(
                client, config["repair_model"], messages, pipeline.REPAIR_SCHEMA,
                reasoning_effort=config.get("repair_reasoning_effort"),
                max_completion_tokens=7000,
            )
            try:
                return candidate_id, contextual.normalize_repair(dialogue, value)
            except (KeyError, TypeError, ValueError) as exc:
                last_error = exc
                messages.extend([
                    {"role": "assistant", "content": json.dumps(value, ensure_ascii=False)},
                    {"role": "user", "content": f"検証エラー: {exc}。修正して再出力してください。"},
                ])
        raise RuntimeError(f"context repair validation failed: {last_error}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(repair_one, candidate_id): candidate_id for candidate_id in pending}
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures),
            desc="surface-context-repair",
        ):
            candidate_id = futures[future]
            try:
                _, value = future.result()
                row = {"candidate_id": candidate_id, "status": "completed", "repair": value}
                completed[candidate_id] = row
            except Exception as exc:
                row = {
                    "candidate_id": candidate_id, "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            pipeline.append_jsonl(file_paths["context_repairs"], row)
    repaired = [
        contextual.rebuild_dialogue(dialogues[candidate_id], completed[candidate_id]["repair"])
        for candidate_id in sorted(audits) if candidate_id in completed
    ]
    pipeline.write_jsonl(file_paths["context_repaired"], repaired)
    print(f"surface context repaired: {len(repaired)} / {len(audits)}")


def reaudit_context(
    config: dict[str, Any], file_paths: dict[str, Path], workers: int, overwrite: bool,
) -> None:
    if workers < 1:
        raise ValueError("--workersは1以上にしてください")
    if overwrite:
        file_paths["context_reaudits"].unlink(missing_ok=True)
    dialogues = pipeline.read_jsonl(file_paths["context_repaired"])
    completed = latest_completed(file_paths["context_reaudits"])
    pending = [row for row in dialogues if str(row["candidate_id"]) not in completed]
    system = (V4_DIR / "prompts" / "dialogue_quality_judge_system.txt").read_text(
        encoding="utf-8"
    )
    client = api_client()

    def audit_one(dialogue: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        value = pipeline.chat_call(
            client, config["judge_model"],
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(dialogue, ensure_ascii=False)},
            ],
            pipeline.DIALOGUE_AUDIT_SCHEMA,
            reasoning_effort=config.get("judge_reasoning_effort"),
            max_completion_tokens=5000,
        )
        return str(dialogue["candidate_id"]), value

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(audit_one, row): row for row in pending}
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures),
            desc="surface-context-reaudit",
        ):
            dialogue = futures[future]
            candidate_id = str(dialogue["candidate_id"])
            try:
                _, value = future.result()
                row = {
                    "candidate_id": candidate_id, "status": "completed",
                    "classification": pipeline.classify_dialogue_audit(value),
                    "total_score": sum(int(value[field]) for field in pipeline.SCORE_FIELDS),
                    "audit": value,
                }
                completed[candidate_id] = row
            except Exception as exc:
                row = {
                    "candidate_id": candidate_id, "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            pipeline.append_jsonl(file_paths["context_reaudits"], row)
    print(f"surface context reaudited: {len(completed)} / {len(dialogues)}")


def summary(file_paths: dict[str, Path]) -> None:
    repairs = latest_completed(file_paths["repairs"])
    base_dialogues = {
        str(row["candidate_id"]): row for row in pipeline.read_jsonl(file_paths["repaired"])
    }
    context_dialogues = {
        str(row["candidate_id"]): row
        for row in pipeline.read_jsonl(file_paths["context_repaired"])
    }
    audits = latest_completed(file_paths["reaudits"])
    context_audits = latest_completed(file_paths["context_reaudits"])
    audits.update(context_audits)
    base_dialogues.update(context_dialogues)
    pipeline.write_jsonl(
        file_paths["final_dialogues"],
        [base_dialogues[candidate_id] for candidate_id in sorted(base_dialogues)],
    )
    pipeline.write_jsonl(
        file_paths["final_audits"],
        [audits[candidate_id] for candidate_id in sorted(audits)],
    )
    counts = Counter(row["classification"] for row in audits.values())
    lines = [
        "# 表面破損18対話の修正・再監査", "",
        f"- 修正対象: 18件",
        f"- 表面修正完了: {len(repairs)}件",
        f"- 全体再監査完了: {len(audits)}件",
        f"- Keep: {counts.get('Keep', 0)}件",
        f"- Repair: {counts.get('Repair', 0)}件",
        f"- Reject: {counts.get('Reject', 0)}件", "",
    ]
    remaining = [row for row in audits.values() if row["classification"] != "Keep"]
    if remaining:
        lines.extend(["## 追加保留", ""])
        for row in sorted(remaining, key=lambda item: item["candidate_id"]):
            audit = row["audit"]
            issues = list(audit.get("issues") or [])
            if not issues:
                required_true = (
                    "mathematically_correct", "student_answer_assessed_correctly",
                    "cognitive_state_grounded", "emotion_grounded",
                    "analysis_reflected_in_utterance", "student_profile_consistent",
                    "student_role_consistent", "student_state_update_plausible",
                    "initial_emotion_utterance_consistent", "completion_decision_appropriate",
                )
                failed = [name for name in required_true if not bool(audit.get(name))]
                low_scores = [
                    f"{field}={audit[field]}" for field in pipeline.SCORE_FIELDS
                    if int(audit[field]) < 8
                ]
                flags = [
                    name for name in ("false_affirmation", "direct_answer_without_need", "critical_failure")
                    if bool(audit.get(name))
                ]
                details = [*failed, *low_scores, *flags]
                issues = [
                    "必須条件を満たさない: " + ", ".join(details)
                    if details else str(audit.get("reason", "理由なし"))
                ]
            lines.append(
                f"- {row['candidate_id']}（{row['classification']}）: {' / '.join(issues)}"
            )
        lines.append("")
    file_paths["report"].write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "repair_completed": len(repairs), "reaudit_completed": len(audits),
        "classification_counts": counts,
    }, ensure_ascii=False, indent=2, default=dict))


def main() -> None:
    args = parse_args()
    config = pipeline.load_config(args.config.resolve())
    file_paths = paths(config)
    if args.command in {"repair", "all"}:
        repair(config, file_paths, args.workers, args.overwrite)
    if args.command in {"reaudit", "all"}:
        reaudit(config, file_paths, args.workers, args.overwrite)
    if args.command in {"repair-context", "all"}:
        repair_context(config, file_paths, args.workers, args.overwrite)
    if args.command in {"reaudit-context", "all"}:
        reaudit_context(config, file_paths, args.workers, args.overwrite)
    if args.command in {"summary", "all"}:
        summary(file_paths)


if __name__ == "__main__":
    main()
