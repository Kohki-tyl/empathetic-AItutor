"""SFT方針でRepairとなった対話を同期APIで修正し、対話全体を再監査する。"""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["repair", "reaudit", "all", "summary"])
    parser.add_argument(
        "--config", type=Path, default=V4_DIR / "configs" / "pilot10.openai.json"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--retry-non-keep", action="store_true",
        help="最新の再監査でRepairの対話だけを、修正済み対話を基に再Repairする",
    )
    parser.add_argument(
        "--candidate-id", action="append", default=[],
        help="再監査対象を明示する候補ID（複数指定可）",
    )
    return parser.parse_args()


def output_paths(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["output_dir"])
    return {
        "dialogues": root / "candidate_dialogues.jsonl",
        "policy_audits": root / "dialogue_audits_sft_policy.jsonl",
        "repairs": root / "dialogue_repairs_sft_policy.jsonl",
        "repaired_dialogues": root / "repaired_dialogues_sft_policy.jsonl",
        "reaudits": root / "dialogue_reaudits_sft_policy.jsonl",
        "report": root / "REPORT_24_REPAIRS.md",
    }


def api_client() -> OpenAI:
    pipeline.load_env_file(V4_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEYを設定してください")
    return OpenAI(api_key=api_key)


def latest_completed(path: Path, key: str = "candidate_id") -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in pipeline.read_jsonl(path):
        if row.get("status") == "completed":
            rows[str(row[key])] = row
    return rows


def teacher_turn_count(dialogue: dict[str, Any]) -> int:
    return sum(turn.get("role") == "teacher" for turn in dialogue["conversation"])


def normalize_repair(
    dialogue: dict[str, Any], value: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(dialogue["candidate_id"])
    if str(value.get("candidate_id")) != candidate_id:
        raise ValueError("repair candidate_id does not match the request")
    items = value.get("repaired_teacher_turns")
    if not isinstance(items, list) or not items:
        raise ValueError("repaired_teacher_turns must contain at least one turn")
    maximum = teacher_turn_count(dialogue)
    indices: set[int] = set()
    normalized: list[dict[str, Any]] = []
    for item in items:
        index = int(item["teacher_index"])
        if index < 0 or index >= maximum or index in indices:
            raise ValueError(f"invalid or duplicate teacher_index: {index}")
        indices.add(index)
        raw_teacher = {
            key: item[key] for key in pipeline.TEACHER_PROPERTIES
        }
        if raw_teacher["is_completed"]:
            follow_up_markers = (
                "?", "？", "確認してみましょう", "考えてみましょう",
                "説明できますか", "いくつになりますか", "求めてみましょう",
            )
            if any(
                marker in str(raw_teacher["teacher_utterance"])
                for marker in follow_up_markers
            ):
                raw_teacher["is_completed"] = False
            else:
                raw_teacher["support_decision"] = {
                    "next_support": "なし", "change_reason": "なし",
                }
        teacher = pipeline.validate_teacher_turn(raw_teacher)
        normalized.append({"teacher_index": index, **teacher})
    value["repaired_teacher_turns"] = sorted(
        normalized, key=lambda item: int(item["teacher_index"])
    )
    check = str(value.get("context_consistency_check", "")).strip()
    if not check:
        raise ValueError("context_consistency_check is empty")
    value["context_consistency_check"] = check
    return value


def rebuild_dialogue(dialogue: dict[str, Any], repair: dict[str, Any]) -> dict[str, Any]:
    rebuilt = json.loads(json.dumps(dialogue, ensure_ascii=False))
    original_students = [
        turn for turn in dialogue["conversation"] if turn.get("role") == "student"
    ]
    replacements = {
        int(item["teacher_index"]): item
        for item in repair["repaired_teacher_turns"]
    }
    teacher_index = 0
    for index, turn in enumerate(rebuilt["conversation"]):
        if turn.get("role") != "teacher":
            continue
        if teacher_index in replacements:
            replacement = replacements[teacher_index]
            rebuilt["conversation"][index] = {
                "turn": turn["turn"],
                "role": "teacher",
                **{key: replacement[key] for key in pipeline.TEACHER_PROPERTIES},
                "repaired": True,
            }
        teacher_index += 1
    rebuilt_students = [
        turn for turn in rebuilt["conversation"] if turn.get("role") == "student"
    ]
    if rebuilt_students != original_students:
        raise ValueError("student turns changed during repair")
    previous_metadata = rebuilt.get("repair_metadata", {})
    previous_indices = previous_metadata.get("repaired_teacher_indices", [])
    previous_checks = previous_metadata.get("context_consistency_checks", [])
    if not previous_checks and previous_metadata.get("context_consistency_check"):
        previous_checks = [previous_metadata["context_consistency_check"]]
    rebuilt["repair_metadata"] = {
        "source_classification": "Repair",
        "repaired_teacher_indices": sorted({*previous_indices, *replacements}),
        "context_consistency_checks": [
            *previous_checks, repair["context_consistency_check"],
        ],
    }
    teacher_turns = [
        turn for turn in rebuilt["conversation"] if turn.get("role") == "teacher"
    ]
    rebuilt["is_completed"] = bool(
        teacher_turns and teacher_turns[-1].get("is_completed")
    )
    rebuilt["incomplete_reason"] = pipeline.incomplete_reason(rebuilt)
    return rebuilt


def repair(
    config: dict[str, Any], paths: dict[str, Path], workers: int, overwrite: bool,
    retry_non_keep: bool,
) -> None:
    if workers < 1:
        raise ValueError("--workersは1以上にしてください")
    if overwrite:
        paths["repairs"].unlink(missing_ok=True)
        paths["repaired_dialogues"].unlink(missing_ok=True)
        paths["reaudits"].unlink(missing_ok=True)
    source_path = paths["repaired_dialogues"] if retry_non_keep else paths["dialogues"]
    dialogues = {
        str(row["candidate_id"]): row for row in pipeline.read_jsonl(source_path)
    }
    audit_path = paths["reaudits"] if retry_non_keep else paths["policy_audits"]
    latest_audits: dict[str, dict[str, Any]] = {}
    for row in pipeline.read_jsonl(audit_path):
        if row.get("status") == "completed":
            latest_audits[str(row["candidate_id"])] = row
    policy_rows = {
        candidate_id: row for candidate_id, row in latest_audits.items()
        if row.get("classification") == "Repair"
    }
    if not retry_non_keep and len(policy_rows) != 24:
        raise RuntimeError(f"Repair対象は24件の想定ですが、{len(policy_rows)}件です")
    completed = latest_completed(paths["repairs"])
    metadata_only = {
        candidate_id for candidate_id, row in policy_rows.items()
        if row["audit"].get("issues") and all(
            "対話全体は is_completed=true" in issue
            for issue in row["audit"]["issues"]
        )
    }
    if retry_non_keep:
        pending = [candidate_id for candidate_id in policy_rows if candidate_id not in metadata_only]
    else:
        pending = [candidate_id for candidate_id in policy_rows if candidate_id not in completed]
    completed_this_run: dict[str, dict[str, Any]] = {}
    system = (V4_DIR / "prompts" / "dialogue_repair_system.txt").read_text(encoding="utf-8")
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
            "dialogue_audit": policy_rows[candidate_id],
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        last_error: Exception | None = None
        for validation_attempt in range(3):
            value = pipeline.chat_call(
                client, config["repair_model"], messages,
                pipeline.REPAIR_SCHEMA,
                reasoning_effort=config.get("repair_reasoning_effort"),
                max_completion_tokens=7000,
            )
            try:
                return candidate_id, normalize_repair(dialogue, value)
            except (KeyError, TypeError, ValueError) as exc:
                last_error = exc
                messages.extend([
                    {"role": "assistant", "content": json.dumps(value, ensure_ascii=False)},
                    {
                        "role": "user",
                        "content": (
                            f"ローカル検証エラー: {exc}。内容と文脈を保ちつつ、"
                            "全フィールドを検証条件に合わせて修正し、JSON全体を再出力してください。"
                        ),
                    },
                ])
        raise RuntimeError(f"repair validation failed after 3 attempts: {last_error}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(repair_one, cid): cid for cid in pending}
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="repair-sft-policy",
        ):
            candidate_id = futures[future]
            try:
                _, value = future.result()
                row = {"candidate_id": candidate_id, "status": "completed", "repair": value}
                completed[candidate_id] = row
                completed_this_run[candidate_id] = row
            except Exception as exc:
                row = {
                    "candidate_id": candidate_id, "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            pipeline.append_jsonl(paths["repairs"], row)

    if retry_non_keep:
        repaired = []
        for candidate_id in sorted(dialogues):
            dialogue = dialogues[candidate_id]
            if candidate_id in completed_this_run:
                dialogue = rebuild_dialogue(
                    dialogue, completed_this_run[candidate_id]["repair"],
                )
            else:
                dialogue = json.loads(json.dumps(dialogue, ensure_ascii=False))
                teacher_turns = [
                    turn for turn in dialogue["conversation"] if turn.get("role") == "teacher"
                ]
                dialogue["is_completed"] = bool(
                    teacher_turns and teacher_turns[-1].get("is_completed")
                )
                dialogue["incomplete_reason"] = pipeline.incomplete_reason(dialogue)
            repaired.append(dialogue)
    else:
        repaired = [
            rebuild_dialogue(dialogues[candidate_id], completed[candidate_id]["repair"])
            for candidate_id in sorted(policy_rows)
            if candidate_id in completed
        ]
    pipeline.write_jsonl(paths["repaired_dialogues"], repaired)
    print(
        f"repaired dialogues: {len(repaired)}; "
        f"API repaired this run: {len(completed_this_run)}; metadata-only: {len(metadata_only)}"
    )


def reaudit(
    config: dict[str, Any], paths: dict[str, Path], workers: int, overwrite: bool,
    retry_non_keep: bool, candidate_ids: list[str],
) -> None:
    if workers < 1:
        raise ValueError("--workersは1以上にしてください")
    if overwrite:
        paths["reaudits"].unlink(missing_ok=True)
    dialogues = pipeline.read_jsonl(paths["repaired_dialogues"])
    if len(dialogues) != 24:
        raise RuntimeError(f"修正済み対話は24件必要ですが、{len(dialogues)}件です")
    completed = latest_completed(paths["reaudits"])
    if candidate_ids:
        pending_ids = set(candidate_ids)
        unknown = pending_ids - {str(row["candidate_id"]) for row in dialogues}
        if unknown:
            raise ValueError(f"修正済み対話に存在しないcandidate_id: {sorted(unknown)}")
        pending = [row for row in dialogues if str(row["candidate_id"]) in pending_ids]
    elif retry_non_keep:
        pending_ids = {
            candidate_id for candidate_id, row in completed.items()
            if row.get("classification") != "Keep"
        }
        pending = [row for row in dialogues if str(row["candidate_id"]) in pending_ids]
    else:
        pending = [row for row in dialogues if str(row["candidate_id"]) not in completed]
    system = (V4_DIR / "prompts" / "dialogue_quality_judge_system.txt").read_text(encoding="utf-8")
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
            concurrent.futures.as_completed(futures), total=len(futures), desc="reaudit-sft-policy",
        ):
            dialogue = futures[future]
            candidate_id = str(dialogue["candidate_id"])
            try:
                _, value = future.result()
                row = {
                    "candidate_id": candidate_id, "status": "completed",
                    "classification": pipeline.classify_dialogue_audit(value),
                    "total_score": sum(int(value[name]) for name in pipeline.SCORE_FIELDS),
                    "audit": value,
                }
                completed[candidate_id] = row
            except Exception as exc:
                row = {
                    "candidate_id": candidate_id, "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            pipeline.append_jsonl(paths["reaudits"], row)
    print(f"reaudited dialogues: {len(completed)} / {len(dialogues)}")


def write_summary(paths: dict[str, Path]) -> None:
    repairs = latest_completed(paths["repairs"])
    reaudits = latest_completed(paths["reaudits"])
    repaired_dialogues = pipeline.read_jsonl(paths["repaired_dialogues"])
    counts = Counter(row["classification"] for row in reaudits.values())
    repaired_turns = sum(
        len(row.get("repair_metadata", {}).get("repaired_teacher_indices", []))
        for row in repaired_dialogues
    )
    average_total = (
        sum(int(row["total_score"]) for row in reaudits.values()) / len(reaudits)
        if reaudits else 0
    )
    field_averages = {
        field: (
            sum(int(row["audit"][field]) for row in reaudits.values()) / len(reaudits)
            if reaudits else 0
        )
        for field in pipeline.SCORE_FIELDS
    }
    lines = [
        "# v4 SFT方針 Repair実行結果", "",
        f"- Repair対象: 24件",
        f"- Repair完了: {len(repairs)}件",
        f"- 置換した教師ターン: {repaired_turns}件",
        f"- 対話全体の再監査完了: {len(reaudits)}件", "",
        "## 再監査分類", "",
        f"- Keep: {counts.get('Keep', 0)}件",
        f"- Repair: {counts.get('Repair', 0)}件",
        f"- Reject: {counts.get('Reject', 0)}件", "",
        "## 再監査スコア", "",
        f"- 6項目合計の平均: {average_total:.2f} / 60",
        *[
            f"- {field}: {field_averages[field]:.2f} / 10"
            for field in pipeline.SCORE_FIELDS
        ],
        "",
    ]
    remaining = [row for row in reaudits.values() if row["classification"] != "Keep"]
    if remaining:
        lines.extend(["## 未採択の主な理由", ""])
        for row in sorted(remaining, key=lambda value: value["candidate_id"]):
            reasons = row["audit"].get("issues") or [row["audit"].get("reason", "理由なし")]
            lines.append(f"- {row['candidate_id']}（{row['classification']}）: {' / '.join(reasons)}")
        lines.append("")
    paths["report"].write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "repair_completed": len(repairs), "repaired_teacher_turns": repaired_turns,
        "reaudit_completed": len(reaudits), "classification_counts": counts,
        "average_total_score": round(average_total, 2),
        "field_averages": {
            field: round(value, 2) for field, value in field_averages.items()
        },
    }, ensure_ascii=False, indent=2, default=dict))


def main() -> None:
    args = parse_args()
    config = pipeline.load_config(args.config.resolve())
    paths = output_paths(config)
    if args.command in {"repair", "all"}:
        repair(config, paths, args.workers, args.overwrite, args.retry_non_keep)
    if args.command in {"reaudit", "all"}:
        reaudit(
            config, paths, args.workers, args.overwrite, args.retry_non_keep,
            args.candidate_id,
        )
    if args.command in {"summary", "all"}:
        write_summary(paths)


if __name__ == "__main__":
    main()
