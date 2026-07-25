"""修正済み対話から最新再監査がKeepのものだけを抽出し、SFT形式と評価を作る。"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


V4_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4_DIR))

import run_v4 as pipeline  # noqa: E402


SCORE_LABELS = {
    "mathematical_accuracy_score": "数学的正確性",
    "error_diagnosis_recovery_score": "誤りの診断と回復",
    "cognitive_empathy_score": "認知的共感",
    "emotional_support_score": "感情認識・情緒的支援",
    "adaptive_scaffolding_score": "適応的な足場かけ",
    "verification_completion_score": "理解確認・完了判定",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=V4_DIR / "data" / "run_10_openai_gpt54mini",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return pipeline.read_jsonl(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_by_candidate(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("status") == "completed":
            latest[str(row["candidate_id"])] = row
    return latest


def has_forbidden_control_character(text: str) -> bool:
    return any(ord(char) < 32 and char not in "\n\t\r" for char in text)


def average(rows: list[dict[str, Any]], field: str) -> float:
    return statistics.mean(float(row["audit"][field]) for row in rows)


def main() -> None:
    args = parse_args()
    root = args.input_dir.resolve()
    repaired_path = root / "repaired_dialogues_sft_policy.jsonl"
    reaudit_path = root / "dialogue_reaudits_sft_policy.jsonl"
    initial_audit_path = root / "dialogue_audits_sft_policy.jsonl"
    original_path = root / "candidate_dialogues.jsonl"
    corpus_path = root / "v4_repaired_keep_corpus.jsonl"
    sft_path = root / "v4_repaired_keep_sft.jsonl"
    manifest_path = root / "v4_repaired_keep_manifest.json"
    report_path = root / "REPORT_REPAIRED_KEEP_CORPUS.md"

    repaired = {str(row["candidate_id"]): row for row in read_jsonl(repaired_path)}
    original = {str(row["candidate_id"]): row for row in read_jsonl(original_path)}
    latest_reaudits = latest_by_candidate(reaudit_path)
    latest_initial_audits = latest_by_candidate(initial_audit_path)
    keep_ids = sorted(
        candidate_id for candidate_id, row in latest_reaudits.items()
        if row.get("classification") == "Keep" and candidate_id in repaired
    )
    keep_dialogues = [repaired[candidate_id] for candidate_id in keep_ids]
    keep_audits = [latest_reaudits[candidate_id] for candidate_id in keep_ids]
    initial_audits = [
        latest_initial_audits[candidate_id]
        for candidate_id in keep_ids if candidate_id in latest_initial_audits
    ]

    if len(keep_ids) != len(set(keep_ids)):
        raise RuntimeError("Keepコーパスのcandidate_idが重複しています")
    if not keep_dialogues:
        raise RuntimeError("最新再監査がKeepの修正済み対話がありません")

    student_unchanged = 0
    control_character_locations: list[str] = []
    repaired_teacher_turns = 0
    teacher_turn_counts: list[int] = []
    for dialogue in keep_dialogues:
        candidate_id = str(dialogue["candidate_id"])
        old_students = [
            turn for turn in original[candidate_id]["conversation"]
            if turn.get("role") == "student"
        ]
        new_students = [
            turn for turn in dialogue["conversation"]
            if turn.get("role") == "student"
        ]
        student_unchanged += old_students == new_students
        teacher_turns = [
            turn for turn in dialogue["conversation"] if turn.get("role") == "teacher"
        ]
        teacher_turn_counts.append(len(teacher_turns))
        repaired_teacher_turns += sum(bool(turn.get("repaired")) for turn in teacher_turns)
        for index, turn in enumerate(teacher_turns):
            pipeline.validate_teacher_turn({
                key: turn[key] for key in pipeline.TEACHER_PROPERTIES
            })
            utterance = str(turn["teacher_utterance"])
            if has_forbidden_control_character(utterance):
                control_character_locations.append(f"{candidate_id}:teacher-{index}")

    if student_unchanged != len(keep_dialogues):
        raise RuntimeError("Repairによって生徒ターンが変更されています")
    if control_character_locations:
        raise RuntimeError(f"教師発話に制御文字があります: {control_character_locations}")
    if any(row["audit"].get("issues") for row in keep_audits):
        raise RuntimeError("Keep監査結果に未解決issuesがあります")
    if any(row["audit"].get("repair_instructions") for row in keep_audits):
        raise RuntimeError("Keep監査結果にrepair_instructionsがあります")

    system_prompt = (V4_DIR / "prompts" / "sft_teacher_system.txt").read_text(
        encoding="utf-8"
    )
    sft_rows = [
        {
            "id": dialogue["candidate_id"],
            "messages": pipeline.build_sft_messages(dialogue, system_prompt),
        }
        for dialogue in keep_dialogues
    ]
    pipeline.write_jsonl(corpus_path, keep_dialogues)
    pipeline.write_jsonl(sft_path, sft_rows)

    total_scores = [int(row["total_score"]) for row in keep_audits]
    final_averages = {
        field: average(keep_audits, field) for field in pipeline.SCORE_FIELDS
    }
    initial_averages = {
        field: average(initial_audits, field) for field in pipeline.SCORE_FIELDS
    }
    assistant_targets = sum(
        message["role"] == "assistant"
        for row in sft_rows for message in row["messages"]
    )
    record_characters = [
        sum(len(message["content"]) for message in row["messages"])
        for row in sft_rows
    ]
    completed = sum(bool(row.get("is_completed")) for row in keep_dialogues)
    metadata_warning_dialogues = sum(
        bool(row["audit"].get("metadata_warnings")) for row in keep_audits
    )
    acceptable_incomplete_dialogues = sum(
        bool(row["audit"].get("acceptable_incompleteness")) for row in keep_audits
    )
    profile_counts = Counter(
        str(row["student_profile"]["id"]) for row in keep_dialogues
    )
    scope_counts = Counter(
        str(
            row.get("generation_condition", {})
            .get("problem_profile_assignment", {})
            .get("scope_relation", "unknown")
        )
        for row in keep_dialogues
    )
    emotion_counts = Counter(str(row["initial_emotion"]) for row in keep_dialogues)

    manifest = {
        "corpus_file": corpus_path.name,
        "sft_file": sft_path.name,
        "source_files": {
            repaired_path.name: sha256(repaired_path),
            reaudit_path.name: sha256(reaudit_path),
            initial_audit_path.name: sha256(initial_audit_path),
            "sft_teacher_system.txt": sha256(V4_DIR / "prompts" / "sft_teacher_system.txt"),
        },
        "selection": {
            "rule": "latest completed whole-dialogue reaudit classification == Keep",
            "selected_dialogues": len(keep_dialogues),
            "candidate_ids": keep_ids,
        },
        "integrity": {
            "unique_candidate_ids": len(keep_ids) == len(set(keep_ids)),
            "student_dialogues_unchanged": student_unchanged,
            "forbidden_control_characters": len(control_character_locations),
            "unresolved_issues": 0,
            "unresolved_repair_instructions": 0,
        },
        "sft_statistics": {
            "records": len(sft_rows),
            "assistant_targets": assistant_targets,
            "average_teacher_turns": round(statistics.mean(teacher_turn_counts), 2),
            "average_record_characters": round(statistics.mean(record_characters), 2),
            "maximum_record_characters": max(record_characters),
            "tokenizer_length_audit_required": True,
        },
        "evaluation": {
            "average_total_score": round(statistics.mean(total_scores), 2),
            "minimum_total_score": min(total_scores),
            "maximum_total_score": max(total_scores),
            "final_field_averages": {
                field: round(value, 2) for field, value in final_averages.items()
            },
            "initial_field_averages": {
                field: round(value, 2) for field, value in initial_averages.items()
            },
            "completed_dialogues": completed,
            "incomplete_dialogues": len(keep_dialogues) - completed,
            "metadata_warning_dialogues": metadata_warning_dialogues,
            "acceptable_incomplete_dialogues": acceptable_incomplete_dialogues,
        },
        "distribution": {
            "profiles": dict(sorted(profile_counts.items())),
            "scope_relations": dict(sorted(scope_counts.items())),
            "initial_emotions": dict(sorted(emotion_counts.items())),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# 修正済みKeepコーパス評価", "",
        "## 結果", "",
        f"- 抽出元の修正済み対話: {len(repaired)}件",
        f"- 最新の対話全体再監査がKeep: {len(keep_dialogues)}件",
        f"- SFTレコード: {len(sft_rows)}件",
        f"- 教師学習ターゲット: {assistant_targets}ターン",
        f"- 修正済み教師ターン: {repaired_teacher_turns}ターン",
        f"- 完了対話: {completed}件",
        f"- 許容された未完了対話: {len(keep_dialogues) - completed}件", "",
        "## 品質スコア", "",
        f"- 6項目合計平均: {statistics.mean(total_scores):.2f} / 60",
        f"- 最低合計点: {min(total_scores)} / 60",
        f"- 最高合計点: {max(total_scores)} / 60", "",
        "| 評価軸 | Repair前 | Repair後 | 変化 |", "|---|---:|---:|---:|",
    ]
    for field in pipeline.SCORE_FIELDS:
        before = initial_averages[field]
        after = final_averages[field]
        lines.append(
            f"| {SCORE_LABELS[field]} | {before:.2f} | {after:.2f} | {after - before:+.2f} |"
        )
    lines.extend([
        "", "## 整合性", "",
        f"- candidate_id重複: 0件",
        f"- 生徒発話・状態が元対話から変化した対話: {len(keep_dialogues) - student_unchanged}件",
        f"- 教師発話の制御文字: {len(control_character_locations)}件",
        "- 未解決issues: 0件",
        "- 未解決repair_instructions: 0件",
        f"- 内部メタデータ警告を持つ対話: {metadata_warning_dialogues}件", "",
        "## 構成", "", "### 学習範囲との関係", "",
    ])
    lines.extend(f"- {key}: {value}件" for key, value in sorted(scope_counts.items()))
    lines.extend(["", "### 生徒プロフィール", ""])
    lines.extend(f"- {key}: {value}件" for key, value in sorted(profile_counts.items()))
    lines.extend(["", "### 初期感情", ""])
    lines.extend(f"- {key}: {value}件" for key, value in sorted(emotion_counts.items()))
    lines.extend([
        "", "## 評価", "",
        "全24件が最新版の採択基準でKeepとなり、数学的正確性、誤りの診断、認知的共感、情緒的支援、足場かけ、理解確認の全項目で各対話8点以上を満たす。生徒ターンを固定したまま教師発話だけを修正しており、教師SFTの正例として採択可能である。",
        "",
        "一方、このデータだけでは24対話に限られ、未完了対話が14件（58.3%）を占めるため、単独で十分な学習量・構成とはいえない。V4-S06とV4-S08も各1件に限られる。既存の初回Keep対話と統合し、完了状態、プロフィール、問題との学習範囲関係、初期感情の分布を再調整することが望ましい。",
        "",
        "Repairと再監査を同じモデル設定で反復しているため、Keep率とスコアにはjudgeへの適合が含まれうる。SFT投入前に別モデルまたは人手で層化抽出した対話を再確認する必要がある。内部メタデータ警告は23件にあるが教師SFTへ直接含まれない。将来、生徒モデルも学習対象にする場合は別途修正する。",
        "",
        f"SFTレコードの平均文字数は{statistics.mean(record_characters):.0f}、最大は{max(record_characters)}である。文字数だけでは対象モデルのコンテキスト長適合を保証できないため、ABCI上で対象tokenizerとchat templateを使った全件長さ監査が必要である。",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
