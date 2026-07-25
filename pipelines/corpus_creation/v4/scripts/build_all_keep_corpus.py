"""初回KeepとRepair後Keepを重複なく統合し、教師SFT用コーパスを作る。"""

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


def latest_completed(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in pipeline.read_jsonl(path):
        if row.get("status") == "completed":
            latest[str(row["candidate_id"])] = row
    return latest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scope_relation(dialogue: dict[str, Any]) -> str:
    return str(
        dialogue.get("generation_condition", {})
        .get("problem_profile_assignment", {})
        .get("scope_relation", "unknown")
    )


def has_control_character(text: str) -> bool:
    return any(ord(char) < 32 and char not in "\n\t\r" for char in text)


def main() -> None:
    args = parse_args()
    root = args.input_dir.resolve()
    original_path = root / "candidate_dialogues.jsonl"
    repaired_path = root / "repaired_dialogues_sft_policy.jsonl"
    initial_audit_path = root / "dialogue_audits_sft_policy.jsonl"
    reaudit_path = root / "dialogue_reaudits_sft_policy.jsonl"
    surface_final_path = root / "surface_final_dialogues.jsonl"
    surface_audit_path = root / "surface_final_audits.jsonl"
    corpus_path = root / "v4_all_keep_corpus.jsonl"
    eligible_corpus_path = root / "v4_all_keep_sft_eligible_corpus.jsonl"
    sft_path = root / "v4_all_keep_sft_eligible.jsonl"
    manifest_path = root / "v4_all_keep_manifest.json"
    report_path = root / "REPORT_ALL_KEEP_CORPUS.md"

    original = {
        str(row["candidate_id"]): row for row in pipeline.read_jsonl(original_path)
    }
    repaired = {
        str(row["candidate_id"]): row for row in pipeline.read_jsonl(repaired_path)
    }
    initial_audits = latest_completed(initial_audit_path)
    reaudits = latest_completed(reaudit_path)
    initial_keep_ids = {
        candidate_id for candidate_id, row in initial_audits.items()
        if row.get("classification") == "Keep"
    }
    repaired_keep_ids = {
        candidate_id for candidate_id, row in reaudits.items()
        if row.get("classification") == "Keep"
    }
    overlap = initial_keep_ids & repaired_keep_ids
    if overlap:
        raise RuntimeError(f"初回KeepとRepair後Keepが重複しています: {sorted(overlap)}")
    if not initial_keep_ids <= set(original):
        raise RuntimeError("初回Keepに元対話が存在しないIDがあります")
    if not repaired_keep_ids <= set(repaired):
        raise RuntimeError("Repair後Keepに修正済み対話が存在しないIDがあります")

    dialogues: list[dict[str, Any]] = []
    audits: dict[str, dict[str, Any]] = {}
    for candidate_id in sorted(initial_keep_ids | repaired_keep_ids):
        if candidate_id in initial_keep_ids:
            dialogue = json.loads(json.dumps(original[candidate_id], ensure_ascii=False))
            audit = initial_audits[candidate_id]
            selection_path = "initial_keep"
        else:
            dialogue = json.loads(json.dumps(repaired[candidate_id], ensure_ascii=False))
            audit = reaudits[candidate_id]
            selection_path = "repair_then_full_reaudit_keep"
        dialogue["selection_path"] = selection_path
        dialogue["selection_audit"] = {
            "classification": "Keep",
            "total_score": int(audit["total_score"]),
            "audit_source": (
                initial_audit_path.name if selection_path == "initial_keep"
                else reaudit_path.name
            ),
        }
        dialogues.append(dialogue)
        audits[candidate_id] = audit

    surface_keep_ids: set[str] = set()
    surface_nonkeep_ids: set[str] = set()
    if surface_final_path.exists() and surface_audit_path.exists():
        surface_dialogues = {
            str(row["candidate_id"]): row
            for row in pipeline.read_jsonl(surface_final_path)
        }
        surface_audits = latest_completed(surface_audit_path)
        dialogue_map = {str(row["candidate_id"]): row for row in dialogues}
        for candidate_id, audit in surface_audits.items():
            if candidate_id not in dialogue_map or candidate_id not in surface_dialogues:
                continue
            if audit.get("classification") != "Keep":
                surface_nonkeep_ids.add(candidate_id)
                continue
            replacement = json.loads(
                json.dumps(surface_dialogues[candidate_id], ensure_ascii=False)
            )
            prior_path = str(dialogue_map[candidate_id]["selection_path"])
            replacement["selection_path"] = prior_path + "_surface_repaired_keep"
            replacement["selection_audit"] = {
                "classification": "Keep",
                "total_score": int(audit["total_score"]),
                "audit_source": surface_audit_path.name,
            }
            dialogue_map[candidate_id] = replacement
            audits[candidate_id] = audit
            surface_keep_ids.add(candidate_id)
        dialogues = [dialogue_map[candidate_id] for candidate_id in sorted(dialogue_map)]

    if len(dialogues) != len({str(row["candidate_id"]) for row in dialogues}):
        raise RuntimeError("統合Keepコーパスにcandidate_idの重複があります")
    invalid_audits = [
        candidate_id for candidate_id, row in audits.items()
        if row.get("classification") != "Keep"
        or row["audit"].get("issues")
        or row["audit"].get("repair_instructions")
    ]
    if invalid_audits:
        raise RuntimeError(f"Keep条件を満たさない監査があります: {invalid_audits}")

    control_locations: list[str] = []
    sft_ineligible: dict[str, list[str]] = {}
    teacher_turn_counts: list[int] = []
    repaired_teacher_turns = 0
    for dialogue in dialogues:
        candidate_id = str(dialogue["candidate_id"])
        teacher_turns = [
            turn for turn in dialogue["conversation"] if turn.get("role") == "teacher"
        ]
        teacher_turn_counts.append(len(teacher_turns))
        repaired_teacher_turns += sum(bool(turn.get("repaired")) for turn in teacher_turns)
        for index, turn in enumerate(teacher_turns):
            pipeline.validate_teacher_turn({
                key: turn[key] for key in pipeline.TEACHER_PROPERTIES
            })
            if has_control_character(str(turn["teacher_utterance"])):
                location = f"{candidate_id}:teacher-{index}"
                control_locations.append(location)
                sft_ineligible.setdefault(candidate_id, []).append(
                    f"教師発話の制御文字: teacher-{index}"
                )
        student_turns = [
            turn for turn in dialogue["conversation"] if turn.get("role") == "student"
        ]
        for index, turn in enumerate(student_turns):
            content = str(turn.get("content", ""))
            if has_control_character(content):
                control_locations.append(f"{candidate_id}:student-{index}")
                sft_ineligible.setdefault(candidate_id, []).append(
                    f"生徒発話の制御文字: student-{index}"
                )
            visible = "".join(
                char for char in content
                if ord(char) >= 32 or char in "\n\t\r"
            ).strip()
            if not visible:
                sft_ineligible.setdefault(candidate_id, []).append(
                    f"表示可能な内容がない生徒発話: student-{index}"
                )

    system_prompt_path = V4_DIR / "prompts" / "sft_teacher_system.txt"
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    eligible_dialogues = [
        dialogue for dialogue in dialogues
        if str(dialogue["candidate_id"]) not in sft_ineligible
    ]
    sft_rows = [
        {
            "id": dialogue["candidate_id"],
            "messages": pipeline.build_sft_messages(dialogue, system_prompt),
        }
        for dialogue in eligible_dialogues
    ]
    pipeline.write_jsonl(corpus_path, dialogues)
    pipeline.write_jsonl(eligible_corpus_path, eligible_dialogues)
    pipeline.write_jsonl(sft_path, sft_rows)

    audit_rows = [audits[str(dialogue["candidate_id"])] for dialogue in dialogues]
    eligible_audit_rows = [
        audits[str(dialogue["candidate_id"])] for dialogue in eligible_dialogues
    ]
    initial_keep_audit_rows = [initial_audits[cid] for cid in sorted(initial_keep_ids)]
    repaired_keep_audit_rows = [reaudits[cid] for cid in sorted(repaired_keep_ids)]
    total_scores = [int(row["total_score"]) for row in audit_rows]
    field_averages = {
        field: statistics.mean(float(row["audit"][field]) for row in audit_rows)
        for field in pipeline.SCORE_FIELDS
    }
    field_minima = {
        field: min(int(row["audit"][field]) for row in audit_rows)
        for field in pipeline.SCORE_FIELDS
    }
    completed = sum(bool(row.get("is_completed")) for row in dialogues)
    acceptable_incomplete = sum(
        bool(row["audit"].get("acceptable_incompleteness")) for row in audit_rows
    )
    metadata_warnings = sum(
        bool(row["audit"].get("metadata_warnings")) for row in audit_rows
    )
    assistant_targets = sum(
        message["role"] == "assistant"
        for row in sft_rows for message in row["messages"]
    )
    record_characters = [
        sum(len(message["content"]) for message in row["messages"])
        for row in sft_rows
    ]
    profile_counts = Counter(str(row["student_profile"]["id"]) for row in dialogues)
    scope_counts = Counter(scope_relation(row) for row in dialogues)
    emotion_counts = Counter(str(row["initial_emotion"]) for row in dialogues)

    def cohort_average(rows: list[dict[str, Any]]) -> float:
        return statistics.mean(float(row["total_score"]) for row in rows)

    manifest = {
        "corpus_file": corpus_path.name,
        "sft_eligible_corpus_file": eligible_corpus_path.name,
        "sft_file": sft_path.name,
        "source_files": {
            original_path.name: sha256(original_path),
            repaired_path.name: sha256(repaired_path),
            initial_audit_path.name: sha256(initial_audit_path),
            reaudit_path.name: sha256(reaudit_path),
            system_prompt_path.name: sha256(system_prompt_path),
            **(
                {
                    surface_final_path.name: sha256(surface_final_path),
                    surface_audit_path.name: sha256(surface_audit_path),
                }
                if surface_final_path.exists() and surface_audit_path.exists() else {}
            ),
        },
        "selection": {
            "initial_keep": len(initial_keep_ids),
            "repair_then_full_reaudit_keep": len(repaired_keep_ids),
            "selected_total": len(dialogues),
            "surface_repair_keep": len(surface_keep_ids),
            "surface_repair_nonkeep": len(surface_nonkeep_ids),
            "source_candidates": len(original),
            "selection_rate": round(len(dialogues) / len(original), 4),
            "candidate_ids": [str(row["candidate_id"]) for row in dialogues],
        },
        "integrity": {
            "unique_candidate_ids": True,
            "selection_path_overlap": 0,
            "surface_control_character_locations": len(control_locations),
            "sft_ineligible_dialogues": len(sft_ineligible),
            "sft_ineligible_reasons": dict(sorted(sft_ineligible.items())),
            "unresolved_issues": 0,
            "unresolved_repair_instructions": 0,
        },
        "sft_statistics": {
            "records": len(sft_rows),
            "assistant_targets": assistant_targets,
            "average_teacher_turns": round(assistant_targets / len(sft_rows), 2),
            "average_record_characters": round(statistics.mean(record_characters), 2),
            "maximum_record_characters": max(record_characters),
            "tokenizer_length_audit_required": True,
        },
        "evaluation": {
            "average_total_score": round(statistics.mean(total_scores), 2),
            "sft_eligible_average_total_score": round(
                statistics.mean(float(row["total_score"]) for row in eligible_audit_rows), 2
            ),
            "minimum_total_score": min(total_scores),
            "maximum_total_score": max(total_scores),
            "initial_keep_average": round(cohort_average(initial_keep_audit_rows), 2),
            "repaired_keep_average": round(cohort_average(repaired_keep_audit_rows), 2),
            "field_averages": {
                field: round(value, 2) for field, value in field_averages.items()
            },
            "field_minima": field_minima,
            "completed_dialogues": completed,
            "incomplete_dialogues": len(dialogues) - completed,
            "acceptable_incomplete_dialogues": acceptable_incomplete,
            "metadata_warning_dialogues": metadata_warnings,
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
        "# v4 全Keepコーパス評価", "",
        "## 抽出結果", "",
        f"- 生成候補: {len(original)}件",
        f"- Repairしていない初回Keep: {len(initial_keep_ids)}件",
        f"- Repair後の全体再監査Keep: {len(repaired_keep_ids)}件",
        f"- 統合Keepコーパス: {len(dialogues)}件（{len(dialogues) / len(original) * 100:.1f}%）",
        f"- 除外: {len(original) - len(dialogues)}件",
        f"- 表面破損修正後に再監査Keep: {len(surface_keep_ids)}件",
        f"- 表面破損修正後も保留: {len(surface_nonkeep_ids)}件",
        f"- 構造ゲート通過・SFT利用可能: {len(sft_rows)}件",
        f"- 制御文字等によりSFTから保留: {len(sft_ineligible)}件",
        f"- SFT利用可能データの教師ターゲット: {assistant_targets}ターン",
        f"- 現在のコーパスに含まれる修正済み教師ターン: {repaired_teacher_turns}ターン", "",
        "## 品質", "",
        f"- 合計平均: {statistics.mean(total_scores):.2f} / 60",
        f"- SFT利用可能{len(sft_rows)}件の合計平均: {statistics.mean(float(row['total_score']) for row in eligible_audit_rows):.2f} / 60",
        f"- 最低合計点: {min(total_scores)} / 60",
        f"- 最高合計点: {max(total_scores)} / 60",
        f"- 初回Keep群の平均: {cohort_average(initial_keep_audit_rows):.2f} / 60",
        f"- Repair後Keep群の平均: {cohort_average(repaired_keep_audit_rows):.2f} / 60", "",
        "| 評価軸 | 平均 | 最低 |", "|---|---:|---:|",
    ]
    for field in pipeline.SCORE_FIELDS:
        lines.append(
            f"| {SCORE_LABELS[field]} | {field_averages[field]:.2f} | {field_minima[field]} |"
        )
    lines.extend([
        "", "## 対話状態", "",
        f"- 完了対話: {completed}件",
        f"- 未完了対話: {len(dialogues) - completed}件",
        f"- 採択可能な未完了として監査された対話: {acceptable_incomplete}件",
        f"- 内部メタデータ警告を持つ対話: {metadata_warnings}件", "",
        "完了判定の最低点が2点の対話も、数学・共感・局所的な支援品質を満たし、最大ターン到達だけを理由に除外しない現行方針によってKeepとなっている。したがって、105件は一律に全6項目8点以上の集合ではない。",
        "",
        "## SFT構造ゲート", "",
        f"- 表面発話に制御文字がある位置: {len(control_locations)}か所",
        f"- SFTから保留した対話: {len(sft_ineligible)}件",
    ])
    lines.extend(
        f"- {candidate_id}: {' / '.join(reasons)}"
        for candidate_id, reasons in sorted(sft_ineligible.items())
    )
    lines.extend([
        "",
        "## 学習範囲との関係", "",
    ])
    lines.extend(f"- {key}: {value}件" for key, value in sorted(scope_counts.items()))
    lines.extend(["", "## 生徒プロフィール", ""])
    lines.extend(f"- {key}: {value}件" for key, value in sorted(profile_counts.items()))
    lines.extend(["", "## 初期感情", ""])
    lines.extend(f"- {key}: {value}件" for key, value in sorted(emotion_counts.items()))
    lines.extend([
        "", "## 総合評価", "",
        "初回Keep 81件とRepair後Keep 24件は重複せず、LLM judgeのKeep-only対話コーパスは105件である。未解決issuesと未解決repair_instructionsはない。",
        "",
        f"ただし、統合Keep群のうち{len(sft_ineligible)}件は表面発話に制御文字または空発話があり、LLM judgeが見落とした構造的不適格例である。この{len(sft_ineligible)}件は全Keep対話コーパスには保持したが、SFT用ファイルから保留した。現時点で直接SFTへ使用できるのは{len(sft_rows)}件である。",
        "",
        f"Keep判定は同じLLM judge系統による評価であり、独立した人手評価ではない。内部メタデータ警告を持つ対話も多いため、生徒モデルの学習にはそのまま使用しない。SFT実行前には、保留{len(sft_ineligible)}件の修正・再監査、別モデルまたは人手による層化抜き取り監査、対象tokenizer・chat templateによる長さ監査が必要である。",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
