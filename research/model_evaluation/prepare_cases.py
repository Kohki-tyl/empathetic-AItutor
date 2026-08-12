from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from common import read_json, read_jsonl, resolve_path, sha256_file, stable_fingerprint, utc_now, write_json, write_jsonl


BASE_DIR = Path(__file__).resolve().parent
PROFILE_FIELDS = {"id", "grade", "speech_style"}
SPEECH_STYLE_FIELDS = {"register", "confidence_expression", "response_length"}
LEARNING_STATUSES = {"mastered", "frontier", "one_step_beyond", "far_beyond"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="既存v4資産から確定評価ケースを構築する")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.example.json")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def indexed(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if not value:
            raise ValueError(f"{label}に{key}がありません")
        if value in result:
            raise ValueError(f"{label}の{key}が重複しています: {value}")
        result[value] = row
    return result


def build_cases(config_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    config_path = config_path.resolve()
    config = read_json(config_path)
    paths = config["paths"]
    questions_path = resolve_path(config_path, paths["questions"])
    similar_path = resolve_path(config_path, paths["similar_questions"])
    profiles_path = resolve_path(config_path, paths["profiles"])
    assignments_path = resolve_path(config_path, paths["assignments"])
    selection_path = resolve_path(config_path, paths["selection"])
    output_path = resolve_path(config_path, paths["cases"])
    training_corpora = [resolve_path(config_path, value) for value in paths.get("training_corpora", [])]
    training_audit_path = (
        resolve_path(config_path, paths["training_leakage_audit"])
        if paths.get("training_leakage_audit") else None
    )

    questions = indexed(read_jsonl(questions_path), "id", "問題")
    similar = indexed(read_jsonl(similar_path), "source_id", "類似問題")
    profiles = indexed(read_json(profiles_path), "id", "プロファイル")
    for profile_id, profile in profiles.items():
        if set(profile) != PROFILE_FIELDS:
            raise ValueError(
                f"{profile_id}: 簡易プロファイルはid・grade・speech_styleだけを含めてください"
            )
        speech_style = profile["speech_style"]
        if not isinstance(speech_style, dict) or set(speech_style) != SPEECH_STYLE_FIELDS:
            raise ValueError(f"{profile_id}: speech_styleの形式が不正です")
        if speech_style["register"] not in {"タメ口", "丁寧口調"}:
            raise ValueError(f"{profile_id}: registerが不正です")
        if speech_style["confidence_expression"] not in {"自信がある", "慎重", "控えめ"}:
            raise ValueError(f"{profile_id}: confidence_expressionが不正です")
        if speech_style["response_length"] not in {"短い", "標準"}:
            raise ValueError(f"{profile_id}: response_lengthが不正です")
    assignments = indexed(read_jsonl(assignments_path), "source_id", "割当")
    selection = read_json(selection_path)
    selected = selection.get("records")
    if not isinstance(selected, list) or not selected:
        raise ValueError("selection.recordsが空です")

    cases: list[dict[str, Any]] = []
    for index, selected_row in enumerate(selected):
        source_id = str(selected_row["source_id"])
        question = questions.get(source_id)
        similar_row = similar.get(source_id)
        assignment = assignments.get(source_id)
        if question is None or similar_row is None or assignment is None:
            raise ValueError(f"{source_id}: 問題・類似問題・割当のいずれかがありません")
        profile_id = str(assignment["profile_id"])
        if profile_id != str(selected_row["profile_id"]):
            raise ValueError(f"{source_id}: selectionとassignmentのprofileが不一致です")
        profile = profiles.get(profile_id)
        if profile is None:
            raise ValueError(f"{source_id}: profile {profile_id} がありません")
        learning_status = str(assignment["scope_relation"])
        if learning_status not in LEARNING_STATUSES:
            raise ValueError(f"{source_id}: scope_relationが不正です: {learning_status}")
        problem = question.get("translated_question") or question.get("original_problem")
        solution = question.get("translated_solution") or question.get("original_solution")
        if not isinstance(problem, str) or not problem.strip() or not isinstance(solution, str) or not solution.strip():
            raise ValueError(f"{source_id}: 問題または解答が空です")
        case_identity = {
            "source_id": source_id,
            "profile_id": profile_id,
            "learning_status": learning_status,
            "initial_emotion": assignment["initial_emotion"],
        }
        cases.append({
            "case_id": f"case-{index + 1:04d}-{stable_fingerprint(case_identity)[:10]}",
            "source_id": source_id,
            "order_index": selected_row.get("order_index"),
            "problem": problem.strip(),
            "reference_solution": solution.strip(),
            "similar_question": str(similar_row["similar_question"]).strip(),
            "similar_solution": str(similar_row["similar_solution"]).strip(),
            "profile_id": profile_id,
            "student_profile": {
                "grade": profile["grade"],
                "speech_style": profile["speech_style"],
                "initial_state": {
                    "learning_status": learning_status,
                    "emotion": assignment["initial_emotion"],
                    "emotion_reason": assignment.get("initial_emotion_reason", ""),
                },
            },
            "question_sha256": selected_row.get("question_sha256"),
        })

    source_ids = [row["source_id"] for row in cases]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("選択ケースにsource_id重複があります")
    training_source_ids: set[str] = set()
    for corpus_path in training_corpora:
        for row in read_jsonl(corpus_path):
            source_id = row.get("source_id") or row.get("id")
            if source_id:
                training_source_ids.add(str(source_id))
    overlap = sorted(set(source_ids) & training_source_ids)
    if overlap:
        raise ValueError(f"学習コーパスとのsource_id漏洩があります: {overlap[:10]}")
    precomputed_audit: dict[str, Any] | None = None
    if training_audit_path is not None:
        precomputed_audit = read_json(training_audit_path)
        expected_fingerprint = stable_fingerprint(sorted(source_ids))
        if precomputed_audit.get("selected_source_ids_sha256") != expected_fingerprint:
            raise ValueError("事前学習漏洩監査と選定100問のfingerprintが一致しません")
        if precomputed_audit.get("overlap_count") != 0:
            raise ValueError("事前学習漏洩監査に重複が記録されています")

    input_files = [questions_path, similar_path, profiles_path, assignments_path, selection_path, *training_corpora]
    if training_audit_path is not None:
        input_files.append(training_audit_path)
    audited_training_inputs = (
        precomputed_audit.get("training_corpora", []) if precomputed_audit else []
    )
    manifest = {
        "schema_version": "research-model-evaluation-cases-v3-four-state-profile",
        "created_at_utc": utc_now(),
        "case_count": len(cases),
        "profiles_reused_from": str(profiles_path),
        "profile_count": len(profiles),
        "learning_status_counts": dict(sorted(Counter(
            row["student_profile"]["initial_state"]["learning_status"] for row in cases
        ).items())),
        "initial_emotion_counts": dict(sorted(Counter(
            row["student_profile"]["initial_state"]["emotion"] for row in cases
        ).items())),
        "learning_status_emotion_counts": dict(sorted(Counter(
            f'{row["student_profile"]["initial_state"]["learning_status"]}|'
            f'{row["student_profile"]["initial_state"]["emotion"]}'
            for row in cases
        ).items())),
        "profile_counts": dict(sorted(Counter(row["profile_id"] for row in cases).items())),
        "training_corpora_checked": [str(path) for path in training_corpora],
        "precomputed_training_audit": str(training_audit_path) if training_audit_path else None,
        "precomputed_training_inputs": audited_training_inputs,
        "training_overlap_count": len(overlap),
        "inputs": {str(path): sha256_file(path) for path in input_files},
        "cases_fingerprint": stable_fingerprint(cases),
    }
    return cases, manifest, output_path


def main() -> None:
    args = parse_args()
    cases, manifest, configured_output = build_cases(args.config)
    output = args.output.resolve() if args.output else configured_output
    write_jsonl(output, cases)
    write_json(output.with_suffix(".manifest.json"), manifest)
    print(f"評価ケース: {len(cases)}件")
    print(f"出力: {output}")


if __name__ == "__main__":
    main()
