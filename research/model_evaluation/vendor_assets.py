from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from common import read_json, read_jsonl, sha256_file, stable_fingerprint, utc_now, write_json


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
SOURCE_FILES = {
    "questions": REPO_ROOT / "pipelines/model_evaluation/shared/questions/test_math_questions.jsonl",
    "similar_questions": REPO_ROOT / "pipelines/model_evaluation/shared/questions/similar_test_math_questions.jsonl",
    "assignments": REPO_ROOT / "pipelines/model_evaluation/test_v4/prompts/problem_profile_assignments.jsonl",
    "excluded_question_ids": REPO_ROOT / "pipelines/model_evaluation/shared/questions/excluded_test_question_ids.json",
}
DESTINATIONS = {
    "questions": "test_math_questions.jsonl",
    "similar_questions": "similar_test_math_questions.jsonl",
    "assignments": "problem_profile_assignments.jsonl",
    "excluded_question_ids": "excluded_test_question_ids.json",
}
TRAINING_CORPORA = [
    REPO_ROOT / "pipelines/corpus_creation/v3/data/v3_rebuilt_corpus.jsonl",
    REPO_ROOT / "pipelines/corpus_creation/v4/data/run_10_openai_gpt54mini/v4_all_keep_corpus.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABCI単独実行用の固定評価資産を同梱する")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "assets")
    return parser.parse_args()


def relative_to_repo(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def source_ids(rows: list[dict[str, Any]], *keys: str) -> list[str]:
    result: list[str] = []
    for row in rows:
        value = next((row.get(key) for key in keys if row.get(key)), None)
        if value is None:
            raise ValueError(f"source_idを取得できません: {row}")
        result.append(str(value))
    if len(result) != len(set(result)):
        raise ValueError("source_idが重複しています")
    return result


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = read_jsonl(SOURCE_FILES["questions"])
    similars = read_jsonl(SOURCE_FILES["similar_questions"])
    assignments = read_jsonl(SOURCE_FILES["assignments"])
    question_ids = source_ids(questions, "id", "source_id")
    if len(question_ids) != 200:
        raise ValueError(f"評価候補問題は200件必要です: {len(question_ids)}")
    if set(source_ids(similars, "source_id", "id")) != set(question_ids):
        raise ValueError("元問題と類似問題のID集合が一致しません")
    if set(source_ids(assignments, "source_id", "id")) != set(question_ids):
        raise ValueError("元問題と割当のID集合が一致しません")

    excluded = set(str(value) for value in read_json(
        SOURCE_FILES["excluded_question_ids"]
    )["excluded_source_ids"])
    if not excluded <= set(question_ids):
        raise ValueError("除外IDに評価候補外のIDがあります")

    selection = read_json(BASE_DIR / "selections/evaluation_100.json")
    selected_ids = [str(row["source_id"]) for row in selection["records"]]
    if len(selected_ids) != 100 or not set(selected_ids) <= set(question_ids) - excluded:
        raise ValueError("固定100問選定と同梱候補が一致しません")

    copied: dict[str, dict[str, Any]] = {}
    for name, source in SOURCE_FILES.items():
        destination = output_dir / DESTINATIONS[name]
        shutil.copyfile(source, destination)
        copied[name] = {
            "path": f"assets/{destination.name}",
            "sha256": sha256_file(destination),
            "source": relative_to_repo(source),
            "source_sha256": sha256_file(source),
        }

    selected_set = set(selected_ids)
    training_ids: set[str] = set()
    training_inputs: list[dict[str, Any]] = []
    for corpus in TRAINING_CORPORA:
        for row in read_jsonl(corpus):
            value = row.get("source_id") or row.get("id")
            if value:
                training_ids.add(str(value))
        training_inputs.append({
            "source": relative_to_repo(corpus),
            "sha256": sha256_file(corpus),
        })
    overlap = sorted(selected_set & training_ids)
    if overlap:
        raise ValueError(f"学習コーパスとのsource_id漏洩があります: {overlap[:10]}")

    write_json(output_dir / "training_leakage_audit.json", {
        "schema_version": "precomputed-training-source-id-audit-v1",
        "created_at_utc": utc_now(),
        "selection_sha256": selection["selection_sha256"],
        "selected_source_ids_sha256": stable_fingerprint(sorted(selected_ids)),
        "selected_count": len(selected_ids),
        "training_corpora": training_inputs,
        "overlap_count": 0,
        "overlapping_source_ids": [],
        "scope": "source_id exact-match audit; training corpora are not bundled",
    })
    copied["training_leakage_audit"] = {
        "path": "assets/training_leakage_audit.json",
        "sha256": sha256_file(output_dir / "training_leakage_audit.json"),
    }
    write_json(output_dir / "manifest.json", {
        "schema_version": "standalone-evaluation-assets-v1",
        "created_at_utc": utc_now(),
        "candidate_question_count": len(question_ids),
        "excluded_question_count": len(excluded),
        "selected_evaluation_count": len(selected_ids),
        "selection_sha256": selection["selection_sha256"],
        "files": copied,
    })
    print(f"同梱資産: {output_dir}")
    print(f"候補問題: {len(question_ids)}件 / 評価選定: {len(selected_ids)}件")


if __name__ == "__main__":
    main()
