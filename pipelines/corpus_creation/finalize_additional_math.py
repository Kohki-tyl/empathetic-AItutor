"""Export completed basic translations in the existing translated_MATH question schema."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sft" / "baselines"))
import prepare as p
from audit_translation import audit


def finalize(work, existing_path, destination, translated_path=None):
    en_path = work / "math_additional_500_en.normalized.jsonl"
    ja_path = translated_path or work / "math_additional_500_ja.normalized.jsonl"
    old = p.engine.read_jsonl(existing_path)
    en, ja = p.engine.read_jsonl(en_path), p.engine.read_jsonl(ja_path)
    if len(old) != 1000 or len(en) != 500 or len(ja) != 500:
        raise ValueError("Expected 1000 existing + 500 additional translated problems")
    ids, problems = {r["id"] for r in old}, {r["original_problem"] for r in old}
    extra = []
    for original, translated in zip(en, ja):
        if original["id"] != translated["id"] or original["id"] in ids or original["original_problem"] in problems:
            raise ValueError("ID alignment mismatch or duplicate original problem")
        ids.add(original["id"])
        problems.add(original["original_problem"])
        if translated["translation"]["requested_model"] != "gpt-6-luna" or translated["translation"]["resolved_model"] != "gpt-6-luna":
            raise ValueError("Unexpected translation model")
        if original["original_problem"] != translated["original_problem"] or original["original_solution"] != translated["original_solution"]:
            raise ValueError("Original source content changed")
        extra.append({"id": original["id"], "type": original["type"], "level": original["level"],
                      "original_problem": original["original_problem"], "thought_process": "",
                      "translated_question": translated["messages"][1]["content"],
                      "original_solution": original["original_solution"],
                      "translated_solution": translated["messages"][2]["content"],
                      "source_unique_id": original["source_unique_id"], "original_math_split": original["original_math_split"],
                      "translation": translated["translation"], "translation_qa": translated.get("translation_qa", [])})
    audit(en_path, ja_path, work / "translation_audit.json", work / "translation_review.md")
    name = "translated_additional_500_math_gpt6_luna.jsonl"
    combined_name = "translated_1500_math_mixed_translation.jsonl"
    question_dir = existing_path.parent
    for path in (question_dir / name, question_dir / combined_name):
        if path.exists():
            raise FileExistsError(path)
    destination.mkdir(parents=True, exist_ok=True)
    p.write_rows(question_dir / name, extra)
    p.write_rows(question_dir / combined_name, old + extra)
    for filename in (name, combined_name):
        shutil.copyfile(question_dir / filename, destination / filename)
    for filename in ("selection_manifest.json", en_path.name, ja_path.name, "translation_audit.json", "translation_review.md",
                     ja_path.with_suffix(".manifest.json").name):
        shutil.copyfile(work / filename, destination / filename)
    report = p.engine.read_json(work / "translation_audit.json")
    manifest = {"additional_records": 500, "combined_records": 1500, "model": "gpt-6-luna",
        "existing_corpus": str(existing_path), "existing_sha256": p.engine.sha256_file(existing_path),
        "selection_manifest": "selection_manifest.json", "additional_file": name,
        "additional_sha256": p.engine.sha256_file(question_dir / name), "combined_file": combined_name,
        "combined_sha256": p.engine.sha256_file(question_dir / combined_name),
        "overlap_with_existing_ids": 0, "overlap_with_existing_problem_text": 0,
        "flagged_records": len(report["flagged_records"]), "review_status": "machine_translated_unreviewed",
        "thought_process": "empty: basic translation only; no generated reasoning trace",
        "combined_translation_models": "existing 1000: runtime model not recorded; new 500: gpt-6-luna"}
    p.engine.write_json(destination / "manifest.json", manifest)
    p.engine.write_json((question_dir / name).with_suffix(".manifest.json"), manifest)
    print(json.dumps({"additional": str(destination / name), "combined": str(destination / combined_name),
                      "additional_records": 500, "flagged_records": manifest["flagged_records"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--existing", type=Path, default=Path(__file__).resolve().parent / "questions/translated_1000_math.jsonl")
    parser.add_argument("--destination", type=Path, default=Path("research/corpus/math_additional_500_gpt6_luna"))
    parser.add_argument("--translated", type=Path, help="Completed normalized translation JSONL")
    args = parser.parse_args()
    finalize(args.work_dir, args.existing, args.destination, args.translated)


if __name__ == "__main__":
    main()
