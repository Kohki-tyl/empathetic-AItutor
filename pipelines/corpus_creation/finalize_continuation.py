"""Verify and archive the legacy corpus continuation without changing the original 800."""
import argparse
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

import continue_corpus as c


def finalize(work, destination):
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("Generation has not completed all 500 dialogues")
    existing = c.BASE / "800_empathetic_dialogues.jsonl"
    questions_path = c.BASE / "questions/translated_additional_500_math_gpt6_luna.jsonl"
    old = c.read_rows(existing)
    questions = c.read_rows(questions_path)
    extra_path = work / "additional_500_empathetic_dialogues_801_1300.jsonl"
    combined_path = work / "1300_empathetic_dialogues.jsonl"
    extra, combined = c.read_rows(extra_path), c.read_rows(combined_path)
    config = manifest["config"]
    if c.file_hash(existing) != config["existing_sha256"] or c.file_hash(questions_path) != config["questions_sha256"]:
        raise ValueError("Source corpus/questions changed")
    if c.file_hash(extra_path) != manifest["additional_sha256"] or c.file_hash(combined_path) != manifest["combined_sha256"]:
        raise ValueError("Generation output hash mismatch")
    if len(old) != 800 or len(extra) != 500 or len(combined) != 1300 or combined[:800] != old or combined[800:] != extra:
        raise ValueError("Counts/order/old corpus prefix mismatch")
    if len({row["source_id"] for row in combined}) != 1300:
        raise ValueError("Duplicate source IDs")
    legacy_question_ids = {row["id"] for row in c.read_rows(c.BASE / "questions/translated_1000_math.jsonl")}
    if legacy_question_ids & {row["source_id"] for row in extra}:
        raise ValueError("Additional sources overlap the original 1000-problem pool")
    for index, (question, row) in enumerate(zip(questions, extra)):
        if row["source_id"] != question["id"] or row["problem"] != question["translated_question"]:
            raise ValueError("Question/dialogue alignment mismatch")
        if row["generation_metadata"]["dialogue_number"] != 801 + index:
            raise ValueError("Continuation numbering mismatch")
        if row["generation_metadata"]["run_fingerprint"] != config["run_fingerprint"]:
            raise ValueError("Generation fingerprint mismatch")
        c.validate_state(row, config["max_turns"], final=True)
    cleaned_fields = 0
    for row in extra:
        changes = []
        for index, utterance in enumerate(row["conversation"]):
            for field, text in list(utterance.items()):
                if not isinstance(text, str) or field == "role":
                    continue
                cleaned = c.repair_surface(text)
                if cleaned != text:
                    changes.append({"conversation_index": index, "field": field, "previous_sha256": c.digest(text)})
                    utterance[field] = cleaned
                    cleaned_fields += 1
        if changes:
            row["generation_metadata"]["surface_postprocessing"] = {
                "policy": "restore_known_LaTeX_escapes_remove_nonprinting_delimiters_v2", "changes": changes}
        c.validate_state(row, config["max_turns"], final=True)
    if cleaned_fields:
        backup = work / "additional_500_before_surface_cleanup.jsonl"
        if not backup.exists():
            shutil.copyfile(extra_path, backup)
        c.save_rows(extra_path, extra)
        c.save_rows(combined_path, old + extra)
        manifest.update(additional_sha256=c.file_hash(extra_path), combined_sha256=c.file_hash(combined_path),
                        surface_postprocessing_fields=cleaned_fields)
        c.save_json(work / "manifest.json", manifest)
    audit = {"existing_dialogues": 800, "additional_dialogues": 500, "combined_dialogues": 1300,
             "original_800_unchanged": True, "combined_prefix_matches_original": True,
             "duplicate_source_ids": 0, "overlap_with_original_1000_questions": 0,
             "teacher_turns": sum(len(r["conversation"]) // 2 for r in extra),
             "completed_dialogues": sum(r["is_completed"] for r in extra),
             "max_turn_incomplete_dialogues": sum(not r["is_completed"] for r in extra),
             "review_status": "structure_checked_not_mathematically_audited",
             "resolved_models": sorted({call["resolved_model"] for r in extra for call in r["generation_metadata"]["api_calls"]}),
             "api_calls": sum(len(r["generation_metadata"]["api_calls"]) for r in extra),
             "latex_surface_repaired_calls": sum(bool(call.get("surface_repairs")) for r in extra for call in r["generation_metadata"]["api_calls"])}
    audit["surface_postprocessed_fields"] = cleaned_fields
    destination.mkdir(parents=True, exist_ok=True)
    for path in (extra_path, combined_path, questions_path):
        target = destination / path.name
        if target.exists() and c.file_hash(target) != c.file_hash(path):
            raise FileExistsError(target)
    for path in (extra_path, combined_path):
        shutil.copyfile(path, destination / path.name)
        target = c.BASE / path.name
        if target.exists() and c.file_hash(target) != c.file_hash(path):
            raise FileExistsError(target)
        shutil.copyfile(path, target)
    shutil.copyfile(questions_path, destination / questions_path.name)
    for filename, expected_hash, is_profile in (
        ("teacher_system.txt", config["teacher_prompt_sha256"], False),
        ("student_system.txt", config["student_prompt_sha256"], False),
        ("student_profile.json", config["profiles_sha256"], True),
    ):
        path = c.BASE / "prompts" / filename
        actual = c.file_hash(path) if is_profile else c.digest(path.read_text(encoding="utf-8"))
        if actual != expected_hash:
            raise ValueError("Prompt/profile changed during generation")
        (destination / "prompts").mkdir(exist_ok=True)
        shutil.copyfile(path, destination / "prompts" / filename)
    manifest["archive_verification"] = audit
    manifest["runtime"] = {"python": platform.python_version(), "openai": importlib.metadata.version("openai")}
    manifest["usage"] = {key: sum(call.get("usage", {}).get(key, 0) for r in extra for call in r["generation_metadata"]["api_calls"])
                         for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    c.save_json(destination / "generation_manifest.json", manifest)
    c.save_json(destination / "structural_audit.json", audit)
    c.save_json((c.BASE / combined_path.name).with_suffix(".manifest.json"), manifest)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("data/corpus_continuation_500"))
    parser.add_argument("--destination", type=Path, default=Path("research/corpus/empathetic_dialogues_1300"))
    args = parser.parse_args()
    finalize(args.work_dir, args.destination)
