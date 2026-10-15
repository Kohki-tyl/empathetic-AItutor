"""Check source alignment and write a bilingual review sample for Japanese corpora."""
import argparse
import random
import re
from collections import Counter
from pathlib import Path

import translate_ja as t


def audit(source_path, target_path, report_path, review_path):
    source = t.p.engine.read_jsonl(source_path)
    target = t.p.engine.read_jsonl(target_path)
    t.p.engine.validate_messages(source, len(source), "plain")
    t.p.engine.validate_messages(target, len(source), "plain")
    flagged = []
    for en, ja in zip(source, target):
        for key in ("id", "group_id", "source", "split"):
            if en[key] != ja[key]:
                raise ValueError(f"Metadata mismatch: {en['id']}:{key}")
        if len(en["messages"]) != len(ja["messages"]):
            raise ValueError("Turn count mismatch")
        if ja["language"] != "ja" or ja["translation"]["source_record_sha256"] != t.p.engine.canonical_hash(en):
            raise ValueError("Source hash/language mismatch")
        if ja["messages"][0]["content"] != t.SYSTEM_JA:
            raise ValueError("System prompt mismatch")
        flags = []
        for index, (a, b) in enumerate(zip(en["messages"][1:], ja["messages"][1:]), 1):
            if a["role"] != b["role"] or "__KEEP_" in b["content"]:
                raise ValueError("Role mismatch or unresolved placeholder")
            original_numbers = t.source_numbers(a["content"])
            translated_numbers = Counter(re.findall(r"\d+(?:[.,]\d+)*", b["content"]))
            if original_numbers - translated_numbers:
                raise ValueError(f"Original numbers missing: {en['id']}:{index}")
            for span in t.PROTECTED.findall(t.normalize_typography(a["content"])):
                if span not in b["content"]:
                    raise ValueError(f"Protected text missing: {en['id']}:{index}")
            if translated_numbers - original_numbers:
                flags.append({"turn": index, "type": "additional_numeric_notation",
                              "values": list((translated_numbers - original_numbers).elements())})
            if re.search(r"[\u3040-\u9fff。？！?!]\s*\d+(?:\s+\d+)*[。？！?!]?$", b["content"]) and not re.search(r"\d[.,!?]?$", a["content"]):
                flags.append({"turn": index, "type": "numeric_suffix_requires_review"})
        if flags:
            flagged.append({"id": en["id"], "flags": flags})
    report = {"records": len(target), "source_sha256": t.p.engine.sha256_file(source_path),
              "target_sha256": t.p.engine.sha256_file(target_path),
              "models": sorted({r["translation"]["resolved_model"] for r in target}),
              "train_records": sum(r["split"] == "train" for r in target),
              "validation_records": sum(r["split"] == "validation" for r in target),
              "assistant_turns": sum(m["role"] == "assistant" for r in target for m in r["messages"]),
              "flagged_records": flagged, "status": "structural_checks_passed",
              "review_status": "machine_translated_unreviewed",
              "limitations": "Checks do not establish semantic fidelity, correct empathy, or mathematical correctness."}
    t.p.engine.write_json(report_path, report)
    by_id = {r["id"]: i for i, r in enumerate(source)}
    selected = list(dict.fromkeys([by_id[f["id"]] for f in flagged] + random.Random(42).sample(range(len(source)), min(10, len(source)))))
    lines = ["# 機械翻訳レビュー用対訳", "", "構造・保護対象の検査済み。意味の忠実性は未レビュー。", ""]
    for index in selected:
        en, ja = source[index], target[index]
        lines.extend([f"## {en['id']} ({en['split']})", ""])
        for number, (a, b) in enumerate(zip(en["messages"][1:], ja["messages"][1:]), 1):
            lines.extend([f"### {number}: {a['role']}", "", "原文:", "", a["content"], "", "訳文:", "", b["content"], ""])
    review_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Audited {len(target)} dialogues; {len(flagged)} flagged for translation review")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source", "target", "report", "review"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    audit(args.source, args.target, args.report, args.review)


if __name__ == "__main__":
    main()
