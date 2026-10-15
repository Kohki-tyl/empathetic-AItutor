"""Fix train/validation counts while keeping each problem/conversation group together."""
from __future__ import annotations

import argparse
import copy
import random
from collections import defaultdict
from pathlib import Path

import prepare as p


def fixed_split(rows, validation_count=100, seed=42):
    p.engine.validate_messages(rows, len(rows), "plain")
    if not 0 < validation_count < len(rows):
        raise ValueError("validation_count must leave nonempty train and validation sets")
    groups = defaultdict(list)
    for row in rows:
        if row.get("split") not in {"train", "validation"}:
            raise ValueError("Missing/invalid split")
        groups[str(row["group_id"])].append(row)
    ordered = sorted(groups)
    random.Random(seed).shuffle(ordered)
    # Dynamic programming minimizes the number of changed rows, without splitting groups.
    choices = {0: (0, ())}
    for group in ordered:
        members = groups[group]
        original = {r["split"] for r in members}
        if len(original) != 1:
            raise ValueError(f"Group already leaks across splits: {group}")
        size = len(members)
        was_validation = "validation" in original
        updated = {}
        for count, (cost, selected) in choices.items():
            for selected_now in (False, True):
                new_count = count + (size if selected_now else 0)
                if new_count > validation_count:
                    continue
                new_cost = cost + (size if selected_now != was_validation else 0)
                candidate = (new_cost, selected + ((group,) if selected_now else ()))
                if new_count not in updated or new_cost < updated[new_count][0]:
                    updated[new_count] = candidate
        choices = updated
    if validation_count not in choices:
        raise ValueError("Exact count is impossible without splitting a group")
    selected = set(choices[validation_count][1])
    result = copy.deepcopy(rows)
    changes = []
    for row in result:
        previous = row["split"]
        row.setdefault("original_split", previous)
        row["split"] = "validation" if str(row["group_id"]) in selected else "train"
        if previous != row["split"]:
            changes.append({"id": row["id"], "group_id": row["group_id"],
                            "previous_split": previous, "split": row["split"]})
    assert sum(r["split"] == "validation" for r in result) == validation_count
    return result, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; source is not overwritten")
    parser.add_argument("--validation-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prepared = {}
    manifest = {"seed": args.seed, "validation_count": args.validation_count,
                "method": "minimum_changed_rows_with_indivisible_groups", "datasets": {}}
    for name in ("math", "empathy"):
        en_path = args.corpus_dir / f"{name}_en_1000.jsonl"
        ja_path = args.corpus_dir / f"{name}_ja_1000.jsonl"
        original_en = p.engine.read_jsonl(en_path)
        original_ja = p.engine.read_jsonl(ja_path)
        p.engine.validate_messages(original_ja, len(original_en), "plain")
        if [r["id"] for r in original_en] != [r["id"] for r in original_ja]:
            raise ValueError("English/Japanese IDs or order differ")
        en, changes = fixed_split(original_en, args.validation_count, args.seed)
        ja = copy.deepcopy(original_ja)
        for old, current, translated in zip(original_en, en, ja):
            if translated["translation"]["source_record_sha256"] != p.engine.canonical_hash(old):
                raise ValueError("Translation source hash mismatch")
            translated["original_split"] = current["original_split"]
            translated["split"] = current["split"]
            translated["translation"].setdefault("original_source_record_sha256", p.engine.canonical_hash(old))
            translated["translation"]["source_record_sha256"] = p.engine.canonical_hash(current)
        prepared[name] = (en, ja)
        manifest["datasets"][name] = {"train": len(en) - args.validation_count,
            "validation": args.validation_count, "changed_records": changes,
            "previous_en_sha256": p.engine.sha256_file(en_path), "previous_ja_sha256": p.engine.sha256_file(ja_path)}
    for name, (en, ja) in prepared.items():
        for language, rows in (("en", en), ("ja", ja)):
            path = args.output / f"{name}_{language}_1000.jsonl"
            p.write_rows(path, rows)
            manifest["datasets"][name][f"{language}_sha256"] = p.engine.sha256_file(path)
    p.engine.write_json(args.output / "fixed_split_manifest.json", manifest)
    print({name: {"train": len(en) - args.validation_count, "validation": args.validation_count}
           for name, (en, _) in prepared.items()})


if __name__ == "__main__":
    main()
