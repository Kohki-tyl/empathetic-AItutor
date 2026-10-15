"""Select additional MATH train problems after the existing translated 1,000."""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sft" / "baselines"))
import prepare as p

DATASET = "nlile/hendrycks-MATH-benchmark"


def text_only(problem, solution):
    if "[asy]" in problem or "[/asy]" in problem or "[asy]" in solution or "[/asy]" in solution:
        return False
    return not any(re.search(pattern, problem, re.I | re.S) for pattern in (
        r"shown.*figure", r"shown.*diagram", r"shown.*graph", r"shown.*below", r"as.*shown", r"refer.*diagram"))


def select_additional(raw, existing, count=500):
    indices = [int(r["id"].removeprefix("math_train_")) for r in existing]
    for index, previous in zip(indices, existing):
        if index >= len(raw) or raw[index]["problem"] != previous["original_problem"]:
            raise ValueError(f"Dataset ordering/content differs from existing corpus: math_train_{index}")
        if raw[index]["solution"] != previous["original_solution"]:
            raise ValueError(f"Dataset solution differs: math_train_{index}")
    seen_problems = {r["original_problem"] for r in existing}
    selected, skipped = [], []
    for index in range(max(indices) + 1, len(raw)):
        row = raw[index]
        problem, solution = row["problem"], row["solution"]
        if not text_only(problem, solution):
            skipped.append({"index": index, "reason": "diagram_dependent_or_asy"})
            continue
        if problem in seen_problems:
            skipped.append({"index": index, "reason": "duplicate_problem"})
            continue
        seen_problems.add(problem)
        identifier = f"math_train_{index}"
        selected.append({"id": identifier, "group_id": identifier, "source": DATASET,
            "source_index": index, "source_unique_id": row.get("unique_id"),
            "original_math_split": str(row.get("unique_id", "unknown")).split("/", 1)[0],
            "split": "train", "language": "en", "type": row.get("type", row.get("subject", "unknown")), "level": row.get("level", "unknown"),
            "original_problem": problem, "original_solution": solution,
            "messages": [{"role": "system", "content": p.SYSTEM}, {"role": "user", "content": problem},
                         {"role": "assistant", "content": solution}]})
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} additional valid problems available, requested {count}")
    return selected, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, default=Path(__file__).resolve().parent / "questions/translated_1000_math.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    info_path = root / "dataset_info.json"
    if not info_path.exists():
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{DATASET}", timeout=60) as response:
            p.engine.write_json(info_path, json.load(response))
    info = p.engine.read_json(info_path)
    revision = info["sha"]
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/data/train-00000-of-00001.parquet"
    parquet = root / "train.parquet"
    if not parquet.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            temporary = parquet.with_suffix(".tmp")
            temporary.write_bytes(response.read())
            temporary.replace(parquet)
    import pyarrow.parquet as pq
    raw = pq.read_table(parquet).to_pylist()
    existing = p.engine.read_jsonl(args.existing)
    selected, skipped = select_additional(raw, existing, args.count)
    p.engine.validate_messages(selected, len(selected), "plain")
    output = root / f"math_additional_{args.count}_en.normalized.jsonl"
    if output.exists():
        raise FileExistsError(output)
    p.write_rows(output, selected)
    p.engine.write_json(root / "selection_manifest.json", {
        "dataset": DATASET, "dataset_revision": revision, "source_split": "train",
        "source_url": url, "source_parquet_sha256": p.engine.sha256_file(parquet),
        "existing": str(args.existing), "existing_sha256": p.engine.sha256_file(args.existing),
        "existing_records": len(existing), "existing_alignment_verified": True,
        "selection": "sequential_after_max_existing_source_index_excluding_diagrams_and_duplicates",
        "count": len(selected), "selected_ids": [r["id"] for r in selected], "skipped": skipped,
        "original_math_splits": {split: sum(r["original_math_split"] == split for r in selected)
                                 for split in sorted({r["original_math_split"] for r in selected})},
        "original_math_test_may_be_present": True,
        "output_sha256": p.engine.sha256_file(output),
    })
    print(f"Verified {len(existing)} existing source pairs; selected {len(selected)} additional problems: {selected[0]['id']} .. {selected[-1]['id']}")


if __name__ == "__main__":
    main()
