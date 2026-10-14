"""Convert official corpora and build reproducible SFT controls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v4"))
import train_v4_sft as engine

SYSTEM = "You are a helpful assistant. Respond appropriately to the conversation."


def write_rows(path, rows):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def group_split(key, seed, ratio):
    value = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), 16) / 2**256
    return "validation" if value < ratio else "train"


def mathdial(rows, seed=42, ratio=0.1):
    result = []
    for index, row in enumerate(rows):
        qid = str(row["qid"])
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": row["question"].strip()}]
        for turn in row["conversation"].split("|EOM|"):
            if not turn.strip():
                continue
            student_name = str(row.get("student_profile", "")).split(" ", 1)[0]
            personas = "Teacher|Student" + ("|" + re.escape(student_name) if student_name else "")
            match = re.fullmatch(r"\s*(" + personas + r"):\s*(.*)", turn, re.DOTALL)
            if match is None:
                raise ValueError(f"Unknown MathDial turn: {qid}:{turn!r}")
            role = "assistant" if match[1] == "Teacher" else "user"
            text = match[2].strip()
            if role == "assistant":
                text = re.sub(r"^\([^()]+\)\s*", "", text).strip()
            if not text:
                raise ValueError(f"Empty MathDial turn: {qid}")
            if messages[-1]["role"] == role:
                messages[-1]["content"] += "\n\n" + text
            else:
                messages.append({"role": role, "content": text})
        if messages[-1]["role"] == "user":
            messages.pop()
        result.append({"id": f"mathdial:{qid}:{index}", "source": "mathdial",
                       "group_id": qid, "language": "en",
                       "split": group_split(qid, seed, ratio), "messages": messages})
    engine.validate_messages(result, len(result), "plain")
    return result


def empathetic(rows, split):
    groups = defaultdict(list)
    for row in rows:
        groups[row["conv_id"]].append(row)
    result = []
    for cid, turns in sorted(groups.items()):
        turns.sort(key=lambda t: int(t["utterance_idx"]))
        indices = [int(t["utterance_idx"]) for t in turns]
        if indices != list(range(1, len(turns) + 1)):
            raise ValueError(f"Missing/duplicate utterance indices: {cid}")
        speaker = turns[0]["speaker_idx"]
        messages = [{"role": "system", "content": SYSTEM}]
        for index, turn in enumerate(turns):
            role = "user" if turn["speaker_idx"] == speaker else "assistant"
            if role != ("user" if index % 2 == 0 else "assistant"):
                raise ValueError(f"Invalid speaker order: {cid}")
            messages.append({"role": role, "content": turn["utterance"].replace("_comma_", ",").strip()})
        if messages[-1]["role"] == "user":
            messages.pop()
        result.append({"id": f"empatheticdialogues:{cid}", "source": "empatheticdialogues",
                       "group_id": cid, "language": "en", "split": split, "messages": messages})
    engine.validate_messages(result, len(result), "plain")
    return result


def select(rows, seed, budget=None, tokenizer=None, max_length=8192):
    rows = sorted(rows, key=lambda r: r["id"])
    random.Random(seed).shuffle(rows)
    selected, tokens = [], 0
    for row in rows:
        count = None
        if tokenizer is not None:
            encoded = engine.encode_with_assistant_mask(tokenizer, row, max_length, 1, 1, "plain")
            count = sum(label != -100 for label in encoded.labels)
        if budget is not None and tokens + count > budget:
            continue
        selected.append(row)
        tokens += count or 0
    if not selected:
        raise ValueError("No records fit the selection/budget")
    return selected, tokens if tokenizer is not None else None


def build(args):
    pools = {"math": engine.read_jsonl(args.math), "empathy": engine.read_jsonl(args.empathy)}
    for rows in pools.values():
        engine.validate_messages(rows, len(rows), "plain")
        engine.deterministic_split(rows, 0.1, args.seed)
        seen = {}
        for row in rows:
            if row.get("language") != args.language:
                raise ValueError("language metadata must match --language; translation is not automatic")
            group = row["group_id"]
            if group in seen and seen[group] != row["split"]:
                raise ValueError(f"Group leaks across splits: {group}")
            seen[group] = row["split"]
    base = engine.read_json(Path(__file__).resolve().parents[1] / "v4" / "config.json")
    tokenizer = None
    if args.target_tokens:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base["model_name"], revision=base["model_revision"])
    report = {"language": args.language, "seed": args.seed,
              "inputs": {str(p): engine.sha256_file(p) for p in (args.math, args.empathy)},
              "target_tokens": args.target_tokens, "conditions": {}}
    if args.output.exists():
        raise FileExistsError(f"Use a new output directory: {args.output}")
    prepared = []
    for condition in ("math", "empathy", "mixed"):
        chosen, counts = [], {}
        sources = ("math", "empathy") if condition == "mixed" else (condition,)
        for source in sources:
            training = [r for r in pools[source] if r["split"] == "train"]
            budget = args.target_tokens
            if budget and condition == "mixed":
                budget = budget // 2 if source == "math" else budget - budget // 2
            selected, count = select(training, args.seed, budget, tokenizer, base["max_length"])
            if budget and count < budget * 0.95:
                raise ValueError(f"{condition}/{source}: insufficient tokens ({count}/{budget}); lower budget")
            chosen.extend(selected)
            chosen.extend(r for r in pools[source] if r["split"] == "validation")
            counts[source] = {"train_records": len(selected), "assistant_target_tokens": count}
        engine.validate_messages(chosen, len(chosen), "plain")
        prepared.append((condition, chosen))
        report["conditions"][condition] = counts
    for condition, chosen in prepared:
        dataset = args.output / f"{condition}.jsonl"
        write_rows(dataset, chosen)
        config = dict(base, dataset=dataset.name, dataset_sha256=engine.sha256_file(dataset),
                      expected_records=len(chosen), response_format="plain", seed=args.seed,
                      analysis_loss_weight=1.0, final_loss_weight=1.0,
                      output_dir=f"outputs/{condition}_sft")
        engine.write_json(args.output / f"{condition}.json", config)
    engine.write_json(args.output / "preparation_manifest.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    math = commands.add_parser("mathdial")
    math.add_argument("--input", type=Path, required=True, help="Official train.jsonl only")
    math.add_argument("--seed", type=int, default=42)
    math.add_argument("--validation-ratio", type=float, default=0.1)
    ed = commands.add_parser("empathy")
    ed.add_argument("--train", type=Path, required=True)
    ed.add_argument("--valid", type=Path, required=True)
    for sub in (math, ed):
        sub.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("build")
    run.add_argument("--math", type=Path, required=True)
    run.add_argument("--empathy", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--language", choices=("en", "ja"), required=True)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--target-tokens", type=int, help="Train target budget per condition; requires tokenizer")
    args = parser.parse_args()
    if args.command == "build":
        if args.target_tokens is not None and args.target_tokens < 2:
            parser.error("--target-tokens must be >= 2")
        build(args)
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.command == "mathdial":
        if not 0 < args.validation_ratio < 0.5:
            parser.error("--validation-ratio must be between 0 and 0.5")
        rows = mathdial(engine.read_jsonl(args.input), args.seed, args.validation_ratio)
        inputs = [args.input]
    else:
        rows = []
        inputs = [args.train, args.valid]
        for path, split in zip(inputs, ("train", "validation")):
            with path.open(encoding="utf-8", newline="") as stream:
                rows.extend(empathetic(list(csv.DictReader(stream)), split))
    engine.validate_messages(rows, len(rows), "plain")
    engine.deterministic_split(rows, 0.1, 42)
    write_rows(args.output, rows)
    engine.write_json(args.output.with_suffix(".manifest.json"), {
        "inputs": {str(p): engine.sha256_file(p) for p in inputs},
        "output_sha256": engine.sha256_file(args.output), "records": len(rows),
        "command": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    })


if __name__ == "__main__":
    main()
