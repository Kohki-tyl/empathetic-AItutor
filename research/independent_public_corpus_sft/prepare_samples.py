"""公式train splitから会話単位で各500件を決定論的に抽出する。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


BASE_DIR = Path(__file__).resolve().parent


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def empathetic_dialogues(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metadata: dict[str, dict[str, str]] = {}
    for row in table.to_pylist():
        conv_id = str(row["conv_id"])
        grouped[conv_id].append(row)
        metadata[conv_id] = {
            "emotion": str(row["context"]).replace("_comma_", ","),
            "situation": str(row["prompt"]).replace("_comma_", ","),
        }
    dialogues = []
    for conv_id in sorted(grouped):
        source_turns = sorted(grouped[conv_id], key=lambda row: int(row["utterance_idx"]))
        turns = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": str(row["utterance"]).replace("_comma_", ","),
            }
            for index, row in enumerate(source_turns)
        ]
        if len(turns) >= 2:
            dialogues.append({"id": f"ed:{conv_id}", "source_split": "train", **metadata[conv_id], "turns": turns})
    return dialogues


def parse_mathdial_conversation(value: str) -> list[dict[str, str]]:
    turns = []
    for raw_turn in value.split("|EOM|"):
        raw_turn = raw_turn.strip()
        if not raw_turn:
            continue
        if raw_turn.startswith("Teacher:"):
            role, text = "assistant", raw_turn[len("Teacher:"):].strip()
            dialogue_act = ""
            if text.startswith("(") and ")" in text:
                dialogue_act, text = text[1:].split(")", 1)
                text = text.strip()
            if text:
                turns.append({"role": role, "dialogue_act": dialogue_act, "content": text})
        elif ":" in raw_turn:
            speaker, text = raw_turn.split(":", 1)
            if text.strip():
                turns.append({"role": "user", "speaker": speaker.strip(), "dialogue_act": "", "content": text.strip()})
        else:
            raise ValueError(f"MathDial turnを解析できません: {raw_turn!r}")
    if not turns or turns[0]["role"] != "assistant":
        raise ValueError("MathDial対話はteacher開始である必要があります")
    return turns


def mathdial_dialogues(path: Path) -> list[dict[str, Any]]:
    dialogues = []
    occurrences: dict[str, int] = defaultdict(int)
    for row in read_jsonl(path):
        base_identifier = f"mathdial:{row['qid']}:{row['scenario']}"
        occurrence = occurrences[base_identifier]
        occurrences[base_identifier] += 1
        identifier = f"{base_identifier}:{occurrence}"
        dialogues.append({
            "id": identifier,
            "source_split": "train",
            "question": str(row["question"]),
            "ground_truth": str(row["ground_truth"]),
            "student_incorrect_solution": str(row["student_incorrect_solution"]),
            "student_profile": str(row["student_profile"]),
            "teacher_described_confusion": str(row["teacher_described_confusion"]),
            "self_correctness": str(row["self-correctness"]),
            "source_ends_with": parse_mathdial_conversation(str(row["conversation"]))[-1]["role"],
            "turns": parse_mathdial_conversation(str(row["conversation"])),
        })
    return sorted(dialogues, key=lambda row: row["id"])


def sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) < size:
        raise ValueError(f"抽出元が不足しています: {len(rows)} < {size}")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [rows[index] for index in indices[:size]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.json")
    args = parser.parse_args()
    config = read_json(args.config)
    seed, size = int(config["seed"]), int(config["sample_size"])
    datasets = {
        "empathetic_dialogues": empathetic_dialogues(BASE_DIR / config["empathetic_dialogues"]["source"]),
        "mathdial": mathdial_dialogues(BASE_DIR / config["mathdial"]["source"]),
    }
    manifest: dict[str, Any] = {"schema_version": "public-corpus-samples-v1", "seed": seed, "sample_size": size, "datasets": {}}
    for name, rows in datasets.items():
        selected = sample(rows, size, seed)
        output = BASE_DIR / config[name]["sample"]
        write_jsonl(output, selected)
        manifest["datasets"][name] = {
            "source": config[name]["source"], "source_sha256": sha256_file(BASE_DIR / config[name]["source"]),
            "eligible_dialogues": len(rows), "output": config[name]["sample"], "output_sha256": sha256_file(output),
            "selected_ids": [row["id"] for row in selected],
        }
        print(f"{name}: {len(rows)}件から{len(selected)}件 -> {output}")
    manifest_path = BASE_DIR / "data" / "sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
