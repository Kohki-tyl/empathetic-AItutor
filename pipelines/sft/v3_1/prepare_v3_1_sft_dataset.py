"""v3 SFTデータをv4教師prompt・分析区分へ整合させたv3.1を作成する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
SFT_ROOT = BASE_DIR.parent
REPO_ROOT = BASE_DIR.parents[2]
DEFAULT_INPUT = SFT_ROOT / "v3" / "data" / "v3_cot_sft.jsonl"
DEFAULT_SOURCE_MANIFEST = SFT_ROOT / "v3" / "data" / "v3_cot_sft_manifest.json"
DEFAULT_PROMPT = REPO_ROOT / "SFT_abci" / "test" / "test_v4" / "prompts" / "teacher_system.txt"
DEFAULT_OUTPUT = BASE_DIR / "data" / "v3_1_cot_sft.jsonl"
DEFAULT_MANIFEST = BASE_DIR / "data" / "v3_1_cot_sft_manifest.json"

ASSISTANT_PATTERN = re.compile(
    r"\A<analysis>\n(?P<analysis>.*?)\n</analysis>\n<final>\n(?P<final>.*?)\n</final>\Z",
    re.DOTALL,
)
V3_ANALYSIS_PATTERN = re.compile(
    r"\A【認知状態】(?P<cognition>.*?)\n"
    r"【感情状態】(?P<emotion>.*?)\n"
    r"【数学的検証】(?P<math>.*?)\n"
    r"【次の一歩】(?P<support>.*)\Z",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v3.1 CoT SFTデータを作成する")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def convert_assistant(content: str, record_index: int, message_index: int) -> str:
    outer = ASSISTANT_PATTERN.fullmatch(content)
    if outer is None:
        raise ValueError(f"CoT外側形式が不正です: record={record_index}, message={message_index}")
    inner = V3_ANALYSIS_PATTERN.fullmatch(outer.group("analysis"))
    if inner is None:
        raise ValueError(f"v3分析4区分が不正です: record={record_index}, message={message_index}")
    cognition = inner.group("cognition").strip()
    emotion = inner.group("emotion").strip()
    math = inner.group("math").strip()
    support = inner.group("support").strip()
    final = outer.group("final").strip()
    if not all((cognition, emotion, math, support, final)):
        raise ValueError(f"変換対象フィールドが空です: record={record_index}, message={message_index}")
    return (
        "<analysis>\n"
        f"【数学的評価】{math}\n"
        f"【生徒状態】{cognition} 感情状態は{emotion}\n"
        f"【支援判断】{support}\n"
        "</analysis>\n"
        "<final>\n"
        f"{final}\n"
        "</final>"
    )


def convert_record(record: dict[str, Any], prompt: str, record_index: int) -> tuple[dict[str, Any], int]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages or messages[0].get("role") != "system":
        raise ValueError(f"先頭systemがありません: record={record_index}")
    converted: list[dict[str, str]] = []
    assistant_count = 0
    for message_index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError(f"message形式が不正です: record={record_index}, message={message_index}")
        if message_index == 0:
            converted.append({"role": "system", "content": prompt})
        elif role == "assistant":
            converted.append(
                {
                    "role": "assistant",
                    "content": convert_assistant(content, record_index, message_index),
                }
            )
            assistant_count += 1
        else:
            converted.append({"role": role, "content": content})
    if assistant_count == 0:
        raise ValueError(f"assistant発話がありません: record={record_index}")
    return {"messages": converted}, assistant_count


def main() -> None:
    args = parse_args()
    prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    source_records = read_jsonl(args.input)
    converted_records: list[dict[str, Any]] = []
    assistant_turn_count = 0
    for record_index, record in enumerate(source_records, start=1):
        converted, count = convert_record(record, prompt, record_index)
        converted_records.append(converted)
        assistant_turn_count += count

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for record in converted_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "dataset_name": "v3_1_v4_prompt_cot_sft",
        "source_dataset": str(args.input.resolve()),
        "source_dataset_sha256": sha256_file(args.input),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "system_prompt": str(args.system_prompt.resolve()),
        "system_prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "output_sha256": sha256_file(args.output),
        "selected_count": len(converted_records),
        "assistant_turn_count": assistant_turn_count,
        "source_ids_in_output_order": source_manifest.get("source_ids_in_output_order", []),
        "transformation": {
            "records_reordered": False,
            "user_messages_changed": False,
            "final_messages_changed": False,
            "system_prompt_replaced_with_v4": True,
            "analysis_mapping": {
                "数学的検証": "数学的評価",
                "認知状態 + 感情状態": "生徒状態",
                "次の一歩": "支援判断",
            },
        },
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"v3.1 dataset: {len(converted_records)} records, "
        f"{assistant_turn_count} assistant turns -> {args.output}"
    )


if __name__ == "__main__":
    main()
