"""SFT JSONLへchat templateを適用し、切り詰め前の系列長を監査する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    args = parse_args()
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformersが必要です。ABCI環境ではrequirements-abci.txtを導入してください。"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=False,
    )
    rows = read_jsonl(args.input)
    lengths: list[int] = []
    overlength: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"record {index} has no messages list")
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        length = len(token_ids)
        lengths.append(length)
        if length > args.max_length:
            overlength.append({
                "id": row.get("id", index),
                "tokens": length,
                "excess_tokens": length - args.max_length,
            })
    report = {
        "input": str(args.input.resolve()),
        "model": args.model,
        "revision": args.revision,
        "max_length": args.max_length,
        "records": len(rows),
        "maximum_tokens": max(lengths, default=0),
        "average_tokens": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "overlength_records": len(overlength),
        "overlength": overlength,
        "safe_to_train_without_truncation": not overlength,
    }
    output = args.output or args.input.with_name("sft_length_audit.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if overlength:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
