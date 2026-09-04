"""全翻訳結果とSFT構造を検証し、再現性manifestを生成する。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import translate_samples as pipeline


BASE_DIR = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset(dataset: str, config: dict[str, Any]) -> dict[str, Any]:
    source_path = BASE_DIR / config[dataset]["sample"]
    translated_path = BASE_DIR / config[dataset]["translated"]
    sft_path = BASE_DIR / config[dataset]["sft"]
    sources = pipeline.read_jsonl(source_path)
    records = pipeline.read_jsonl(translated_path)
    sft_rows = pipeline.read_jsonl(sft_path)
    if len(sources) != 500 or len(records) != 500 or len(sft_rows) != 500:
        raise ValueError(f"{dataset}: 件数不一致: source={len(sources)}, translated={len(records)}, sft={len(sft_rows)}")
    source_by_id = {row["id"]: row for row in sources}
    record_by_id = {row["id"]: row for row in records}
    sft_by_id = {row["id"]: row for row in sft_rows}
    if not (len(source_by_id) == len(record_by_id) == len(sft_by_id) == 500):
        raise ValueError(f"{dataset}: 重複IDがあります")
    if set(source_by_id) != set(record_by_id) or set(source_by_id) != set(sft_by_id):
        raise ValueError(f"{dataset}: ID集合が一致しません")

    trailing_user_sources = 0
    merged_adjacent_role_sources = 0
    leading_assistant_turns_removed_for_sft = 0
    for identifier, source in source_by_id.items():
        record = record_by_id[identifier]
        if record.get("model") != config["model"] or not str(record.get("response_id", "")):
            raise ValueError(f"{dataset}: API来歴が不正です: {identifier}")
        expected_hash = pipeline.sha256_text(json.dumps(source, ensure_ascii=False, sort_keys=True))
        if record.get("source_sha256") != expected_hash:
            raise ValueError(f"{dataset}: 原文hashが一致しません: {identifier}")
        pipeline.validate_translation(source, record["translation"], dataset)
        roles = [message["role"] for message in sft_by_id[identifier]["messages"]]
        if roles[0] != "system" or roles[-1] != "assistant":
            raise ValueError(f"{dataset}: SFT終端roleが不正です: {identifier}")
        if any(left == right for left, right in zip(roles[1:], roles[2:])):
            raise ValueError(f"{dataset}: SFT roleが交互ではありません: {identifier}")
        if any(not str(message.get("content", "")).strip() for message in sft_by_id[identifier]["messages"]):
            raise ValueError(f"{dataset}: 空のSFT発話があります: {identifier}")
        source_roles = [turn["role"] for turn in source["turns"]]
        leading_assistant_turns_removed_for_sft += next(
            (index for index, role in enumerate(source_roles) if role == "user"),
            len(source_roles),
        )
        trailing_user_sources += int(source_roles[-1] == "user")
        merged_adjacent_role_sources += int(any(left == right for left, right in zip(source_roles, source_roles[1:])))

    return {
        "records": 500,
        "model": config["model"],
        "source_sha256": sha256_file(source_path),
        "translated_sha256": sha256_file(translated_path),
        "sft_sha256": sha256_file(sft_path),
        "trailing_user_sources_normalized_for_sft": trailing_user_sources,
        "adjacent_same_role_sources_merged_for_sft": merged_adjacent_role_sources,
        "leading_assistant_turns_removed_for_sft": leading_assistant_turns_removed_for_sft,
        "validation": "all source hashes, IDs, API provenance, translated turn structures, and SFT message structures passed",
    }


def main() -> None:
    config = json.loads((BASE_DIR / "config.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "public-corpus-ja-translation-manifest-v1",
        "model": config["model"],
        "reasoning_effort": config["reasoning_effort"],
        "datasets": {dataset: validate_dataset(dataset, config) for dataset in pipeline.DATASETS},
    }
    path = BASE_DIR / "data" / "translation_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
