"""抽出済み対話をOpenAI APIで日本語へ翻訳し、独立SFT用JSONLを作る。"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
DATASETS = ("empathetic_dialogues", "mathdial")
SYSTEM_PROMPTS = {
    "empathetic_dialogues": BASE_DIR / "prompts" / "translate_empathetic_system.txt",
    "mathdial": BASE_DIR / "prompts" / "translate_mathdial_system.txt",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'“”‘’"))


def schema_for(row: dict[str, Any], dataset: str) -> dict[str, Any]:
    turn_properties: dict[str, Any] = {
        "role": {"type": "string", "enum": ["user", "assistant"]},
        "content": {"type": "string", "minLength": 1},
    }
    required = ["role", "content"]
    if dataset == "mathdial":
        turn_properties["dialogue_act"] = {"type": "string"}
        required.append("dialogue_act")
    properties: dict[str, Any] = {
        "id": {"type": "string", "const": row["id"]},
        "turns": {"type": "array", "minItems": len(row["turns"]), "maxItems": len(row["turns"]), "items": {"type": "object", "additionalProperties": False, "properties": turn_properties, "required": required}},
    }
    fields = ["emotion", "situation"] if dataset == "empathetic_dialogues" else ["question", "ground_truth", "student_incorrect_solution", "student_profile", "teacher_described_confusion"]
    for field in fields:
        properties[field] = {"type": "string", "minLength": 1}
    return {"type": "json_schema", "json_schema": {"name": f"{dataset}_ja_translation", "strict": True, "schema": {"type": "object", "additionalProperties": False, "properties": properties, "required": ["id", *fields, "turns"]}}}


def translation_payload(row: dict[str, Any], dataset: str) -> dict[str, Any]:
    fields = ["id", "emotion", "situation", "turns"] if dataset == "empathetic_dialogues" else ["id", "question", "ground_truth", "student_incorrect_solution", "student_profile", "teacher_described_confusion", "turns"]
    clean = {field: row[field] for field in fields}
    if dataset == "mathdial":
        clean["turns"] = [{key: turn[key] for key in ("role", "dialogue_act", "content")} for turn in row["turns"]]
    return clean


def validate_translation(source: dict[str, Any], translated: dict[str, Any], dataset: str) -> None:
    if translated.get("id") != source["id"]:
        raise ValueError("idが一致しません")
    source_turns, target_turns = source["turns"], translated.get("turns")
    if not isinstance(target_turns, list) or len(target_turns) != len(source_turns):
        raise ValueError("turn数が一致しません")
    for index, (original, target) in enumerate(zip(source_turns, target_turns)):
        if target.get("role") != original["role"] or not str(target.get("content", "")).strip():
            raise ValueError(f"turn {index}のroleまたはcontentが不正です")
        if dataset == "mathdial" and target.get("dialogue_act") != original["dialogue_act"]:
            raise ValueError(f"turn {index}のdialogue_actが一致しません")


def translate_one(client: OpenAI, row: dict[str, Any], dataset: str, config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    prompt = SYSTEM_PROMPTS[dataset].read_text(encoding="utf-8").strip()
    last_error: Exception | None = None
    for attempt in range(int(config["request_retries"])):
        try:
            response = client.chat.completions.create(
                model=config["model"], reasoning_effort=config["reasoning_effort"],
                max_completion_tokens=int(config["max_completion_tokens"]),
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(translation_payload(row, dataset), ensure_ascii=False)}],
                response_format=schema_for(row, dataset),
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("空のAPI応答です")
            translated = json.loads(content)
            validate_translation(row, translated, dataset)
            return translated, response.id
        except Exception as exc:
            last_error = exc
            if attempt + 1 < int(config["request_retries"]):
                time.sleep(float(config["retry_base_seconds"]) * (2 ** attempt) + random.random())
    raise RuntimeError(f"翻訳に失敗しました: {row['id']}: {last_error}") from last_error


def sft_row(source: dict[str, Any], translated: dict[str, Any], dataset: str) -> dict[str, Any]:
    if dataset == "empathetic_dialogues":
        system = "あなたは相手の感情と状況を丁寧に受け止め、自然で共感的な日本語で応答する対話相手です。"
        messages = [{"role": "system", "content": system}, *translated["turns"]]
        if messages[-1]["role"] == "user":
            messages.pop()
    else:
        system = (
            "あなたは、生徒の誤解を診断し、答えを性急に明かさず、段階的な問いかけで数学的理解を支援する教師です。\n\n"
            f"問題: {translated['question']}\n\n正答・解法: {translated['ground_truth']}\n\n"
            f"生徒プロフィール: {translated['student_profile']}\n教師が記録した混乱: {translated['teacher_described_confusion']}\n\n"
            f"生徒の初期解答（教師向け事前情報）: {translated['student_incorrect_solution']}"
        )
        first_user_index = next(
            (index for index, turn in enumerate(translated["turns"]) if turn["role"] == "user"),
            None,
        )
        if first_user_index is None:
            raise ValueError(f"MathDialに生徒発話がありません: {source['id']}")
        messages = [{"role": "system", "content": system}]
        # MathDial収録時の会話開始用teacher発話は落とし、実際のstudent発話からSFT文脈を開始する。
        for turn in translated["turns"][first_user_index:]:
            item = {"role": turn["role"], "content": turn["content"]}
            if messages[-1]["role"] == item["role"]:
                messages[-1]["content"] += "\n\n" + item["content"]
            else:
                messages.append(item)
        if messages[-1]["role"] == "user":
            messages.pop()
    return {"id": source["id"], "messages": messages, "source_dataset": dataset}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, help="並列APIリクエスト数（既定はconfig.json）")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    source_path = BASE_DIR / config[args.dataset]["sample"]
    output_path = BASE_DIR / config[args.dataset]["translated"]
    sft_path = BASE_DIR / config[args.dataset]["sft"]
    sources = read_jsonl(source_path)
    completed = {row["id"]: row for row in read_jsonl(output_path)}
    pending = [row for row in sources if row["id"] not in completed]
    if args.limit is not None:
        pending = pending[:args.limit]
    print(json.dumps({"dataset": args.dataset, "model": config["model"], "source_records": len(sources), "completed": len(completed), "pending_this_run": len(pending), "output": str(output_path)}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    load_env(BASE_DIR.parents[1] / ".env")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEYまたはGPT_API_KEYを設定してください")
    workers = args.workers or int(config.get("translation_workers", 1))
    if workers < 1:
        raise ValueError("--workersは1以上にしてください")
    client = OpenAI(api_key=api_key)

    def run(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        translated, response_id = translate_one(client, row, args.dataset, config)
        record = {"id": row["id"], "source_sha256": sha256_text(json.dumps(row, ensure_ascii=False, sort_keys=True)), "model": config["model"], "response_id": response_id, "translation": translated}
        return row, record

    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run, row): row for row in pending}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            source_row = futures[future]
            try:
                _, record = future.result()
            except Exception as exc:
                failures.append(f"{source_row['id']}: {exc}")
                print(f"[{index}/{len(pending)}] ERROR {source_row['id']}: {exc}")
                continue
            append_jsonl(output_path, record)
            completed[source_row["id"]] = record
            print(f"[{index}/{len(pending)}] {source_row['id']}")
    ordered_completed = [completed[row["id"]] for row in sources if row["id"] in completed]
    with sft_path.open("w", encoding="utf-8", newline="\n") as stream:
        for source, record in zip((row for row in sources if row["id"] in completed), ordered_completed):
            stream.write(json.dumps(sft_row(source, record["translation"], args.dataset), ensure_ascii=False) + "\n")
    print(f"SFT: {sft_path} ({len(ordered_completed)}件)")
    if failures:
        raise RuntimeError(f"{len(failures)}件が未完了です。再実行してください。先頭: {failures[0]}")


if __name__ == "__main__":
    main()
