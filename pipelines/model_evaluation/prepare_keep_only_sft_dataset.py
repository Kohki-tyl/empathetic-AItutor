import argparse
import json
import random
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
DEFAULT_CORPUS = REPO_ROOT / "pipelines" / "corpus_creation" / "500_empathetic_dialogues.jsonl"
DEFAULT_EVALUATIONS = REPO_ROOT / "pipelines" / "corpus_creation" / "500_dialogue_evaluations.jsonl"
DEFAULT_OUTPUT = BASE_DIR / "v2_keep_only_sft_train.jsonl"
DEFAULT_MANIFEST = BASE_DIR / "v2_keep_only_sft_manifest.json"
DEFAULT_SYSTEM_PROMPT = BASE_DIR / "prompts" / "sft_teacher_system.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v2 Keep-only SFTデータセットを作成します。")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--evaluations", type=Path, default=DEFAULT_EVALUATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def index_unique(items: list[dict], key: str, label: str) -> dict[str, dict]:
    indexed = {}
    for item in items:
        item_id = item.get(key)
        if not item_id:
            raise ValueError(f"{label}に{key}がないレコードがあります。")
        if item_id in indexed:
            raise ValueError(f"{label}で{key}が重複しています: {item_id}")
        indexed[item_id] = item
    return indexed


def convert_session(session: dict, system_prompt: str) -> dict:
    problem = session.get("problem", "").strip()
    conversation = session.get("conversation", [])
    if not problem or not conversation:
        raise ValueError(f"問題文または対話が空です: {session.get('source_id')}")

    messages = [{"role": "system", "content": system_prompt}]
    for index, turn in enumerate(conversation):
        role = "user" if turn.get("role") == "student" else "assistant"
        content = turn.get("content", "").strip()
        if not content:
            raise ValueError(f"空の発話があります: {session.get('source_id')} / turn={index}")

        if role == "user" and index == 0:
            content = (
                f"問題: {problem}\n\n"
                "上記の問題を出題しました。生徒の発話を待機し、対応を開始してください。\n\n"
                f"{content}"
            )
        elif role == "assistant" and index == len(conversation) - 1:
            if "[指導完了]" not in content:
                content += "\n\n[指導完了]"

        messages.append({"role": role, "content": content})

    if messages[-1]["role"] != "assistant":
        raise ValueError(f"最後の発話が教師ではありません: {session.get('source_id')}")
    return {"messages": messages}


def main() -> None:
    args = parse_args()
    if not args.system_prompt.exists():
        raise FileNotFoundError(f"システムプロンプトが見つかりません: {args.system_prompt}")

    corpus_rows = load_jsonl(args.corpus)
    evaluation_rows = load_jsonl(args.evaluations)
    corpus = index_unique(corpus_rows, "source_id", "コーパス")
    evaluations = index_unique(evaluation_rows, "source_id", "評価結果")

    missing_evaluations = sorted(set(corpus) - set(evaluations))
    if missing_evaluations:
        raise ValueError(f"未評価の対話が{len(missing_evaluations)}件あります。")

    keep_ids = sorted(
        source_id
        for source_id, evaluation_row in evaluations.items()
        if evaluation_row.get("status") == "completed"
        and evaluation_row.get("evaluation", {}).get("recommendation") == "keep"
        and corpus.get(source_id, {}).get("is_completed") is True
    )

    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    records = [
        {"source_id": source_id, "record": convert_session(corpus[source_id], system_prompt)}
        for source_id in keep_ids
    ]
    random.Random(args.seed).shuffle(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item["record"], ensure_ascii=False) + "\n")

    manifest = {
        "dataset_name": "v2_keep_only_sft",
        "corpus_path": str(args.corpus.resolve()),
        "evaluations_path": str(args.evaluations.resolve()),
        "system_prompt_path": str(args.system_prompt.resolve()),
        "filter": {
            "evaluation_status": "completed",
            "recommendation": "keep",
            "is_completed": True,
        },
        "shuffle_seed": args.seed,
        "source_count": len(corpus_rows),
        "evaluation_count": len(evaluation_rows),
        "selected_count": len(records),
        "source_ids_in_output_order": [item["source_id"] for item in records],
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("v2 Keep-only SFTデータセットを作成しました。")
    print(f" - 元対話: {len(corpus_rows)}件")
    print(f" - 評価結果: {len(evaluation_rows)}件")
    print(f" - Keep-only採用: {len(records)}件")
    print(f" - 学習データ: {args.output}")
    print(f" - Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
