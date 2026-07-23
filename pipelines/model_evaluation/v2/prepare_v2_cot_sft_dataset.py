"""Keep-only対話から教師CoTを含むv2 SFTデータを作成する。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]
DEFAULT_CORPUS = REPO_ROOT / "pipelines" / "corpus_creation" / "500_empathetic_dialogues.jsonl"
DEFAULT_EVALUATIONS = REPO_ROOT / "pipelines" / "corpus_creation" / "500_dialogue_evaluations.jsonl"
DEFAULT_OUTPUT = BASE_DIR / "data" / "v2_keep_only_cot_sft_train.jsonl"
DEFAULT_MANIFEST = BASE_DIR / "data" / "v2_keep_only_cot_sft_manifest.json"
DEFAULT_SYSTEM_PROMPT = BASE_DIR / "prompts" / "v2_cot_teacher_system.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoTを含むv2 Keep-only SFTデータを作成する")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--evaluations", type=Path, default=DEFAULT_EVALUATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def index_unique(rows: list[dict], key: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in rows:
        row_id = row.get(key)
        if not row_id or row_id in result:
            raise ValueError(f"欠損または重複した{key}: {row_id}")
        result[row_id] = row
    return result


def teacher_content(turn: dict, is_last: bool) -> str:
    thought = str(turn.get("thought_process", "")).strip()
    emotion = str(turn.get("student_emotion", "")).strip()
    plan = str(turn.get("next_step_plan", "")).strip()
    final = str(turn.get("content", "")).strip()
    if not all((thought, emotion, plan, final)):
        raise ValueError("教師ターンのCoTフィールドまたは発話が空です。")
    if is_last and "[指導完了]" not in final:
        final += "\n[指導完了]"
    return (
        "<analysis>\n"
        f"【認知状態】{thought}\n"
        f"【感情状態】{emotion}\n"
        f"【次の一歩】{plan}\n"
        "</analysis>\n"
        "<final>\n"
        f"{final}\n"
        "</final>"
    )


def convert_session(session: dict, system_prompt: str) -> dict:
    problem = str(session.get("problem", "")).strip()
    conversation = session.get("conversation", [])
    if not problem or not conversation or conversation[-1].get("role") != "teacher":
        raise ValueError(f"不完全な対話: {session.get('source_id')}")

    messages = [{"role": "system", "content": system_prompt}]
    last_index = len(conversation) - 1
    for index, turn in enumerate(conversation):
        if turn.get("role") == "student":
            content = str(turn.get("content", "")).strip()
            if index == 0:
                content = (
                    f"問題: {problem}\n\n"
                    "上記の問題を出題しました。生徒の発話を待機し、対応を開始してください。\n\n"
                    f"{content}"
                )
            messages.append({"role": "user", "content": content})
        elif turn.get("role") == "teacher":
            messages.append({"role": "assistant", "content": teacher_content(turn, index == last_index)})
        else:
            raise ValueError(f"不明なrole: {turn.get('role')}")
    return {"messages": messages}


def main() -> None:
    args = parse_args()
    corpus_rows = load_jsonl(args.corpus)
    evaluation_rows = load_jsonl(args.evaluations)
    corpus = index_unique(corpus_rows, "source_id")
    evaluations = index_unique(evaluation_rows, "source_id")
    if set(corpus) - set(evaluations):
        raise ValueError("未評価の対話があります。")

    keep_ids = sorted(
        source_id
        for source_id, evaluation in evaluations.items()
        if evaluation.get("status") == "completed"
        and evaluation.get("evaluation", {}).get("recommendation") == "keep"
        and corpus[source_id].get("is_completed") is True
    )
    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    records = [{"source_id": source_id, "record": convert_session(corpus[source_id], system_prompt)} for source_id in keep_ids]
    random.Random(args.seed).shuffle(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for item in records:
            stream.write(json.dumps(item["record"], ensure_ascii=False) + "\n")

    manifest = {
        "dataset_name": "v2_keep_only_cot_sft",
        "format": {
            "analysis_fields": ["thought_process", "student_emotion", "next_step_plan"],
            "assistant_template": "<analysis>...</analysis><final>...</final>",
        },
        "filter": {"evaluation_status": "completed", "recommendation": "keep", "is_completed": True},
        "shuffle_seed": args.seed,
        "selected_count": len(records),
        "source_ids_in_output_order": [item["source_id"] for item in records],
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"CoT付きv2 Keep-only SFTデータ: {len(records)}件")
    print(f"学習データ: {args.output}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
