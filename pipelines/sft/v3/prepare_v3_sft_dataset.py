"""ターン監査・修正済みv3コーパスから単一のCoT付きSFTデータを作成する。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]
DEFAULT_CORPUS = REPO_ROOT / "pipelines" / "corpus_creation" / "v3" / "data" / "v3_rebuilt_corpus.jsonl"
DEFAULT_OUTPUT = BASE_DIR / "data" / "v3_cot_sft.jsonl"
DEFAULT_MANIFEST = BASE_DIR / "data" / "v3_cot_sft_manifest.json"
DEFAULT_PROMPT = BASE_DIR / "prompts" / "v3_cot_teacher_system.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="単一のv3 CoT SFTデータを作成する")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def teacher_content(turn: dict[str, Any]) -> str:
    thought = str(turn.get("thought_process", "")).strip()
    emotion = str(turn.get("student_emotion", "")).strip()
    verification = str(turn.get("v3_audit", {}).get("mathematical_verification", "")).strip()
    plan = str(turn.get("next_step_plan", "")).strip()
    final = str(turn.get("content", "")).strip()
    if not all((thought, emotion, verification, plan, final)):
        raise ValueError("教師ターンの必須フィールドが空です。")
    return (
        "<analysis>\n"
        f"【認知状態】{thought}\n"
        f"【感情状態】{emotion}\n"
        f"【数学的検証】{verification}\n"
        f"【次の一歩】{plan}\n"
        "</analysis>\n"
        "<final>\n"
        f"{final}\n"
        "</final>"
    )


def convert(session: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    problem = str(session.get("problem", "")).strip()
    conversation = session.get("conversation", [])
    if not problem or not conversation or conversation[-1].get("role") != "teacher":
        raise ValueError(f"不完全な対話: {session.get('source_id')}")
    messages = [{"role": "system", "content": system_prompt}]
    for index, turn in enumerate(conversation):
        role = turn.get("role")
        if role == "student":
            content = str(turn.get("content", "")).strip()
            if not content:
                raise ValueError(f"空の生徒発話: {session.get('source_id')}")
            if index == 0:
                content = f"問題: {problem}\n\n{content}"
            messages.append({"role": "user", "content": content})
        elif role == "teacher":
            content = teacher_content(turn)
            if index == len(conversation) - 1 and session.get("is_completed"):
                content = content.replace("\n</final>", "\n[指導完了]\n</final>")
            messages.append({"role": "assistant", "content": content})
        else:
            raise ValueError(f"不明なrole: {role}")
    return {"messages": messages}


def main() -> None:
    args = parse_args()
    prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    sessions = read_jsonl(args.corpus)
    selected = [row for row in sessions if row.get("is_completed") is True]
    records = []
    seen = set()
    for session in selected:
        source_id = str(session.get("source_id", ""))
        if not source_id or source_id in seen:
            raise ValueError(f"欠損または重複したsource_id: {source_id}")
        seen.add(source_id)
        records.append({"source_id": source_id, "record": convert(session, prompt)})
    random.Random(args.seed).shuffle(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for item in records:
            stream.write(json.dumps(item["record"], ensure_ascii=False) + "\n")
    manifest = {
        "dataset_name": "v3_turn_audited_cot_sft",
        "source": str(args.corpus),
        "filter": {"is_completed": True, "all_teacher_turns_audited": True},
        "split": "not_applied; split at SFT runtime",
        "shuffle_seed": args.seed,
        "selected_count": len(records),
        "source_ids_in_output_order": [item["source_id"] for item in records],
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"dataset: {len(records)} -> {args.output}")


if __name__ == "__main__":
    main()
