"""旧500対話から全件条件とresearch採択350件条件のSFT入力を作る。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
DEFAULT_CORPUS = REPO_ROOT / "pipelines" / "corpus_creation" / "500_empathetic_dialogues.jsonl"
DEFAULT_EVALUATED = REPO_ROOT / "research" / "model_evaluation" / "results" / "legacy_500" / "evaluated.jsonl"
DEFAULT_PROMPT = REPO_ROOT / "pipelines" / "sft" / "v2" / "prompts" / "v2_cot_teacher_system.txt"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--evaluated", type=Path, default=DEFAULT_EVALUATED)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "data")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_unique(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row.get(key, "")).strip()
        if not identifier or identifier in result:
            raise ValueError(f"欠損または重複した{key}: {identifier!r}")
        result[identifier] = row
    return result


def strict_research_pass(row: dict[str, Any]) -> bool:
    if row.get("evaluation_status") != "evaluated":
        return False
    evaluation = row["evaluation"]
    axes = evaluation["axes"]
    scores = [axis["score"] for axis in axes.values() if axis["score"] is not None]
    return (
        axes["mathematical_accuracy"]["score"] == 10
        and bool(scores)
        and all(score >= 8 for score in scores)
        and evaluation.get("critical_failure") is False
    )


def sanitize_legacy_text(value: Any) -> str:
    """ANSI装飾を除き、旧データの数式境界用制御文字を可視な$へ直す。"""
    text = ANSI_ESCAPE_RE.sub("", str(value))
    return "".join(
        "$" if ord(char) < 32 and char not in "\n\r\t" else char
        for char in text
    ).strip()


def teacher_content(turn: dict[str, Any], *, completed: bool) -> str:
    thought = sanitize_legacy_text(turn.get("thought_process", ""))
    emotion = sanitize_legacy_text(turn.get("student_emotion", ""))
    plan = sanitize_legacy_text(turn.get("next_step_plan", ""))
    final = sanitize_legacy_text(turn.get("content", ""))
    if not all((thought, emotion, plan, final)):
        raise ValueError("教師ターンのCoTフィールドまたは可視発話が空です")
    if completed and "[指導完了]" not in final:
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


def convert_session(session: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    source_id = str(session["source_id"])
    problem = sanitize_legacy_text(session.get("problem", ""))
    conversation = session.get("conversation")
    if not problem or not isinstance(conversation, list) or not conversation:
        raise ValueError(f"問題または対話が空です: {source_id}")
    if conversation[-1].get("role") != "teacher":
        raise ValueError(f"最終発話が教師ではありません: {source_id}")

    messages = [{"role": "system", "content": system_prompt}]
    for index, turn in enumerate(conversation):
        role = turn.get("role")
        if role == "student":
            content = sanitize_legacy_text(turn.get("content", ""))
            if index == 0:
                content = (
                    f"問題: {problem}\n\n"
                    "上記の問題を出題しました。生徒の発話を待機し、対応を開始してください。\n\n"
                    f"{content}"
                )
            messages.append({"role": "user", "content": content})
        elif role == "teacher":
            messages.append({
                "role": "assistant",
                "content": teacher_content(
                    turn,
                    completed=index == len(conversation) - 1 and session.get("is_completed") is True,
                ),
            })
        else:
            raise ValueError(f"不明なroleです: {source_id}: {role!r}")
    return {"id": source_id, "messages": messages}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    corpus_rows = read_jsonl(args.corpus)
    evaluated_rows = read_jsonl(args.evaluated)
    corpus = index_unique(corpus_rows, "source_id")
    evaluated = index_unique(evaluated_rows, "source_id")
    if set(corpus) != set(evaluated):
        raise ValueError("コーパスとresearch評価のsource_id集合が一致しません")
    if len(corpus) != 500:
        raise ValueError(f"元コーパスは500件である必要があります: {len(corpus)}")

    strict_ids = {source_id for source_id, row in evaluated.items() if strict_research_pass(row)}
    if len(strict_ids) != 350:
        raise ValueError(f"research採択数が350件ではありません: {len(strict_ids)}")
    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    affected_ids = sorted({
        source_id
        for source_id, session in corpus.items()
        if any(
            ord(char) < 32 and char not in "\n\r\t"
            for turn in session["conversation"]
            for value in turn.values()
            for char in str(value)
        ) or any(
            ANSI_ESCAPE_RE.search(str(value)) is not None
            for turn in session["conversation"]
            for value in turn.values()
        )
    })
    ordered_ids = sorted(corpus)
    random.Random(args.seed).shuffle(ordered_ids)
    all_rows = [convert_session(corpus[source_id], system_prompt) for source_id in ordered_ids]
    strict_rows = [row for row in all_rows if row["id"] in strict_ids]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "all_500": (args.output_dir / "legacy_all_500_sft.jsonl", all_rows),
        "research_350": (args.output_dir / "legacy_research_350_sft.jsonl", strict_rows),
    }
    datasets: dict[str, Any] = {}
    for name, (path, rows) in outputs.items():
        write_jsonl(path, rows)
        datasets[name] = {
            "path": path.name,
            "records": len(rows),
            "sha256": sha256_file(path),
            "source_ids_in_output_order": [row["id"] for row in rows],
        }
    manifest = {
        "schema_version": "legacy-500-vs-research-350-sft-manifest-v1",
        "corpus_sha256": sha256_file(args.corpus),
        "research_evaluation_sha256": sha256_file(args.evaluated),
        "system_prompt_sha256": sha256_file(args.system_prompt),
        "shuffle_seed": args.seed,
        "assistant_template": "<analysis>...</analysis><final>...</final>",
        "legacy_text_sanitization": {
            "ansi_escape_sequences": "removed",
            "non-whitespace_ascii_controls": "replaced_with_dollar_math_delimiter",
            "affected_records": len(affected_ids),
            "affected_source_ids": affected_ids,
        },
        "selection": {
            "all_500": "all source dialogues",
            "research_350": "mathematical_accuracy=10, every applicable axis>=8, critical_failure=false",
            "not_yet_gated": ["unresolved repair instruction", "evaluation-set leakage"],
        },
        "datasets": datasets,
    }
    manifest_path = args.output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"500件条件: {outputs['all_500'][0]}")
    print(f"350件条件: {outputs['research_350'][0]}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
