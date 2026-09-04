"""新規合成300件を加えた800対話から、未監査800件と監査採択500件のSFT入力を作る。"""

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
DEFAULT_CORPUS = REPO_ROOT / "pipelines" / "corpus_creation" / "800_empathetic_dialogues.jsonl"
DEFAULT_EVALUATED = REPO_ROOT / "research" / "model_evaluation" / "results" / "legacy_800" / "evaluated.jsonl"
DEFAULT_PROMPT = REPO_ROOT / "pipelines" / "sft" / "v2" / "prompts" / "v2_cot_teacher_system.txt"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--evaluated", type=Path, default=DEFAULT_EVALUATED)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audited-count", type=int, default=500)
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


def strict_audit_pass(row: dict[str, Any]) -> bool:
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
        "</analysis>\n<final>\n"
        f"{final}\n</final>"
    )


def convert_session(session: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    source_id = str(session["source_id"])
    problem = sanitize_legacy_text(session.get("problem", ""))
    raw_conversation = session.get("conversation")
    conversation = list(raw_conversation) if isinstance(raw_conversation, list) else raw_conversation
    if not problem or not isinstance(conversation, list) or not conversation:
        raise ValueError(f"問題または対話が空です: {source_id}")
    # SFT教師信号を持たない末尾の生徒発話だけを除き、対話レコード自体は保持する。
    if conversation[-1].get("role") == "student":
        conversation.pop()
    if not conversation or conversation[-1].get("role") != "teacher":
        raise ValueError(f"教師発話で終了する有効な対話がありません: {source_id}")

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
    corpus = index_unique(read_jsonl(args.corpus), "source_id")
    evaluated = index_unique(read_jsonl(args.evaluated), "source_id")
    if set(corpus) != set(evaluated):
        raise ValueError("コーパスと監査結果のsource_id集合が一致しません")
    if len(corpus) != 800:
        raise ValueError(f"元コーパスは800件である必要があります: {len(corpus)}")

    passing_ids = {source_id for source_id, row in evaluated.items() if strict_audit_pass(row)}
    if len(passing_ids) < args.audited_count:
        raise ValueError(f"監査通過数が抽出数未満です: {len(passing_ids)} < {args.audited_count}")

    ordered_ids = sorted(corpus)
    random.Random(args.seed).shuffle(ordered_ids)
    all_rows = [convert_session(corpus[source_id], args.system_prompt.read_text(encoding="utf-8").strip()) for source_id in ordered_ids]
    audited_rows = [row for row in all_rows if row["id"] in passing_ids][:args.audited_count]

    affected_ids = sorted({
        source_id for source_id, session in corpus.items()
        if any(
            (ord(char) < 32 and char not in "\n\r\t") or ANSI_ESCAPE_RE.search(str(value)) is not None
            for turn in session["conversation"] for value in turn.values() for char in str(value)
        )
    })
    truncated_trailing_student_ids = sorted(
        source_id for source_id, session in corpus.items()
        if session["conversation"][-1].get("role") == "student"
    )
    outputs = {
        "unaudited_800": (args.output_dir / "legacy_unaudited_800_sft.jsonl", all_rows),
        "audited_500": (args.output_dir / "legacy_audited_500_sft.jsonl", audited_rows),
    }
    datasets: dict[str, Any] = {}
    for name, (path, rows) in outputs.items():
        write_jsonl(path, rows)
        datasets[name] = {"path": path.name, "records": len(rows), "sha256": sha256_file(path), "source_ids_in_output_order": [row["id"] for row in rows]}

    manifest = {
        "schema_version": "legacy-800-vs-audited-500-sft-manifest-v1",
        "corpus_sha256": sha256_file(args.corpus),
        "audit_sha256": sha256_file(args.evaluated),
        "system_prompt_sha256": sha256_file(args.system_prompt),
        "shuffle_seed": args.seed,
        "audit_pass_records": len(passing_ids),
        "audited_sample_records": args.audited_count,
        "assistant_template": "<analysis>...</analysis><final>...</final>",
        "selection": {
            "unaudited_800": "all source dialogues; audit results are not used as a gate",
            "audited_500": "seeded sample from mathematical_accuracy=10, every applicable axis>=8, critical_failure=false",
            "sampling": "sort source_id, shuffle with Python random.Random(seed), retain passing records, take first N",
            "not_yet_gated": ["unresolved repair instruction", "evaluation-set leakage", "teacher internal CoT quality"],
        },
        "legacy_text_sanitization": {"ansi_escape_sequences": "removed", "non_whitespace_ascii_controls": "replaced_with_dollar_math_delimiter", "affected_records": len(affected_ids), "affected_source_ids": affected_ids},
        "structural_sanitization": {
            "trailing_unanswered_student_turn": "removed so every SFT record ends with an assistant target",
            "affected_records": len(truncated_trailing_student_ids),
            "affected_source_ids": truncated_trailing_student_ids,
        },
        "datasets": datasets,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"未監査800件条件: {outputs['unaudited_800'][0]}")
    print(f"監査通過{len(passing_ids)}件から抽出した500件条件: {outputs['audited_500'][0]}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
