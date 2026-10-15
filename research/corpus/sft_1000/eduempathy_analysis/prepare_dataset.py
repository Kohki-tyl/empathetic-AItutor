"""既存EduEmpathyの選択・split・本文を維持して保存済みanalysisを復元する。"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[3]
PLAIN = FOLDER.parent / "eduempathy"
CORPUS = ROOT / "research/corpus/empathetic_dialogues_1300/1300_empathetic_dialogues.jsonl"
PROMPT_SOURCE = ROOT / "pipelines/sft/v2/prompts/v2_cot_teacher_system.txt"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
FIELDS = ("thought_process", "student_emotion", "next_step_plan")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def normalize_analysis(value):
    # 800対話版と同じ規則。DELも除外する。数式の意味の復元は保証しない。
    if not isinstance(value, str) or not value.strip():
        raise ValueError("analysisフィールドが欠損または空です")
    text = ANSI.sub("", value)
    return "".join(
        "$" if (ord(char) < 32 and char not in "\n\r\t") or ord(char) == 127 else char
        for char in text
    ).strip()


def convert_record(row, session, prompt):
    result = copy.deepcopy(row)
    messages = result["messages"]
    turns = session["conversation"]
    if turns and turns[-1]["role"] == "student":
        turns = turns[:-1]
    expected_roles = ["system"] + [{"student": "user", "teacher": "assistant"}[t["role"]] for t in turns]
    if [m["role"] for m in messages] != expected_roles:
        raise ValueError(f"元コーパスとSFTのturn対応が不一致: {row['id']}")
    messages[0]["content"] = prompt
    changes = []
    for index, (message, turn) in enumerate(zip(messages[1:], turns)):
        if message["role"] != "assistant":
            continue
        values = {}
        for field in FIELDS:
            original = turn.get(field)
            values[field] = normalize_analysis(original)
            if values[field] != original:
                changes.append({"id": row["id"], "turn_index": index, "field": field,
                                "before": original, "after": values[field]})
        final = message["content"]
        if index == len(turns) - 1 and session.get("is_completed") is True and "[指導完了]" not in final:
            final += "\n[指導完了]"
        message["content"] = (
            f"<analysis>\n【認知状態】{values['thought_process']}\n"
            f"【感情状態】{values['student_emotion']}\n"
            f"【次の一歩】{values['next_step_plan']}\n</analysis>\n"
            f"<final>\n{final}\n</final>"
        )
    return result, changes


def main():
    plain = read_jsonl(PLAIN / "data.jsonl")
    source_rows = read_jsonl(CORPUS)
    sessions = {row["source_id"]: row for row in source_rows}
    if len(sessions) != len(source_rows):
        raise ValueError("元コーパスのsource_idが重複しています")
    prompt_path = FOLDER / "prompts/teacher_system.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_bytes(PROMPT_SOURCE.read_bytes())
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    rows, changes = [], []
    for row in plain:
        converted, edits = convert_record(row, sessions[row["group_id"]], prompt)
        rows.append(converted)
        changes.extend(edits)
    counts = Counter(row["split"] for row in rows)
    if len(rows) != 1000 or counts != {"train": 900, "validation": 100}:
        raise ValueError(f"想定外の件数・分割: {counts}")
    for filename, selected in (("data.jsonl", rows), ("train.jsonl", [r for r in rows if r["split"] == "train"]),
                               ("validation.jsonl", [r for r in rows if r["split"] == "validation"])):
        with (FOLDER / filename).open("w", encoding="utf-8", newline="\n") as stream:
            for row in selected:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    config_source = ROOT / "research/800_vs_audited_500/config.unaudited800.json"
    config = json.loads(config_source.read_text(encoding="utf-8"))
    config.update(dataset="data.jsonl", dataset_sha256=sha256(FOLDER / "data.jsonl"), expected_records=1000,
                  output_dir="outputs/eduempathy_ja1000_analysis", response_format="structured")
    write_json(FOLDER / "config.json", config)
    write_json(FOLDER / "analysis_surface_changes.json", changes)
    manifest = {
        "condition": "eduempathy_analysis", "records": len(rows), "split_counts": dict(counts),
        "assistant_turns": sum(m["role"] == "assistant" for r in rows for m in r["messages"]),
        "format": "<analysis>...</analysis><final>...</final>; assistant-only weighted loss",
        "provenance": {p.relative_to(ROOT).as_posix(): sha256(p) for p in
                       (CORPUS, PLAIN / "data.jsonl", PLAIN / "manifest.json", PLAIN / "surface_changes.json",
                        PLAIN / "quality_metadata.json", PROMPT_SOURCE, config_source)},
        "selection": "Same ordered IDs, group_id and explicit split as sibling eduempathy; no resampling",
        "analysis_fields": list(FIELDS), "roadmap_breakdown": "excluded, same as 800-dialogue SFT",
        "visible_content": "Preserve normalized EduEmpathy user/final text; append [指導完了] to completed last assistant turns, same as 800-dialogue SFT",
        "analysis_normalization": "Remove ANSI escapes; replace non-whitespace C0/DEL with dollar; log every changed field",
        "analysis_changed_fields": len(changes), "analysis_changed_records": len({c['id'] for c in changes}),
        "limitations": "Inherited surface-normalization limitations apply; analysis quality and normalized mathematical meaning have not been re-audited",
        "tokenizer_preflight": "not yet run; trainer refuses overlength dialogues",
        "train_ids": [r["id"] for r in rows if r["split"] == "train"],
        "validation_ids": [r["id"] for r in rows if r["split"] == "validation"],
        "files": {name: sha256(FOLDER / name) for name in
                  ("data.jsonl", "train.jsonl", "validation.jsonl", "config.json", "prompts/teacher_system.txt", "analysis_surface_changes.json")},
    }
    write_json(FOLDER / "manifest.json", manifest)
    print(f"records={len(rows)}, splits={dict(counts)}, assistant_turns={manifest['assistant_turns']}, changed_analysis_fields={len(changes)}")


if __name__ == "__main__":
    main()
