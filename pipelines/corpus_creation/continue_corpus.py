"""Continue the legacy 800-dialogue corpus using the additional 500 translated questions."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import generate_corpus as legacy

BASE = Path(__file__).resolve().parent
CONDITIONS = [
    "あなたは現在【Frustrated】状態です。気分の悪さ、あるいはこの問題への強い苦手意識により、最初からイライラしており投げやりなトーンです。",
    "あなたは現在【Confusion】状態です。問題を見て少し戸惑っており、自信がなさそうに解き始めます。",
    "あなたは現在【Engaged】状態です。前向きに解く意欲があり、集中して取り組み始めます。",
]


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace(source, target):
    for attempt in range(10):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05 * (attempt + 1))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    replace(temp, path)


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    replace(temp, path)


def contexts(problem, profile, condition, conversation):
    # Keep the legacy prompt and conversational history construction exactly.
    formatted = (f"【学年】: {profile.get('grade', '不明')}\n"
                 f"【学習済みの範囲】: {profile.get('learned_scope', '不明')}\n"
                 f"【ミスしやすい点】: {profile.get('error_tendency', '不明')}\n"
                 f"【苦手な範囲】: {profile.get('weak_area', '不明')}")
    student_prompt = legacy.STUDENT_SYSTEM_TEMPLATE.format(
        TARGET_PROBLEM=problem, STUDENT_PROFILE=formatted, INITIAL_CONDITION=condition)
    teacher = [{"role": "system", "content": legacy.TEACHER_SYSTEM},
               {"role": "user", "content": f"問題: {problem}\n\n上記の問題を出題しました。生徒の最初の発話を待機し、対応を開始してください。"}]
    student = [{"role": "system", "content": student_prompt}]
    for message in conversation:
        if message["role"] == "student":
            teacher.append({"role": "user", "content": message["content"]})
            student.append({"role": "assistant", "content": message["content"]})
        else:
            teacher.append({"role": "assistant", "content": message["content"]})
            student.append({"role": "user", "content": message["content"]})
    return student, teacher


def validate_state(state, max_turns, final=False):
    messages = state["conversation"]
    if len(messages) > max_turns * 2:
        raise ValueError("Too many turns")
    for index, message in enumerate(messages):
        if message["role"] != ("student" if index % 2 == 0 else "teacher") or message["turn"] != index // 2:
            raise ValueError("Invalid role/turn sequence")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError("Empty utterance")
        if any(ord(c) < 32 and c not in "\n\r\t" for c in message["content"]):
            codes = sorted({f"U+{ord(c):04X}" for c in message["content"] if ord(c) < 32 and c not in "\n\r\t"})
            raise ValueError(f"Control character in utterance: role={message['role']}, codes={codes}, text={message['content']!r}")
    if final and (not messages or len(messages) % 2 or (not state["is_completed"] and len(messages) != max_turns * 2)):
        raise ValueError("API-interrupted dialogue cannot be finalized")


def repair_surface(raw):
    corrected = raw
    for character, letter, suffix in (
        ("\x0c", "f", r"rac|rak|orall|loor|box|lat"),
        ("\x08", "b", r"eta|egin|oxed|ig|inom|oldsymbol|oldface|ar|mod|reve|ullet|ecause|f\b|m\b"),
        ("\x0b", "v", r"arepsilon|arphi|artheta|arpi|arnothing|ert|ec|ee|space|rule|center|dots|phantom"),
        ("\x07", "a", r"lpha|ngle|pprox|rcsin|rccos|rctan|rray|rg\b|leph|malg"),
    ):
        corrected = re.sub(re.escape(character) + "(?=(?:" + suffix + "))", lambda _, key=letter: "\\" + key, corrected)
    corrected = re.sub(r"\t(?=(?:heta|imes|ext|frac|an|au|riangle)(?![a-z]))", lambda _: r"\t", corrected)
    corrected = re.sub(r"\r(?=(?:ight|angle)(?![a-z]))", lambda _: r"\r", corrected)
    corrected = re.sub(r"\n(?=(?:eq|otin|abla|eg)(?![a-z]))", lambda _: r"\n", corrected)
    # Earlier recovery may have exposed non-mathematical backspace/formfeed delimiters as \b/\f.
    def remove_exposed_delimiter(match):
        command = match[1]
        if command in {"a", "b", "f", "v"}:
            return ""
        if len(command) == 2 and command.startswith("b") and command not in {"bf", "bm"}:
            return command[1:]
        return match[0]
    corrected = re.sub(r"\\([A-Za-z]+)", remove_exposed_delimiter, corrected)
    corrected = re.sub(r"\\u00(?:0[0-8bcef]|1[0-9a-f])", "", corrected, flags=re.IGNORECASE)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", corrected)


def completion(client, messages, model, temperature, teacher, config):
    kwargs = {"model": model, "messages": messages, "temperature": temperature,
              "reasoning_effort": "none", "max_completion_tokens": config["max_completion_tokens"], "store": False}
    if teacher:
        kwargs["response_format"] = legacy.teacher_response_schema
    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    if choice.finish_reason != "stop" or not choice.message.content:
        raise ValueError(f"Incomplete/refused API response: {response.id}")
    content = choice.message.content.strip()
    surface_repairs = {}
    if teacher:
        value = json.loads(content)
        for key, raw in list(value.items()):
            if not isinstance(raw, str):
                continue
            corrected = repair_surface(raw)
            if corrected != raw:
                surface_repairs[key] = {"original_sha256": digest(raw), "method": "restore_JSON_decoded_LaTeX_escapes_and_remove_nonprinting_delimiters"}
                value[key] = corrected
        fields = legacy.teacher_response_schema["json_schema"]["schema"]["properties"]
        if set(value) != set(fields) or not isinstance(value["is_completed"], bool):
            raise ValueError("Teacher response schema mismatch")
        for key in fields:
            if key != "is_completed" and not isinstance(value[key], str):
                raise ValueError(f"Teacher field is not a string: {key}")
        if value["student_emotion"] not in fields["student_emotion"]["enum"] or not value["teacher_utterance"].strip():
            raise ValueError("Invalid teacher emotion/utterance")
    else:
        value = repair_surface(content)
        if value != content:
            surface_repairs["content"] = {"original_sha256": digest(content), "method": "restore_LaTeX_and_remove_nonprinting_delimiters"}
    metadata = {"role": "teacher" if teacher else "student", "response_id": response.id,
                "requested_model": model, "resolved_model": response.model,
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "usage": response.usage.model_dump() if response.usage else {}}
    if surface_repairs:
        metadata["surface_repairs"] = surface_repairs
    return value, metadata


def generate_one(client, question, offset, profiles, config, checkpoint):
    profile = profiles[offset % len(profiles)]
    condition = CONDITIONS[offset % len(CONDITIONS)]
    expected = {"source_id": question["id"], "question_sha256": digest(question),
                "profile": profile, "initial_condition": condition, "offset": offset,
                "run_fingerprint": config["run_fingerprint"]}
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state["identity"] != expected:
            raise ValueError(f"Checkpoint identity mismatch: {question['id']}")
    else:
        state = {"identity": expected, "conversation": [], "is_completed": False, "api_calls": [], "done": False}
    validate_state(state, config["max_turns"], final=state["done"])
    while not state["done"]:
        turn = len(state["conversation"]) // 2
        student, teacher = contexts(question["translated_question"], profile, condition, state["conversation"])
        if len(state["conversation"]) % 2 == 0:
            if turn == 0:
                student.append({"role": "user", "content": "それでは、提示された問題を解き始めてください。"})
            content, metadata = completion(client, student, config["student_model"], 0.8, False, config)
            state["conversation"].append({"turn": turn, "role": "student", "content": content})
            metadata["turn"] = turn
            state["api_calls"].append(metadata)
            validate_state(state, config["max_turns"])
            save_json(checkpoint, state)
            teacher.append({"role": "user", "content": content})
        value, metadata = completion(client, teacher, config["teacher_model"], 0.2, True, config)
        state["conversation"].append({"turn": turn, "role": "teacher",
            "thought_process": value["thought_process"], "student_emotion": value["student_emotion"],
            "roadmap_breakdown": value["roadmap_breakdown"], "next_step_plan": value["next_step_plan"],
            "content": value["teacher_utterance"]})
        metadata["turn"] = turn
        state["api_calls"].append(metadata)
        state["is_completed"] = value["is_completed"]
        state["done"] = value["is_completed"] or turn + 1 == config["max_turns"]
        validate_state(state, config["max_turns"], final=state["done"])
        save_json(checkpoint, state)
    validate_state(state, config["max_turns"], final=True)
    return {"problem": question["translated_question"], "student_profile": copy.deepcopy(profile),
            "is_completed": state["is_completed"], "conversation": state["conversation"], "source_id": question["id"],
            "generation_metadata": {"dialogue_number": offset + 1, "initial_condition": condition,
                                    "question_sha256": expected["question_sha256"],
                                    "run_fingerprint": config["run_fingerprint"], "api_calls": state["api_calls"]}}


def run(args, client=None):
    old, questions = read_rows(args.existing), read_rows(args.questions)
    profiles_path = BASE / "prompts/student_profile.json"
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    if len(old) != 800 or len(questions) != 500:
        raise ValueError("Expected 800 existing dialogues and 500 additional questions")
    ids = [q["id"] for q in questions]
    if any(not re.fullmatch(r"math_train_\d+", identifier) for identifier in ids):
        raise ValueError("Invalid question source ID")
    if len(set(ids)) != 500 or set(ids) & {r["source_id"] for r in old}:
        raise ValueError("Duplicate source IDs")
    if {q["translated_question"] for q in questions} & {r["problem"] for r in old}:
        raise ValueError("Duplicate problem text")
    config = {"existing_sha256": file_hash(args.existing), "questions_sha256": file_hash(args.questions),
              "teacher_model": args.teacher_model, "student_model": args.student_model,
              "teacher_temperature": 0.2, "student_temperature": 0.8, "reasoning_effort": "none",
              "max_turns": 15, "max_completion_tokens": 4096, "profile_offset": 800,
              "teacher_prompt_sha256": digest(legacy.TEACHER_SYSTEM),
              "student_prompt_sha256": digest(legacy.STUDENT_SYSTEM_TEMPLATE), "profiles_sha256": file_hash(profiles_path),
              "initial_conditions": CONDITIONS, "teacher_schema": legacy.teacher_response_schema}
    config["run_fingerprint"] = digest(config)
    if args.dry_run:
        print(json.dumps({"existing": len(old), "additional": len(questions), "config": config}, ensure_ascii=False, indent=2))
        return
    manifest_path = args.work_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config"] != config:
            raise ValueError("Resume config/input/prompt mismatch")
    else:
        manifest = {"config": config, "created_at": datetime.now(timezone.utc).isoformat(), "status": "running"}
        save_json(manifest_path, manifest)
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY"), timeout=180, max_retries=3)
    def task(item):
        index, question = item
        for attempt in range(5):
            try:
                return generate_one(client, question, 800 + index, profiles, config,
                                    args.work_dir / "checkpoints" / f"{question['id']}.json")
            except ValueError as exc:
                save_json(args.work_dir / "diagnostics" / f"{question['id']}_{attempt + 1}.json",
                          {"source_id": question["id"], "attempt": attempt + 1, "error": str(exc)})
                if attempt == 4:
                    raise
                time.sleep(1)
    results = []
    manifest.update(status="running", saved_in_source_order=0)
    manifest.pop("error_type", None)
    save_json(manifest_path, manifest)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Failures retain per-utterance checkpoints; running dialogues finish saving before exit.
            selected = questions[:args.limit] if args.limit else questions
            for result in executor.map(task, enumerate(selected)):
                results.append(result)
                manifest.update(status="running", saved_in_source_order=len(results))
                save_json(manifest_path, manifest)
                print(f"Generated {len(results)}/500: {result['source_id']}", flush=True)
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        save_json(manifest_path, manifest)
        raise
    if file_hash(args.existing) != config["existing_sha256"]:
        raise ValueError("Original 800 changed during generation")
    if len(results) < len(questions):
        manifest.update(status="pilot_completed", pilot_dialogues=len(results))
        save_json(manifest_path, manifest)
        return
    additional = args.work_dir / "additional_500_empathetic_dialogues_801_1300.jsonl"
    combined = args.work_dir / "1300_empathetic_dialogues.jsonl"
    save_rows(additional, results)
    save_rows(combined, old + results)
    manifest.update(status="completed", additional_dialogues=500, combined_dialogues=1300,
        completed_dialogues=sum(r["is_completed"] for r in results),
        max_turn_incomplete_dialogues=sum(not r["is_completed"] for r in results),
        additional_sha256=file_hash(additional), combined_sha256=file_hash(combined),
        review_status="generated_not_mathematically_audited")
    save_json(manifest_path, manifest)
    print(json.dumps({"additional": str(additional), "combined": str(combined), "completed": manifest["completed_dialogues"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, default=BASE / "800_empathetic_dialogues.jsonl")
    parser.add_argument("--questions", type=Path, default=BASE / "questions/translated_additional_500_math_gpt6_luna.jsonl")
    parser.add_argument("--work-dir", type=Path, default=Path("data/corpus_continuation_500"))
    parser.add_argument("--teacher-model", default="gpt-5.4")
    parser.add_argument("--student-model", default="gpt-5.4-mini")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, help="Pilot only; omit to complete all 500 using the same checkpoints")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and not 1 <= args.limit <= 500):
        parser.error("workers must be positive; limit must be between 1 and 500")
    run(args)


if __name__ == "__main__":
    main()
