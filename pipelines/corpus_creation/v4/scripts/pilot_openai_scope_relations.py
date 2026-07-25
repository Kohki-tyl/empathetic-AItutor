"""OpenAI APIで4種類のscope_relationを各1件生成・監査する小規模パイロット。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = BASE_DIR / "run_v4.py"
SPEC = importlib.util.spec_from_file_location("run_v4_openai_pilot", MODULE_PATH)
assert SPEC and SPEC.loader
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)

RELATIONS = ("mastered", "frontier", "one_step_beyond", "far_beyond")
DEFAULT_SELECTION = BASE_DIR / "assignments" / "corpus_120_selection.json"
DEFAULT_OUTPUT = BASE_DIR / "data" / "openai_scope_pilot_v2" / "results.jsonl"


def select_examples(assignments: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in assignments[:limit]:
        relation = str(row["scope_relation"])
        if relation in RELATIONS and relation not in selected:
            selected[relation] = row
    missing = [relation for relation in RELATIONS if relation not in selected]
    if missing:
        raise ValueError(f"先頭{limit}件にscope_relationがありません: {missing}")
    return [selected[relation] for relation in RELATIONS]


def generate_student(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    state: dict[str, Any],
    profile: dict[str, Any],
    response_mode: str,
    required_initial_disclosure: str,
    attempts: int,
) -> tuple[dict[str, Any], list[str]]:
    validation_errors: list[str] = []
    retry_messages = list(messages)
    for _ in range(attempts):
        value = pipeline.chat_call(
            client,
            model,
            retry_messages,
            pipeline.STUDENT_SCHEMA,
            reasoning_effort="medium",
            max_completion_tokens=2500,
            retries=2,
        )
        try:
            return pipeline.validate_student_turn(
                value,
                state,
                allow_emotion_change=False,
                allowed_knowledge=profile["prior_knowledge"],
                expected_response_mode=response_mode,
                latest_teacher_utterance="",
                required_initial_disclosure=required_initial_disclosure,
            ), validation_errors
        except ValueError as exc:
            validation_errors.append(str(exc))
            retry_messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        f"前回出力は検証エラーでした: {exc}。"
                        "初期状態、知識境界、指定応答形式を守ってJSONを再生成してください。"
                        "far_beyondではrequired_initial_disclosureを発話の先頭へ完全一致で置いてください。"
                    ),
                },
            ]
    raise RuntimeError(f"student validation failed: {validation_errors}")


def generate_teacher(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    attempts: int,
) -> tuple[dict[str, Any], list[str]]:
    validation_errors: list[str] = []
    retry_messages = list(messages)
    for _ in range(attempts):
        value = pipeline.chat_call(
            client,
            model,
            retry_messages,
            pipeline.TEACHER_SCHEMA,
            reasoning_effort="high",
            max_completion_tokens=3000,
            retries=2,
        )
        try:
            return pipeline.validate_teacher_turn(value), validation_errors
        except ValueError as exc:
            validation_errors.append(str(exc))
            retry_messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        f"前回出力は検証エラーでした: {exc}。"
                        "数学的検証、完了判定、次の支援を整合させてJSONを再生成してください。"
                    ),
                },
            ]
    raise RuntimeError(f"teacher validation failed: {validation_errors}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--candidate-limit", type=int, default=120)
    parser.add_argument("--problem-selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"出力済みです。再実行には--overwriteが必要です: {args.output}")
    pipeline.load_env_file(BASE_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEYを設定してください")

    config = pipeline.load_config(BASE_DIR / "config.json")
    profiles = pipeline.read_json(BASE_DIR / "prompts" / "student_profiles.json")
    profile_by_id = {str(profile["id"]): profile for profile in profiles}
    emotion_rows = pipeline.read_json(BASE_DIR / "prompts" / "initial_emotions.json")["emotions"]
    emotion_by_name = {str(row["name"]): row for row in emotion_rows}
    questions = pipeline.ordered_math_questions(pipeline.read_jsonl(Path(config["questions"])))
    question_by_id = {
        pipeline.question_fields(question)[0]: question for question in questions
    }
    assignments = pipeline.load_problem_profile_assignments(
        Path(config["problem_profile_assignments"]), questions, profiles,
    )
    selected_pool = pipeline.load_problem_selection(
        args.problem_selection, assignments, int(config["max_candidates"]),
    )
    selected = select_examples(selected_pool, min(args.candidate_limit, len(selected_pool)))

    client = OpenAI(api_key=api_key)
    student_template = (BASE_DIR / "prompts" / "student_system.txt").read_text(encoding="utf-8")
    teacher_system = (BASE_DIR / "prompts" / "teacher_system.txt").read_text(encoding="utf-8")
    judge_system = (BASE_DIR / "prompts" / "turn_quality_judge_system.txt").read_text(encoding="utf-8")
    results: list[dict[str, Any]] = []

    for index, assignment in enumerate(selected):
        source_id = str(assignment["source_id"])
        _, problem, solution, metadata = pipeline.question_fields(question_by_id[source_id])
        profile = json.loads(json.dumps(profile_by_id[str(assignment["profile_id"])], ensure_ascii=False))
        profile["problem_epistemic_state"] = {
            "curriculum_annotation": assignment["curriculum_annotation"],
            "scope_relation": assignment["scope_relation"],
            "prior_attempt_history": assignment["prior_attempt_history"],
            "misconception_model": assignment["misconception_model"],
            "initial_response_constraint": assignment["initial_response_constraint"],
        }
        emotion = str(assignment["initial_emotion"])
        state = pipeline.initial_state(profile, emotion, assignment)
        response_mode = pipeline.effective_initial_response_mode(
            str(assignment["initial_response_mode"]), assignment,
        )
        student_payload = {
            "problem": problem,
            "turn": 0,
            "state_before": state,
            "initial_emotion": emotion,
            "initial_emotion_condition": emotion_by_name[emotion],
            "initial_response_condition": response_mode,
            "problem_level": pipeline.problem_level(metadata),
            "epistemic_state_specification": assignment,
            "knowledge_boundary": {
                "prior_knowledge": profile["prior_knowledge"],
                "acquired_knowledge": [],
                "unknown_knowledge": profile["unknown_knowledge"],
                "max_independent_math_level": profile["max_independent_math_level"],
            },
            "latest_teacher_utterance": "",
            "recent_dialogue": [],
            "instruction": "問題を解き始めてください",
        }
        student_messages = [
            {
                "role": "system",
                "content": student_template.replace("{STUDENT_PROFILE}", pipeline.profile_text(profile)),
            },
            {"role": "user", "content": json.dumps(student_payload, ensure_ascii=False)},
        ]
        student, student_errors = generate_student(
            client, args.model, student_messages, state, profile, response_mode,
            pipeline.required_initial_disclosure(assignment), args.attempts,
        )
        teacher_messages = [
            {"role": "system", "content": teacher_system},
            {
                "role": "user",
                "content": pipeline.teacher_turn_input(
                    problem=problem,
                    reference_solution=solution,
                    student_utterance=student["utterance"],
                    profile=profile,
                    epistemic_assignment=assignment,
                    initial_emotion=emotion,
                    turn_index=0,
                ),
            },
        ]
        teacher, teacher_errors = generate_teacher(
            client, args.model, teacher_messages, args.attempts,
        )
        student_turn = {
            "turn": 0,
            "role": "student",
            "content": student["utterance"],
            "response_stage": student["response_stage"],
            "knowledge_used": student["knowledge_used"],
            "state_after": student["state_after"],
            "state_update_reason": student["state_update_reason"],
        }
        teacher_turn = {"turn": 0, "role": "teacher", **teacher}
        dialogue = {
            "candidate_id": f"openai-scope-{index:02d}-{assignment['scope_relation']}",
            "source_id": source_id,
            "problem": problem,
            "reference_solution": solution,
            "student_profile": profile,
            "initial_emotion": emotion,
            "initial_student_state": state,
            "conversation": [student_turn, teacher_turn],
            "generation_condition": {
                "effective_initial_response_mode": response_mode,
                "problem_profile_assignment": assignment,
            },
            "models": {"student": args.model, "teacher": args.model, "judge": args.model},
        }
        audit = pipeline.chat_call(
            client,
            args.model,
            [
                {"role": "system", "content": judge_system},
                {
                    "role": "user",
                    "content": json.dumps(
                        pipeline.audit_payload(dialogue, 1, teacher_turn), ensure_ascii=False,
                    ),
                },
            ],
            pipeline.AUDIT_SCHEMA,
            reasoning_effort="high",
            max_completion_tokens=3500,
            retries=2,
        )
        dialogue["audit"] = audit
        dialogue["audit_decision"] = pipeline.classify_audit(audit)
        dialogue["validation"] = {
            "student_schema_and_constraints": True,
            "teacher_schema_and_constraints": True,
            "student_retry_errors": student_errors,
            "teacher_retry_errors": teacher_errors,
        }
        results.append(dialogue)
        print(json.dumps({
            "relation": assignment["scope_relation"],
            "source_id": source_id,
            "profile_id": assignment["profile_id"],
            "initial_emotion": emotion,
            "response_stage": student["response_stage"],
            "audit_decision": dialogue["audit_decision"],
        }, ensure_ascii=False))

    pipeline.write_jsonl(args.output, results)
    print(json.dumps({
        "output": str(args.output),
        "records": len(results),
        "decisions": {relation: next(
            row["audit_decision"] for row in results
            if row["generation_condition"]["problem_profile_assignment"]["scope_relation"] == relation
        ) for relation in RELATIONS},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
