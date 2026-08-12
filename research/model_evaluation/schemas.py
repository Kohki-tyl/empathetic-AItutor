from __future__ import annotations

from typing import Any

from common import AXES


STUDENT_UTTERANCE_SCHEMA: dict[str, Any] = {
    "name": "student_visible_utterance",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"utterance": {"type": "string"}},
        "required": ["utterance"],
        "additionalProperties": False,
    },
}

TRANSFER_ANSWER_SCHEMA: dict[str, Any] = {
    "name": "near_transfer_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"final_answer": {"type": "string"}},
        "required": ["final_answer"],
        "additionalProperties": False,
    },
}


def _axis_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "score": {
                "anyOf": [
                    {"type": "integer", "minimum": 0, "maximum": 10},
                    {"type": "null"},
                ]
            },
            "evidence": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["score", "evidence", "reason"],
        "additionalProperties": False,
    }


DIALOGUE_JUDGE_SCHEMA: dict[str, Any] = {
    "name": "teacher_dialogue_evaluation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "axes": {
                "type": "object",
                "properties": {name: _axis_schema() for name in AXES},
                "required": list(AXES),
                "additionalProperties": False,
            },
            "critical_failure_details": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "turn": {"type": "integer", "minimum": 0},
                        "category": {"type": "string"},
                        "evidence": {"type": "string"},
                        "impact": {"type": "string"},
                        "recovery_status": {"type": "string"},
                        "related_axes": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(AXES)},
                        },
                    },
                    "required": ["turn", "category", "evidence", "impact", "recovery_status", "related_axes"],
                    "additionalProperties": False,
                },
            },
            "instruction_completed": {"type": "boolean"},
            "completion_reason": {"type": "string"},
            "judge_summary": {"type": "string"},
        },
        "required": ["axes", "critical_failure_details", "instruction_completed", "completion_reason", "judge_summary"],
        "additionalProperties": False,
    },
}

TRANSFER_JUDGE_SCHEMA: dict[str, Any] = {
    "name": "near_transfer_math_judge",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "is_correct": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["is_correct", "reason"],
        "additionalProperties": False,
    },
}
