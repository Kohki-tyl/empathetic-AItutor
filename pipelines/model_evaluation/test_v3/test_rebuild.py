from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generation = load_module("v3_generation", "generate_in_context_dialogues.py")
evaluation = load_module("v3_evaluation", "evaluate_in_context_dialogues.py")


class ImprovedV3Test(unittest.TestCase):
    def setUp(self) -> None:
        self.state = {"understanding_level": 1, "confidence": 0.5,
                      "active_misconception": "符号を混同する", "emotion": "confused",
                      "acquired_knowledge": [], "remaining_unknowns": ["移項"]}

    def test_unclosed_final_is_extracted(self) -> None:
        final, analysis, completed = generation.parse_teacher_response(
            "<analysis>内部推論</analysis><final>両辺から2を引こう。[指導完了]")
        self.assertEqual(final, "両辺から2を引こう。")
        self.assertEqual(analysis, "内部推論")
        self.assertTrue(completed)

    def test_analysis_without_final_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generation.parse_teacher_response("<analysis>内部推論だけ")

    def test_invalid_misconception_preserves_previous(self) -> None:
        normalized, notes = generation.validate_student_state(
            dict(self.state, active_misconception=None), self.state)
        self.assertEqual(normalized["active_misconception"], self.state["active_misconception"])
        self.assertEqual(notes, ["active_misconception:preserved_previous"])

    def test_nested_utterance_is_unwrapped(self) -> None:
        payload = {"state_after": self.state, "state_update_reason": "維持",
                   "utterance": json.dumps({"utterance": "ここが分かりません。"}, ensure_ascii=False)}
        parsed = generation.parse_student_turn(json.dumps(payload, ensure_ascii=False), self.state)
        self.assertEqual(parsed["utterance"], "ここが分かりません。")

    def test_failed_generation_is_retried(self) -> None:
        self.assertFalse(generation.generation_succeeded(
            {"run_id": "failed", "generation_error": "bad JSON", "phase1_turns": 2}))

    def test_only_complete_evaluation_is_skipped(self) -> None:
        row = {field: {"ok": True} for field in (
            "math_judge", "empathic_instruction_evaluation",
            "mathematical_instruction_evaluation", "student_realism_evaluation")}
        self.assertTrue(evaluation.evaluation_succeeded(row))
        row["math_judge"] = {"error": "timeout"}
        self.assertFalse(evaluation.evaluation_succeeded(row))


if __name__ == "__main__":
    unittest.main()
