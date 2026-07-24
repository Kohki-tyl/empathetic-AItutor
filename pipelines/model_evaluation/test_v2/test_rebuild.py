from __future__ import annotations

import importlib.util
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


generation = load_module("v2_generation", "generate_profile_update_dialogues.py")
evaluation = load_module("v2_evaluation", "evaluate_profile_update_dialogues.py")
validation = load_module("v2_validation", "prepare_validated_questions.py")


class RebuiltV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.state = {
            "understanding_level": 1,
            "confidence": 0.5,
            "active_misconception": "符号を混同する",
            "emotion": "confused",
            "acquired_knowledge": [],
            "remaining_unknowns": ["移項"],
        }

    def test_unclosed_final_is_safely_extracted(self) -> None:
        final, analysis, completed = generation.parse_teacher_response(
            "<analysis>内部推論</analysis><final>まず両辺から2を引こう。[指導完了]"
        )
        self.assertEqual(final, "まず両辺から2を引こう。")
        self.assertEqual(analysis, "内部推論")
        self.assertTrue(completed)

    def test_first_teacher_user_message_combines_problem_and_student(self) -> None:
        first = generation.teacher_user_message("1+1は？", "2だと思います。", 0)
        later = generation.teacher_user_message("1+1は？", "理由も説明します。", 1)
        self.assertEqual(first, "問題: 1+1は？\n\n生徒発話: 2だと思います。")
        self.assertEqual(later, "理由も説明します。")

    def test_analysis_without_final_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generation.parse_teacher_response("<analysis>内部推論だけ")

    def test_student_json_leak_is_rejected(self) -> None:
        payload = {
            "state_after": self.state,
            "state_update_reason": "説明を一段階理解した",
            "utterance": '{"problem":"hidden state"}',
        }
        import json
        with self.assertRaises(ValueError):
            generation.parse_student_turn(json.dumps(payload, ensure_ascii=False), self.state)

    def test_extra_fields_are_normalized(self) -> None:
        import json
        payload = {
            "problem": "input echo",
            "state_after": dict(self.state, current_task="移項する"),
            "state_update_reason": "説明を一段階理解した",
            "utterance": "まず2を引けばよいですか？",
        }
        parsed = generation.parse_student_turn(json.dumps(payload, ensure_ascii=False), self.state)
        self.assertEqual(
            set(parsed),
            {"state_after", "state_update_reason", "utterance", "_state_normalizations"},
        )
        self.assertNotIn("current_task", parsed["state_after"])

    def test_nested_utterance_is_unwrapped(self) -> None:
        import json
        nested = json.dumps({"problem": "hidden", "utterance": "ここが分かりません。"}, ensure_ascii=False)
        payload = {
            "state_after": self.state,
            "state_update_reason": "状態は維持した",
            "utterance": nested,
        }
        parsed = generation.parse_student_turn(json.dumps(payload, ensure_ascii=False), self.state)
        self.assertEqual(parsed["utterance"], "ここが分かりません。")

    def test_invalid_state_jump_is_rejected(self) -> None:
        changed = dict(self.state, understanding_level=4)
        with self.assertRaises(ValueError):
            generation.validate_student_state(changed, self.state)

    def test_invalid_misconception_preserves_previous(self) -> None:
        changed = dict(self.state, active_misconception=None)
        normalized, notes = generation.validate_student_state(changed, self.state)
        self.assertEqual(normalized["active_misconception"], self.state["active_misconception"])
        self.assertEqual(notes, ["active_misconception:preserved_previous"])

    def test_only_successful_generation_is_skipped_on_resume(self) -> None:
        self.assertTrue(generation.generation_succeeded({
            "run_id": "ok", "generation_error": None, "phase1_turns": 1,
        }))
        self.assertFalse(generation.generation_succeeded({
            "run_id": "failed", "generation_error": "bad JSON", "phase1_turns": 3,
        }))

    def test_only_complete_judging_is_resumable(self) -> None:
        row = {
            "math_judge": {"is_correct": True},
            "empathic_instruction_evaluation": {"total_score": 20},
            "mathematical_instruction_evaluation": {"total_score": 40},
            "student_realism_evaluation": {"realism_score": 8},
        }
        self.assertTrue(evaluation.evaluation_succeeded(row))
        row["math_judge"] = {"error": "connection failed"}
        self.assertFalse(evaluation.evaluation_succeeded(row))

    def test_problem_rewrite_is_excluded(self) -> None:
        original = {"translated_question": "問題", "translated_solution": "解答"}
        similar = {
            "original_question": "問題", "similar_question": "類似問題",
            "similar_solution": "元の問題と同じ構造にならない。修正後の類似問題。\\boxed{1}",
        }
        self.assertIn("solution_rejects_structure", validation.pair_issues(original, similar))


if __name__ == "__main__":
    unittest.main()
