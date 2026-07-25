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


generation = load_module("v4_test_generation", "generate_in_context_dialogues.py")
evaluation = load_module("v4_test_evaluation", "evaluate_in_context_dialogues.py")


class V4StudentAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = {
            "id": "V4-S01",
            "ability_level": 2,
            "max_independent_math_level": 2,
            "prior_knowledge": ["一次方程式"],
            "unknown_knowledge": ["連立方程式"],
            "target_misconception": "表面的な手順に頼る",
            "confidence_bias": "underconfident",
        }
        self.state = generation.initial_state(self.profile, "confused")

    def test_assigned_initial_emotion_sets_initial_state(self) -> None:
        self.assertEqual(self.state["emotion"], "confused")
        self.assertEqual(self.state["confidence"], 0.35)

    def test_initial_emotion_cannot_change_before_teacher_intervention(self) -> None:
        changed = dict(self.state, emotion="engaged")
        with self.assertRaisesRegex(ValueError, "initial emotion"):
            generation.validate_student_state(
                changed, self.state, allow_emotion_change=False,
            )

    def test_initial_knowledge_cannot_change_before_teacher_intervention(self) -> None:
        changed = dict(self.state, acquired_knowledge=["new knowledge"])
        with self.assertRaisesRegex(ValueError, "initial acquired knowledge"):
            generation.validate_student_state(
                changed, self.state, allow_emotion_change=False,
            )

    def test_initial_confidence_change_is_limited(self) -> None:
        changed = dict(self.state, confidence=self.state["confidence"] + 0.15)
        with self.assertRaisesRegex(ValueError, "initial confidence"):
            generation.validate_student_state(
                changed, self.state, allow_emotion_change=False,
            )

    def test_allowed_emotion_transition_is_accepted(self) -> None:
        changed = dict(self.state, emotion="engaged", confidence=0.55)
        normalized, _ = generation.validate_student_state(changed, self.state)
        self.assertEqual(normalized["emotion"], "engaged")

    def test_skipped_emotion_transition_is_rejected(self) -> None:
        changed = dict(self.state, emotion="proud")
        with self.assertRaisesRegex(ValueError, "permitted cycle"):
            generation.validate_student_state(changed, self.state)

    def test_large_confidence_change_is_rejected(self) -> None:
        changed = dict(self.state, confidence=0.7)
        with self.assertRaisesRegex(ValueError, "more than 0.25"):
            generation.validate_student_state(changed, self.state)

    def test_acquired_knowledge_cannot_be_removed(self) -> None:
        previous = dict(self.state, acquired_knowledge=["移項"])
        changed = dict(previous, acquired_knowledge=[])
        with self.assertRaisesRegex(ValueError, "was removed"):
            generation.validate_student_state(changed, previous)

    def test_corpus_prompts_and_profiles_are_synced(self) -> None:
        corpus_prompt_dir = BASE_DIR.parent.parent / "corpus_creation" / "v4" / "prompts"
        if not corpus_prompt_dir.exists():
            self.skipTest("コーパス作成フォルダーを同時にコピーしていない環境")
        pairs = {
            "student_profiles.json": "student_profiles.json",
            "initial_emotions.json": "initial_emotions.json",
            "student_system.txt": "student_system.txt",
            "sft_teacher_system.txt": "teacher_system.txt",
        }
        for corpus_name, test_name in pairs.items():
            corpus_text = (corpus_prompt_dir / corpus_name).read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n")
            test_text = (BASE_DIR / "prompts" / test_name).read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n")
            if corpus_name.endswith(".json"):
                self.assertEqual(json.loads(corpus_text), json.loads(test_text), f"同期が必要です: {test_name}")
            else:
                self.assertEqual(corpus_text, test_text, f"同期が必要です: {test_name}")

    def test_out_of_scope_problem_forces_scope_limited_response(self) -> None:
        self.assertEqual(
            generation.initial_response_condition({"scope_relation": "one_step_beyond"}),
            "scope_limited_help_seeking",
        )

    def test_in_scope_problem_uses_preassigned_response_mode(self) -> None:
        self.assertEqual(
            generation.initial_response_condition({
                "scope_relation": "frontier",
                "initial_response_mode": "plausible_incorrect",
            }),
            "plausible_incorrect",
        )

    def test_phase2_receives_natural_dialogue_only(self) -> None:
        dialogue = [
            {"role": "student", "content": "2だと思います。", "state_after": self.state},
            {"role": "teacher", "content": "理由を確認しよう。", "analysis": "非公開状態"},
        ]
        payload = generation.build_phase2_input("元問題", dialogue, "類似問題")
        self.assertEqual(
            payload["phase1_dialogue"],
            [
                {"role": "student", "content": "2だと思います。"},
                {"role": "teacher", "content": "理由を確認しよう。"},
            ],
        )
        self.assertNotIn("final_student_state", payload)

    def test_phase2_knowledge_source_must_be_profile_or_teacher_quote(self) -> None:
        dialogue = [
            {"role": "teacher", "content": "両辺に3を足すと5x=15です。"},
        ]
        valid = {
            "answer": r"\boxed{18}",
            "knowledge_sources": [{
                "source_type": "phase1_teacher", "source_text": "5x=15",
            }],
            "application_summary": "教師が示した等式変形を類似問題へ適用した。",
        }
        self.assertEqual(
            generation.validate_phase2_transfer(valid, self.profile, dialogue)["answer"],
            r"\boxed{18}",
        )
        invalid = {
            **valid,
            "knowledge_sources": [{
                "source_type": "phase1_teacher", "source_text": "解の公式",
            }],
        }
        with self.assertRaisesRegex(ValueError, "exact dialogue quote"):
            generation.validate_phase2_transfer(invalid, self.profile, dialogue)

    def test_teacher_cot_final_is_separated(self) -> None:
        final, analysis, completed = generation.parse_teacher_response(
            "<analysis>検算済み</analysis><final>理由も確認できたね。[指導完了]</final>"
        )
        self.assertEqual(analysis, "検算済み")
        self.assertEqual(final, "理由も確認できたね。")
        self.assertTrue(completed)


class V4EvaluationTest(unittest.TestCase):
    def test_only_all_three_completed_judges_are_skipped(self) -> None:
        row = {
            field: {"ok": True}
            for field in (
                "math_judge", "v4_instruction_evaluation", "student_realism_evaluation",
            )
        }
        self.assertTrue(evaluation.evaluation_succeeded(row))
        row["v4_instruction_evaluation"] = {"error": "timeout"}
        self.assertFalse(evaluation.evaluation_succeeded(row))

    def test_v4_total_is_recomputed_from_six_scores(self) -> None:
        fields = [
            "mathematical_accuracy_score", "error_diagnosis_recovery_score",
            "cognitive_empathy_score", "emotional_support_score",
            "adaptive_scaffolding_score", "verification_completion_score",
        ]
        result = evaluation.recompute_total({field: 8 for field in fields}, fields)
        self.assertEqual(result["total_score"], 48)


if __name__ == "__main__":
    unittest.main()
