from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from common import AXES, extract_visible_teacher_utterance, overall_score  # noqa: E402
from build_selection import build_selection  # noqa: E402
from evaluate_dialogues import dialogue_judge_payload, validate_judge_result  # noqa: E402
from generate_initial_student_utterances import initial_payload  # noqa: E402
from generate_dialogues import student_payload, validate_initial_responses  # noqa: E402
from prepare_cases import build_cases  # noqa: E402
from prepare_corpus_evaluation import build_subset  # noqa: E402
from summarize_results import comparison, condition_summary  # noqa: E402


def evaluation_with(score: int = 8):
    return {
        "axes": {
            name: {"score": score, "evidence": "発話", "reason": "基準を満たす"}
            for name in AXES
        },
        "critical_failure_details": [],
        "instruction_completed": True,
        "completion_reason": "理解を確認した",
        "judge_summary": "妥当",
    }


class CorpusEvaluationPreparationTests(unittest.TestCase):
    def test_builds_fixed_visible_only_50_dialogue_subset(self) -> None:
        corpus = BASE_DIR.parent / "corpus" / "v3_397_dialogues.jsonl"
        metadata = BASE_DIR.parent / "corpus" / "v3_397_metadata.jsonl"
        rows, manifest = build_subset(corpus, metadata, 50)

        self.assertEqual(len(rows), 50)
        self.assertEqual(manifest["population_size"], 397)
        self.assertEqual(manifest["source_shuffle_seed"], 42)
        self.assertEqual(rows[0]["source_id"], "math_train_508")
        self.assertNotIn("問題:", rows[0]["dialogue"][0]["content"])
        self.assertTrue(all("<analysis>" not in turn["content"] for row in rows for turn in row["dialogue"]))
        self.assertTrue(all("<final>" not in turn["content"] for row in rows for turn in row["dialogue"]))


class PipelineTest(unittest.TestCase):
    def test_standalone_runtime_assets_are_complete(self) -> None:
        config = json.loads((BASE_DIR / "config.example.json").read_text(encoding="utf-8"))
        runtime_inputs = [
            "questions", "similar_questions", "profiles", "assignments",
            "selection", "training_leakage_audit",
        ]
        for name in runtime_inputs:
            configured = config["paths"][name]
            self.assertNotIn("..", Path(configured).parts)
            self.assertTrue((BASE_DIR / configured).is_file(), f"missing {name}: {configured}")
        self.assertEqual(config["paths"]["training_corpora"], [])

    def test_builds_balanced_100_selection_from_last_200_translations(self) -> None:
        selection = build_selection(
            questions_path=(BASE_DIR / "assets/test_math_questions.jsonl"),
            assignments_path=(BASE_DIR / "assets/problem_profile_assignments.jsonl"),
            exclusion_path=(BASE_DIR / "assets/excluded_test_question_ids.json"),
            tail_size=200,
            per_scope=25,
            seed=42,
        )
        self.assertEqual(selection["selected_count"], 100)
        self.assertEqual(selection["tail_first_source_id"], "math_train_884")
        self.assertEqual(selection["tail_last_source_id"], "math_train_1097")
        self.assertEqual(Counter(row["scope_relation"] for row in selection["records"]), {
            "mastered": 25, "frontier": 25, "one_step_beyond": 25, "far_beyond": 25,
        })

    def test_reuses_100_existing_cases_and_eight_profiles(self) -> None:
        cases, manifest, _ = build_cases(BASE_DIR / "config.example.json")
        self.assertEqual(len(cases), 100)
        self.assertEqual(manifest["profile_count"], 8)
        self.assertEqual(manifest["training_overlap_count"], 0)
        self.assertEqual(manifest["learning_status_counts"], {
            "far_beyond": 25, "frontier": 25, "mastered": 25, "one_step_beyond": 25,
        })
        self.assertEqual(manifest["initial_emotion_counts"], {
            "anxious": 22, "confused": 18, "curious": 10,
            "engaged": 18, "frustrated": 25, "neutral": 7,
        })
        self.assertEqual(manifest["learning_status_emotion_counts"], {
            "far_beyond|frustrated": 25,
            "frontier|confused": 15,
            "frontier|curious": 10,
            "mastered|engaged": 18,
            "mastered|neutral": 7,
            "one_step_beyond|anxious": 22,
            "one_step_beyond|confused": 3,
        })
        self.assertTrue(
            all(set(case["student_profile"]) == {"grade", "speech_style", "initial_state"} for case in cases)
        )
        styles = [case["student_profile"]["speech_style"] for case in cases]
        self.assertTrue(all(set(style) == {
            "register", "confidence_expression", "response_length",
        } for style in styles))
        self.assertTrue(all("problem_profile_assignment" not in case for case in cases))

    def test_fixed_initial_response_requires_success_and_matching_profile(self) -> None:
        cases = [{"case_id": "case-1", "profile_id": "P1"}]
        rows = [{
            "case_id": "case-1", "profile_id": "P1", "generation_succeeded": True,
            "initial_response": "分からないところがあります。",
        }]
        self.assertEqual(
            validate_initial_responses(cases, rows)["case-1"]["initial_response"],
            rows[0]["initial_response"],
        )
        bad = [{**rows[0], "profile_id": "P2"}]
        with self.assertRaisesRegex(ValueError, "profile_id"):
            validate_initial_responses(cases, bad)

    def test_initial_payload_has_only_simple_profile_and_initial_state(self) -> None:
        cases, _, _ = build_cases(BASE_DIR / "config.example.json")
        payload = json.loads(initial_payload(cases[0]))
        self.assertEqual(set(payload["student_profile"]), {"grade", "speech_style", "initial_state"})
        self.assertNotIn("initial_state", payload)
        initial_state = payload["student_profile"]["initial_state"]
        self.assertEqual(set(initial_state), {"learning_status", "emotion", "emotion_reason"})
        self.assertIn(initial_state["learning_status"], {
            "mastered", "frontier", "one_step_beyond", "far_beyond",
        })
        forbidden = {
            "calculation_accuracy", "metacognitive_skill", "topic_mastery",
            "prior_knowledge", "unknown_knowledge", "ability_level",
        }
        self.assertTrue(forbidden.isdisjoint(payload["student_profile"]))

    def test_speech_styles_are_observable_and_balanced_by_register(self) -> None:
        profiles = json.loads((BASE_DIR / "profiles" / "simple_student_profiles.json").read_text(encoding="utf-8"))
        styles = [profile["speech_style"] for profile in profiles]
        self.assertEqual(Counter(style["register"] for style in styles), {"丁寧口調": 4, "タメ口": 4})
        self.assertEqual(
            set(style["confidence_expression"] for style in styles), {"自信がある", "慎重", "控えめ"},
        )

    def test_followup_profile_contains_fixed_initial_response(self) -> None:
        cases, _, _ = build_cases(BASE_DIR / "config.example.json")
        dialogue = [{"turn": 0, "role": "student", "content": "最初はここが分かりません。"}]
        payload = json.loads(student_payload(cases[0], dialogue))
        state = payload["student_profile"]["initial_state"]
        self.assertEqual(state["initial_response"], dialogue[0]["content"])

    def test_revised_student_prompts_do_not_restore_removed_parameters(self) -> None:
        prompt_names = [
            "student_initial_system.txt", "student_followup_system.txt", "transfer_student_system.txt",
        ]
        combined = "\n".join((BASE_DIR / "prompts" / name).read_text(encoding="utf-8") for name in prompt_names)
        for forbidden in (
            "calculation_accuracy", "metacognitive_skill", "topic_mastery",
            "prior_knowledge", "unknown_knowledge", "misconception_model",
        ):
            self.assertNotIn(forbidden, combined)
        self.assertIn("無条件に同意しない", combined)
        self.assertIn("感情名", combined)

    def test_revised_teacher_prompt_requires_grounded_completion(self) -> None:
        prompt = (BASE_DIR / "prompts" / "teacher_system.txt").read_text(encoding="utf-8")
        self.assertIn("独立に検算", prompt)
        self.assertIn("支援方法を変更", prompt)
        self.assertIn("理解根拠", prompt)
        self.assertIn("[指導完了]", prompt)

    def test_revised_judge_prompt_preserves_dialogue_only_rubric(self) -> None:
        prompt = (BASE_DIR / "prompts" / "dialogue_judge_system.txt").read_text(encoding="utf-8")
        self.assertIn("対話全体", prompt)
        self.assertIn("観察可能な発話", prompt)
        self.assertIn("内部感情ラベルを正解として", prompt)
        self.assertIn("機械的に0へ上書きせず", prompt)
        self.assertIn("評価機会が存在しない場合だけ", prompt)

    def test_teacher_internal_analysis_is_not_visible(self) -> None:
        visible, completed = extract_visible_teacher_utterance(
            "<analysis>秘密の内部推論</analysis><final>次の式を試そう。[指導完了]</final>", "[指導完了]"
        )
        self.assertEqual(visible, "次の式を試そう。")
        self.assertTrue(completed)
        self.assertNotIn("秘密", visible)
        with self.assertRaisesRegex(ValueError, "タグ"):
            extract_visible_teacher_utterance("<analysis>閉じていない秘密", "[指導完了]")

    def test_judge_payload_contains_only_visible_dialogue(self) -> None:
        row = {
            "problem": "1+1は？", "reference_solution": "2", "termination_reason": "teacher_completed",
            "teacher_declared_completion": True,
            "dialogue": [
                {"role": "student", "content": "2です"},
                {"role": "teacher", "content": "理由も説明できましたね。"},
            ],
            "call_metadata": [{"raw_response": "秘密の内部推論"}],
        }
        payload = dialogue_judge_payload(row)
        self.assertNotIn("秘密の内部推論", payload)
        self.assertNotIn("call_metadata", payload)
        self.assertNotIn("termination_reason", payload)
        self.assertNotIn("teacher_declared_completion", payload)
        self.assertIn("理由も説明", payload)

    def test_na_is_excluded_from_overall_score(self) -> None:
        scores = dict(zip(AXES, [10.0, 8.0, None, 6.0, None, 4.0]))
        self.assertEqual(overall_score(scores), 42.0)

    def test_required_axes_cannot_be_na(self) -> None:
        value = evaluation_with()
        value["axes"]["mathematical_accuracy"]["score"] = None
        with self.assertRaisesRegex(ValueError, "NA"):
            validate_judge_result(value)

    def test_generation_rate_uses_all_planned_cases(self) -> None:
        evaluation = validate_judge_result(evaluation_with())
        rows = [
            {
                "case_id": "a", "condition": "base", "dialogue_generation_succeeded": True,
                "evaluation_status": "evaluated", "evaluation": evaluation,
                "termination_reason": "teacher_completed", "transfer_evaluation": {"is_correct": True},
            },
            {
                "case_id": "b", "condition": "base", "dialogue_generation_succeeded": False,
                "evaluation_status": "not_evaluable_generation_failure", "evaluation": None,
                "termination_reason": "generation_error", "transfer_evaluation": None,
            },
        ]
        summary = condition_summary(rows)
        self.assertEqual(summary["dialogue_generation_success_rate"], 0.5)
        self.assertEqual(summary["evaluated_cases"], 1)

    def test_quality_threshold_uses_all_applicable_axes(self) -> None:
        passing = evaluation_with(8)
        passing["axes"]["emotional_support"]["score"] = None
        failing = evaluation_with(8)
        failing["axes"]["scaffolding"]["score"] = 7
        rows = [
            {
                "case_id": "pass", "condition": "base", "dialogue_generation_succeeded": True,
                "evaluation_status": "evaluated", "evaluation": validate_judge_result(passing),
                "termination_reason": "teacher_completed",
            },
            {
                "case_id": "fail", "condition": "base", "dialogue_generation_succeeded": True,
                "evaluation_status": "evaluated", "evaluation": validate_judge_result(failing),
                "termination_reason": "teacher_completed",
            },
        ]
        summary = condition_summary(rows)
        self.assertEqual(summary["all_applicable_axes_at_least_8"], 1)
        self.assertEqual(summary["all_applicable_axes_at_least_8_rate"], 0.5)

    def test_paired_comparison_rejects_different_initial_responses(self) -> None:
        evaluation = validate_judge_result(evaluation_with())
        base = {
            "case_id": "a", "evaluation_status": "evaluated", "evaluation": evaluation,
            "initial_response_sha256": "one",
        }
        other = {**base, "initial_response_sha256": "two"}
        with self.assertRaisesRegex(ValueError, "初回生徒発話"):
            comparison([base], [other], 10, 42)


if __name__ == "__main__":
    unittest.main()
