from __future__ import annotations

import importlib.util
import hashlib
import json
import os
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


def canonical_prompt_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n")
    if path.suffix == ".json":
        text = json.dumps(
            json.loads(text), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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

    def test_teacher_receives_initial_emotion_label_only_on_first_turn(self) -> None:
        first = generation.teacher_user_input("1+1は？", "confused", "3かな。", 0)
        self.assertIn("初期感情ラベル: confused", first)
        self.assertIn("生徒発話: 3かな。", first)
        self.assertEqual(
            generation.teacher_user_input("1+1は？", "confused", "2です。", 1),
            "2です。",
        )

    def test_lora_serving_metadata_requires_actual_child_model(self) -> None:
        evidence = generation.validate_teacher_serving_metadata(
            [{"id": "v4-sft", "root": "adapter/v4", "parent": "base-swallow"}],
            "v4-sft", "lora", "adapter/v4", "base-swallow",
        )
        self.assertEqual(evidence["serving_mode"], "lora")
        self.assertEqual(evidence["parent"], "base-swallow")
        with self.assertRaisesRegex(RuntimeError, "LoRA model card"):
            generation.validate_teacher_serving_metadata(
                [{"id": "v4-sft", "root": "base-swallow", "parent": None}],
                "v4-sft", "lora", "adapter/v4",
            )

    def test_judge_rejects_incomplete_generation_rows(self) -> None:
        valid = {
            "run_id": "run-1",
            "phase1_turns": 1,
            "dialogue_log": [{"role": "student", "content": "分かりません"}],
            "phase2_student_answer": r"\boxed{わからない}",
            "phase2_student_trace": {"answer": r"\boxed{わからない}"},
            "generation_error": None,
        }
        evaluation.validate_generation_inputs([valid])
        failed = dict(valid, generation_error="Phase2 timeout", phase2_student_answer="")
        with self.assertRaisesRegex(ValueError, "Judgeを開始しません"):
            evaluation.validate_generation_inputs([failed])

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

    def test_empty_delta_preserves_acquired_knowledge(self) -> None:
        previous = dict(self.state, acquired_knowledge=["移項"])
        model_state = {
            key: value for key, value in previous.items()
            if key != "acquired_knowledge"
        }
        normalized, _ = generation.validate_student_state(
            model_state, previous, newly_acquired_knowledge=[],
        )
        self.assertEqual(normalized["acquired_knowledge"], ["移項"])

    def test_student_schema_uses_knowledge_delta(self) -> None:
        schema = generation.STUDENT_TURN_SCHEMA["json_schema"]["schema"]
        self.assertNotIn(
            "acquired_knowledge",
            schema["properties"]["state_after"]["properties"],
        )
        self.assertIn("newly_acquired_knowledge", schema["properties"])

    def test_corpus_prompts_and_profiles_are_synced(self) -> None:
        sync_manifest = json.loads(
            (BASE_DIR / "prompts" / "corpus_prompt_sync.json").read_text(encoding="utf-8")
        )
        configured = os.getenv("V4_CORPUS_PROMPT_DIR")
        candidates = [
            Path(configured) if configured else None,
            BASE_DIR.parent.parent / "corpus_creation" / "v4" / "prompts",
            BASE_DIR.parents[2] / "v4" / "prompts",
        ]
        corpus_prompt_dir = next(
            (candidate for candidate in candidates if candidate and candidate.is_dir()), None,
        )
        for entry in sync_manifest["files"]:
            test_path = BASE_DIR / "prompts" / entry["test_name"]
            self.assertEqual(
                canonical_prompt_hash(test_path), entry["canonical_sha256"],
                f"同梱promptが同期manifestと不一致です: {entry['test_name']}",
            )
            if corpus_prompt_dir is not None:
                corpus_path = corpus_prompt_dir / entry["corpus_name"]
                self.assertTrue(corpus_path.is_file(), f"コーパスpromptがありません: {corpus_path}")
                self.assertEqual(
                    canonical_prompt_hash(corpus_path), entry["canonical_sha256"],
                    f"コーパスとtest-v4の同期が必要です: {entry['test_name']}",
                )

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

    def test_staged_test_selections_are_balanced_disjoint_parent_partition(self) -> None:
        assignments = generation.load_epistemic_assignments(
            BASE_DIR / "prompts" / "problem_profile_assignments.jsonl",
        )
        excluded = generation.read_excluded_ids(
            BASE_DIR.parent / "shared" / "questions" / "excluded_test_question_ids.json",
        )
        parent_ids = set(generation.load_problem_selection(
            BASE_DIR / "prompts" / "test_120_selection.json", assignments, excluded,
        ))
        primary_ids = set(generation.load_problem_selection(
            BASE_DIR / "prompts" / "test_60_primary_selection.json", assignments, excluded,
        ))
        confirmation_ids = set(generation.load_problem_selection(
            BASE_DIR / "prompts" / "test_60_confirmation_selection.json", assignments, excluded,
        ))
        self.assertEqual(len(parent_ids), 120)
        self.assertEqual(len(primary_ids), 60)
        self.assertEqual(len(confirmation_ids), 60)
        self.assertFalse(primary_ids & confirmation_ids)
        self.assertEqual(primary_ids | confirmation_ids, parent_ids)
        for source_ids, expected in ((primary_ids, 15), (confirmation_ids, 15)):
            counts = {
                relation: sum(
                    assignments[source_id]["scope_relation"] == relation
                    for source_id in source_ids
                )
                for relation in ("mastered", "frontier", "one_step_beyond", "far_beyond")
            }
            self.assertEqual(set(counts.values()), {expected})

    def test_far_beyond_initial_utterance_requires_explicit_two_attempt_prefix(self) -> None:
        disclosure = (
            "1回目は問題文の条件を書き出しました。"
            "2回目は既習の方法で式を作ろうとしました。"
            "2回とも、未習の関係が必要な箇所で止まりました。"
        )
        value = {
            "state_after": self.state,
            "newly_acquired_knowledge": [],
            "response_stage": "help_seeking",
            "knowledge_used": [],
            "state_update_reason": "同じ箇所で二度停止しているため",
            "utterance": "次にどの関係を使えばよいですか。",
        }
        with self.assertRaisesRegex(ValueError, "2回の試行履歴"):
            generation.parse_student_turn(
                json.dumps(value, ensure_ascii=False), self.state,
                allow_emotion_change=False,
                expected_response_mode="scope_limited_help_seeking",
                required_initial_disclosure=disclosure,
            )
        value["utterance"] = disclosure + "次にどの関係を使えばよいですか。"
        accepted = generation.parse_student_turn(
            json.dumps(value, ensure_ascii=False), self.state,
            allow_emotion_change=False,
            expected_response_mode="scope_limited_help_seeking",
            required_initial_disclosure=disclosure,
        )
        self.assertTrue(accepted["utterance"].startswith(disclosure))

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
