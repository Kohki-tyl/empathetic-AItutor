from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_v4.py"
SPEC = importlib.util.spec_from_file_location("run_v4", MODULE_PATH)
assert SPEC and SPEC.loader
run_v4 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_v4
SPEC.loader.exec_module(run_v4)


def audit_with(score: int = 9, **updates):
    value = {name: score for name in run_v4.SCORE_FIELDS}
    value.update({
        "mathematically_correct": True,
        "student_answer_assessed_correctly": True,
        "cognitive_state_grounded": True,
        "emotion_grounded": True,
        "analysis_reflected_in_utterance": True,
        "student_profile_consistent": True,
        "student_role_consistent": True,
        "student_state_update_plausible": True,
        "initial_emotion_utterance_consistent": True,
        "false_affirmation": False,
        "direct_answer_without_need": False,
        "completion_decision_appropriate": True,
        "critical_failure": False,
        "context_repairable": True,
        "mathematical_verification": "verified",
        "issues": [], "repair_instructions": [], "reason": "ok",
    })
    value.update(updates)
    return value


def valid_teacher_turn(**updates):
    value = {
        "mathematical_assessment": {
            "status": "correct", "verification": "1+1=2",
            "correct_part": "加法", "error_part": "なし",
        },
        "learner_state": {
            "cognitive_state": "理由を含めて正答した",
            "emotion": "Neutral", "evidence": "感情表現はない",
        },
        "support_decision": {"next_support": "なし", "change_reason": "なし"},
        "is_completed": True,
        "teacher_utterance": "式と理由を示せています。1+1=2で正解です。根拠も確認できました。",
    }
    value.update(updates)
    return value


class PipelineLogicTests(unittest.TestCase):
    def test_checked_in_config_uses_bundled_questions(self):
        config_path = MODULE_PATH.parent / "config.json"
        config = run_v4.load_config(config_path)
        questions = Path(config["questions"])
        self.assertTrue(questions.is_file())
        self.assertEqual(questions.parent, MODULE_PATH.parent / "questions")

    def test_teacher_turn_input_exposes_only_initial_fixed_context(self):
        profile = {
            "prior_knowledge": ["整数の加法"],
            "unknown_knowledge": ["指数法則"],
            "max_independent_math_level": "中学1年相当",
        }
        assignment = {
            "scope_relation": "far_beyond",
            "prior_attempt_history": {
                "attempt_count": 2,
                "attempts": ["1回目", "2回目"],
                "repeated_stuck_point": "指数記法",
            },
        }
        content = run_v4.teacher_turn_input(
            problem="x²とは何か", reference_solution="xを2回掛ける",
            student_utterance="2回試したけれど分かりません。",
            profile=profile, epistemic_assignment=assignment,
            initial_emotion="frustrated", turn_index=0,
        )
        self.assertIn("内部検算用の参照解答", content)
        self.assertIn('"scope_relation": "far_beyond"', content)
        self.assertIn('"prior_knowledge": ["整数の加法"]', content)
        self.assertIn('"unknown_knowledge": ["指数法則"]', content)
        self.assertIn('"attempt_count": 2', content)
        self.assertIn('"initial_emotion": "frustrated"', content)
        self.assertNotIn("current_emotion", content)
        self.assertNotIn("acquired_knowledge", content)
        self.assertIn("生徒発話: 2回試したけれど分かりません。", content)

        later = run_v4.teacher_turn_input(
            problem="x²とは何か", reference_solution="xを2回掛ける",
            student_utterance="文字は使えます。",
            profile=profile, epistemic_assignment=assignment,
            initial_emotion="frustrated", turn_index=1,
        )
        self.assertNotIn("内部検算用の参照解答", later)
        self.assertNotIn("学習者条件", later)
        self.assertEqual(later, "文字は使えます。")

    def test_student_vllm_call_uses_model_card_max_tokens_parameter(self):
        captured = {}

        class Completions:
            @staticmethod
            def create(**kwargs):
                captured.update(kwargs)
                message = type("Message", (), {"content": "{}"})()
                choice = type("Choice", (), {"message": message})()
                return type("Response", (), {"choices": [choice]})()

        client = type("Client", (), {
            "chat": type("Chat", (), {"completions": Completions()})(),
        })()
        run_v4.chat_call(
            client, "student", [{"role": "user", "content": "x"}], {},
            max_completion_tokens=4096, use_max_tokens=True, retries=1,
        )
        self.assertEqual(captured["max_tokens"], 4096)
        self.assertNotIn("max_completion_tokens", captured)

    def test_teacher_schema_uses_five_decision_blocks(self):
        schema = run_v4.TEACHER_SCHEMA["json_schema"]["schema"]
        self.assertEqual(
            set(schema["properties"]),
            {"mathematical_assessment", "learner_state", "support_decision", "is_completed", "teacher_utterance"},
        )

    def test_keep_threshold(self):
        self.assertEqual(run_v4.classify_audit(audit_with(score=8)), "Keep")

    def test_dialogue_keep_ignores_metadata_only_warnings(self):
        audit = audit_with(score=8)
        audit["metadata_warnings"] = [
            "knowledge_usedが実使用知識より広い",
            "active_misconceptionの更新が遅れている",
        ]
        self.assertEqual(run_v4.classify_dialogue_audit(audit), "Keep")

    def test_dialogue_rejects_observable_student_profile_violation(self):
        audit = audit_with(score=8, student_profile_consistent=False)
        audit["metadata_warnings"] = []
        self.assertEqual(run_v4.classify_dialogue_audit(audit), "Reject")

    def test_dialogue_keep_allows_accurate_empathetic_max_turn_incompletion(self):
        audit = audit_with(
            score=8,
            adaptive_scaffolding_score=6,
            verification_completion_score=2,
            completion_decision_appropriate=False,
        )
        audit["metadata_warnings"] = []
        audit["acceptable_incompleteness"] = [
            "最大10ターンで最終解と理由説明に未到達",
            "高難度問題で足場を細かく提示中に上限へ到達",
        ]
        self.assertEqual(run_v4.classify_dialogue_audit(audit), "Keep")

    def test_acceptable_incompleteness_never_hides_math_error(self):
        audit = audit_with(
            score=8, mathematically_correct=False, critical_failure=True,
            context_repairable=True, repair_instructions=["数学的誤りを修正する"],
        )
        audit["metadata_warnings"] = []
        audit["acceptable_incompleteness"] = ["最大ターン到達"]
        self.assertEqual(run_v4.classify_dialogue_audit(audit), "Repair")

    def test_keep_cannot_contain_unresolved_issue_or_repair_instruction(self):
        self.assertEqual(run_v4.classify_audit(audit_with(
            issues=["説明量を減らす"], repair_instructions=["一つの問いに絞る"],
        )), "Repair")

    def test_any_score_below_common_threshold_is_not_keep(self):
        self.assertEqual(
            run_v4.classify_audit(audit_with(
                score=8, adaptive_scaffolding_score=7, repair_instructions=["fix"],
            )),
            "Repair",
        )

    def test_false_affirmation_is_not_keep(self):
        self.assertEqual(run_v4.classify_audit(audit_with(
            false_affirmation=True, repair_instructions=["fix"],
        )), "Repair")

    def test_every_required_boolean_is_enforced(self):
        self.assertEqual(run_v4.classify_audit(audit_with(
            cognitive_state_grounded=False, repair_instructions=["fix"],
        )), "Repair")
        self.assertEqual(run_v4.classify_audit(audit_with(
            emotion_grounded=False, repair_instructions=["fix"],
        )), "Repair")
        self.assertEqual(run_v4.classify_audit(audit_with(
            direct_answer_without_need=True, repair_instructions=["fix"],
        )), "Repair")
        self.assertEqual(run_v4.classify_audit(audit_with(
            completion_decision_appropriate=False, repair_instructions=["fix"],
        )), "Repair")

    def test_unrepairable_or_invalid_student_context_is_reject(self):
        self.assertEqual(run_v4.classify_audit(audit_with(
            score=7, context_repairable=False, repair_instructions=["fix"],
        )), "Reject")
        self.assertEqual(run_v4.classify_audit(audit_with(
            student_role_consistent=False, repair_instructions=["fix"],
        )), "Reject")
        self.assertEqual(run_v4.classify_audit(audit_with(
            score=7, context_repairable=True, repair_instructions=[],
        )), "Reject")

    def test_questions_start_at_math_train_0_and_use_numeric_order(self):
        rows = [
            {"id": "math_train_10"}, {"id": "other_1"},
            {"id": "math_train_2"}, {"id": "math_train_0"},
            {"id": "math_train_1"},
        ]
        self.assertEqual(
            [row["id"] for row in run_v4.ordered_math_questions(rows)],
            ["math_train_0", "math_train_1", "math_train_2", "math_train_10"],
        )

    def test_out_of_scope_problem_forces_help_seeking(self):
        self.assertEqual(
            run_v4.effective_initial_response_mode(
                "correct_but_uncertain", {"scope_relation": "one_step_beyond"},
            ),
            "scope_limited_help_seeking",
        )
        self.assertEqual(
            run_v4.effective_initial_response_mode(
                "partial_reasoning", {"scope_relation": "frontier"},
            ),
            "partial_reasoning",
        )

    def test_student_understanding_cannot_jump(self):
        previous = {
            "understanding_level": 1, "confidence": 0.5,
            "active_misconception": "m", "emotion": "neutral",
            "acquired_knowledge": [], "remaining_unknowns": ["x"],
        }
        value = {
            "state_after": {**previous, "understanding_level": 3},
            "response_stage": "attempt", "knowledge_used": [],
            "state_update_reason": "jump", "utterance": "わかりました。",
        }
        with self.assertRaises(ValueError):
            run_v4.validate_student_turn(value, previous)

    def test_student_confidence_cannot_jump(self):
        previous = {
            "understanding_level": 1, "confidence": 0.5,
            "active_misconception": "m", "emotion": "neutral",
            "acquired_knowledge": [], "remaining_unknowns": ["x"],
        }
        value = {
            "state_after": {**previous, "confidence": 0.8},
            "response_stage": "attempt", "knowledge_used": [],
            "state_update_reason": "jump", "utterance": "わかりました。",
        }
        with self.assertRaises(ValueError):
            run_v4.validate_student_turn(value, previous)

    def test_teacher_reserved_marker_is_rejected(self):
        value = {
            "mathematical_assessment": {}, "learner_state": {}, "support_decision": {},
            "is_completed": True, "teacher_utterance": "正解です。[指導完了]",
        }
        with self.assertRaises(ValueError):
            run_v4.validate_teacher_turn(value)

    def test_student_emotion_follows_cycle(self):
        previous = {
            "understanding_level": 1, "confidence": 0.5,
            "active_misconception": "m", "emotion": "frustrated",
            "acquired_knowledge": [], "remaining_unknowns": ["x"],
        }
        value = {
            "state_after": {**previous, "emotion": "confused"},
            "response_stage": "attempt", "knowledge_used": [],
            "state_update_reason": "支援が具体化された", "utterance": "それなら少し考えられそう。",
        }
        self.assertEqual(run_v4.validate_student_turn(value, previous)["state_after"]["emotion"], "confused")

    def test_student_emotion_cannot_skip_cycle(self):
        previous = {
            "understanding_level": 1, "confidence": 0.5,
            "active_misconception": "m", "emotion": "frustrated",
            "acquired_knowledge": [], "remaining_unknowns": ["x"],
        }
        value = {
            "state_after": {**previous, "emotion": "proud"},
            "response_stage": "attempt", "knowledge_used": [],
            "state_update_reason": "急に解決した", "utterance": "全部わかった。",
        }
        with self.assertRaises(ValueError):
            run_v4.validate_student_turn(value, previous)

    def test_initial_emotion_is_preserved_before_teacher_turn(self):
        previous = {
            "understanding_level": 1, "confidence": 0.5,
            "active_misconception": "m", "emotion": "anxious",
            "acquired_knowledge": [], "remaining_unknowns": ["x"],
        }
        value = {
            "state_after": {**previous, "emotion": "relieved"},
            "response_stage": "attempt", "knowledge_used": [],
            "state_update_reason": "開始時に改善した", "utterance": "できそうです。",
        }
        with self.assertRaises(ValueError):
            run_v4.validate_student_turn(value, previous, allow_emotion_change=False)

    def test_initial_turn_cannot_acquire_knowledge_or_raise_understanding(self):
        previous = {
            "understanding_level": 1, "confidence": 0.5,
            "active_misconception": "m", "emotion": "curious",
            "acquired_knowledge": [], "remaining_unknowns": ["x"],
        }
        acquired = {
            "state_after": {
                **previous, "understanding_level": 2,
                "acquired_knowledge": ["new"],
            },
            "response_stage": "answer", "knowledge_used": [],
            "state_update_reason": "自力で習得した", "utterance": "答えは2です。",
        }
        with self.assertRaisesRegex(ValueError, "initial understanding"):
            run_v4.validate_student_turn(
                acquired, previous, allow_emotion_change=False
            )

    def test_initial_turn_confidence_change_is_limited(self):
        previous = {
            "understanding_level": 1, "confidence": 0.5,
            "active_misconception": "m", "emotion": "anxious",
            "acquired_knowledge": [], "remaining_unknowns": ["x"],
        }
        value = {
            "state_after": {**previous, "confidence": 0.65},
            "response_stage": "answer", "knowledge_used": [],
            "state_update_reason": "考えた", "utterance": "2だと思います。",
        }
        with self.assertRaisesRegex(ValueError, "initial confidence"):
            run_v4.validate_student_turn(value, previous, allow_emotion_change=False)

    def test_far_beyond_initial_turn_adds_two_attempt_disclosure_prefix(self):
        previous = {
            "understanding_level": 0, "confidence": 0.35,
            "active_misconception": "未習概念で停止する", "emotion": "frustrated",
            "acquired_knowledge": [], "remaining_unknowns": ["高校数学"],
        }
        disclosure = (
            "1回目は問題文の条件を書き出しました。"
            "2回目は既習の方法で式を作ろうとしました。"
            "2回とも、未習の関係が必要な箇所で止まりました。"
        )
        value = {
            "state_after": previous,
            "response_stage": "help_seeking", "knowledge_used": [],
            "state_update_reason": "同じ箇所で二度停止しているため",
            "utterance": "中心と半径は分かりますが、次に何を使いますか。",
        }
        accepted = run_v4.validate_student_turn(
            value, previous, allow_emotion_change=False,
            expected_response_mode="scope_limited_help_seeking",
            required_initial_disclosure=disclosure,
        )
        self.assertTrue(accepted["utterance"].startswith(disclosure))
        self.assertIn("required_attempt_history_prefixed", accepted["state_normalizations"])

    def test_blank_active_misconception_is_preserved(self):
        previous = {
            "understanding_level": 1, "confidence": 0.5,
            "active_misconception": "符号を反転し忘れる", "emotion": "neutral",
            "acquired_knowledge": [], "remaining_unknowns": ["x"],
        }
        value = {
            "state_after": {**previous, "active_misconception": ""},
            "newly_acquired_knowledge": [], "response_stage": "attempt",
            "knowledge_used": [], "state_update_reason": "まだ誤りが残っている",
            "utterance": "ここを移項するのかな。",
        }
        accepted = run_v4.validate_student_turn(value, previous)
        self.assertEqual(
            accepted["state_after"]["active_misconception"],
            previous["active_misconception"],
        )
        self.assertIn("blank_active_misconception_preserved", accepted["state_normalizations"])

    def test_out_of_boundary_knowledge_metadata_is_deferred_to_dialogue_audit(self):
        previous = {
            "understanding_level": 1, "confidence": 0.5,
            "active_misconception": "m", "emotion": "neutral",
            "acquired_knowledge": [], "remaining_unknowns": ["x"],
        }
        value = {
            "state_after": previous,
            "response_stage": "attempt", "knowledge_used": ["微分"],
            "state_update_reason": "試した", "utterance": "微分してみます。",
        }
        accepted = run_v4.validate_student_turn(
            value, previous, allowed_knowledge=["一次方程式"],
        )
        self.assertIn(
            "knowledge_used_outside_boundary_retained_for_audit",
            accepted["state_normalizations"],
        )

    def test_new_knowledge_must_be_copied_from_latest_teacher(self):
        previous = {
            "understanding_level": 1, "confidence": 0.5,
            "active_misconception": "m", "emotion": "neutral",
            "acquired_knowledge": [], "remaining_unknowns": ["x"],
        }
        value = {
            "state_after": {**previous, "acquired_knowledge": ["解の公式"]},
            "response_stage": "help_seeking", "knowledge_used": [],
            "state_update_reason": "教わった", "utterance": "ここまでは分かりました。",
        }
        with self.assertRaisesRegex(ValueError, "not copied"):
            run_v4.validate_student_turn(
                value, previous, latest_teacher_utterance="因数分解を試そう。",
            )
        accepted = run_v4.validate_student_turn(
            value, previous, latest_teacher_utterance="ここでは解の公式を使います。",
        )
        self.assertEqual(accepted["state_after"]["acquired_knowledge"], ["解の公式"])

    def test_new_schema_accumulates_knowledge_delta_without_reoutput(self):
        previous = {
            "understanding_level": 1, "confidence": 0.5,
            "active_misconception": "m", "emotion": "neutral",
            "acquired_knowledge": ["移項"], "remaining_unknowns": ["x"],
        }
        model_state = {
            key: value for key, value in previous.items()
            if key != "acquired_knowledge"
        }
        value = {
            "state_after": model_state,
            "newly_acquired_knowledge": ["両辺を同じ数で割る"],
            "response_stage": "attempt", "knowledge_used": ["移項"],
            "state_update_reason": "直前の説明を一段階使った",
            "utterance": "まず移項してみます。",
        }
        accepted = run_v4.validate_student_turn(
            value, previous,
            allowed_knowledge=["移項"],
            latest_teacher_utterance="次は両辺を同じ数で割ると整理できます。",
        )
        self.assertEqual(
            accepted["state_after"]["acquired_knowledge"],
            ["移項", "両辺を同じ数で割る"],
        )
        self.assertEqual(
            accepted["newly_acquired_knowledge"], ["両辺を同じ数で割る"],
        )

    def test_student_schema_requests_delta_not_full_acquired_knowledge(self):
        state_schema = run_v4.STUDENT_SCHEMA["json_schema"]["schema"]["properties"]["state_after"]
        top_properties = run_v4.STUDENT_SCHEMA["json_schema"]["schema"]["properties"]
        self.assertNotIn("acquired_knowledge", state_schema["properties"])
        self.assertIn("newly_acquired_knowledge", top_properties)

    def test_student_turn_schema_constrains_initial_emotion_and_stage(self):
        previous = {"emotion": "anxious"}
        schema = run_v4.student_schema_for_turn(
            previous, "scope_limited_help_seeking", allow_emotion_change=False,
        )
        properties = schema["json_schema"]["schema"]["properties"]
        self.assertEqual(
            properties["state_after"]["properties"]["emotion"]["enum"], ["anxious"],
        )
        self.assertEqual(
            properties["response_stage"]["enum"], ["observation", "help_seeking"],
        )

    def test_completed_teacher_cannot_add_next_support(self):
        value = valid_teacher_turn(
            support_decision={"next_support": "代入を確認する", "change_reason": "なし"}
        )
        with self.assertRaisesRegex(ValueError, "next support"):
            run_v4.validate_teacher_turn(value)

    def test_completed_teacher_cannot_ask_follow_up(self):
        value = valid_teacher_turn(
            teacher_utterance="式は正しいです。答えは2です。確認してみましょう。"
        )
        with self.assertRaisesRegex(ValueError, "follow-up"):
            run_v4.validate_teacher_turn(value)

    def test_completed_teacher_without_follow_up_is_valid(self):
        self.assertTrue(run_v4.validate_teacher_turn(valid_teacher_turn())["is_completed"])

    def test_config_rejects_target_above_candidates(self):
        config = {
            "target_dialogues": 2, "max_candidates": 1, "max_turns": 1, "seed": 1,
            "teacher_model": "t", "teacher_reasoning_effort": "medium",
            "student_model": "s", "judge_model": "j", "judge_reasoning_effort": "high",
            "student_model_revision": "rev", "vllm_version": "0.25.1",
            "student_temperature": 0.6, "student_top_p": 0.95,
            "student_top_k": 20, "student_min_p": 0, "student_max_tokens": 100,
            "repair_model": "r", "repair_reasoning_effort": "medium",
            "questions": "q.jsonl", "problem_profile_assignments": "a.jsonl",
            "problem_selection": "s.json",
            "output_dir": "out",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "target_dialogues"):
                run_v4.load_config(path)

    def test_finalize_builds_corpus_and_sft(self):
        teacher = {
            "turn": 0, "role": "teacher",
            "mathematical_assessment": {
                "status": "incorrect", "verification": "1+1=2",
                "correct_part": "なし", "error_part": "計算ミス",
            },
            "learner_state": {
                "cognitive_state": "計算を確認中", "emotion": "Confusion",
                "evidence": "3と回答し、迷いを示した",
            },
            "support_decision": {"next_support": "1+1を数える", "change_reason": "なし"},
            "is_completed": False,
            "teacher_utterance": "1個と1個を合わせるといくつかな？",
        }
        dialogue = {
            "candidate_id": "v4-0000", "source_id": "q1", "problem": "1+1は？",
            "reference_solution": "2", "student_profile": {"id": "V2-S01"},
            "initial_emotion": "confused", "initial_student_state": {},
            "final_student_state": {}, "is_completed": False, "generation_error": None,
            "models": {}, "conversation": [
                {"turn": 0, "role": "student", "content": "3かな。"}, teacher,
            ],
        }
        audit = {
            "turn_key": "v4-0000:teacher:0", "status": "completed",
            "classification": "Keep", "total_score": 48, "audit": audit_with(score=8),
        }
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "target_dialogues": 1, "max_candidates": 1, "max_turns": 1, "seed": 1,
                "teacher_model": "gpt-5.6-terra", "student_model": "s",
                "student_model_revision": "rev", "vllm_version": "0.25.1",
                "student_temperature": 0.6, "student_max_tokens": 100,
                "judge_model": "gpt-5.6-terra",
                "repair_model": "gpt-5.6-terra", "repair_reasoning_effort": "medium",
                "teacher_reasoning_effort": "medium", "judge_reasoning_effort": "high",
                "questions": "q", "output_dir": directory,
            }
            file_paths = run_v4.paths(config)
            run_v4.write_jsonl(file_paths["dialogues"], [dialogue])
            run_v4.write_jsonl(file_paths["audits"], [audit])
            run_v4.finalize(config, file_paths)
            self.assertEqual(len(run_v4.read_jsonl(file_paths["corpus"])), 1)
            sft = run_v4.read_jsonl(file_paths["sft"])
            messages = sft[0]["messages"]
            self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant"])
            self.assertIn("問題: 1+1は？", messages[1]["content"])
            self.assertIn("初期感情ラベル: confused", messages[1]["content"])
            self.assertIn("生徒発話: 3かな。", messages[1]["content"])
            self.assertIn("<analysis>", sft[0]["messages"][-1]["content"])
            manifest = run_v4.read_json(file_paths["manifest"])
            self.assertTrue(manifest["final"]["sft_format"]["tokenizer_length_audit_required"])
            self.assertEqual(
                manifest["selection_policy"],
                "keep_or_contextual_repair_with_full_dialogue_reaudit",
            )
            self.assertEqual(manifest["final"]["accepted_repaired_turns"], 0)

    def test_resume_allows_target_extension_but_rejects_candidate_change(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "target_dialogues": 1, "max_candidates": 2, "max_turns": 1, "seed": 1,
                "teacher_model": "t", "teacher_reasoning_effort": "medium",
                "student_model": "s", "student_model_revision": "rev",
                "vllm_version": "0.25.1", "student_temperature": 0.6,
                "student_max_tokens": 100,
                "judge_model": "j", "judge_reasoning_effort": "high",
                "repair_model": "r", "repair_reasoning_effort": "medium",
                "questions": "q", "output_dir": directory,
            }
            file_paths = run_v4.paths(config)
            run_v4.save_manifest(file_paths["manifest"], run_v4.default_manifest(config))

            extended = {**config, "target_dialogues": 2}
            manifest = run_v4.load_manifest(extended, file_paths)
            self.assertEqual(manifest["current_limits"]["target_dialogues"], 2)

            run_v4.save_manifest(file_paths["manifest"], manifest)
            decreased = {**extended, "target_dialogues": 1}
            with self.assertRaises(RuntimeError):
                run_v4.load_manifest(decreased, file_paths)

            changed_candidates = {**extended, "max_candidates": 3}
            with self.assertRaises(RuntimeError):
                run_v4.load_manifest(changed_candidates, file_paths)

            changed = {**extended, "seed": 2}
            with self.assertRaises(RuntimeError):
                run_v4.load_manifest(changed, file_paths)

    def test_resume_rejects_question_source_content_change(self):
        with tempfile.TemporaryDirectory() as directory:
            question_path = Path(directory) / "questions.jsonl"
            question_path.write_text('{"id":"q1","problem":"1+1?"}\n', encoding="utf-8")
            config = {
                "target_dialogues": 1, "max_candidates": 1, "max_turns": 1, "seed": 1,
                "teacher_model": "t", "teacher_reasoning_effort": "medium",
                "student_model": "s", "student_model_revision": "rev",
                "vllm_version": "0.25.1", "student_temperature": 0.6,
                "student_max_tokens": 100,
                "judge_model": "j", "judge_reasoning_effort": "high",
                "repair_model": "r", "repair_reasoning_effort": "medium",
                "questions": str(question_path), "output_dir": directory,
            }
            file_paths = run_v4.paths(config)
            run_v4.save_manifest(file_paths["manifest"], run_v4.default_manifest(config))
            question_path.write_text('{"id":"q1","problem":"2+2?"}\n', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                run_v4.load_manifest(config, file_paths)

    def test_overwriting_audit_invalidates_all_downstream_results(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"output_dir": directory}
            file_paths = run_v4.paths(config)
            for key in ("audits", "repairs", "reaudits", "corpus", "sft", "report"):
                path = file_paths[key]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("stale\n", encoding="utf-8")
            manifest = {
                "batch_jobs": {name: {"status": "completed"} for name in ("audit", "repair", "reaudit")},
                "final": {"accepted_dialogues": 1},
            }
            run_v4.invalidate_batch_stage(manifest, file_paths, "audit")
            self.assertEqual(manifest["batch_jobs"], {})
            self.assertNotIn("final", manifest)
            for key in ("audits", "repairs", "reaudits", "corpus", "sft", "report"):
                self.assertFalse(file_paths[key].exists())

    def test_finalize_accepts_contextual_repair_only_after_every_turn_reaudit_keeps(self):
        def teacher(turn: int, utterance: str):
            return {
                "turn": turn, "role": "teacher",
                "mathematical_assessment": {
                    "status": "incorrect", "verification": "1+1=2",
                    "correct_part": "なし", "error_part": "計算ミス",
                },
                "learner_state": {
                    "cognitive_state": "計算を確認中", "emotion": "Confusion",
                    "evidence": "3と答えて迷っている",
                },
                "support_decision": {"next_support": "数え直す", "change_reason": "なし"},
                "is_completed": False, "teacher_utterance": utterance,
            }

        original_first = teacher(0, "その通り、3です。")
        repaired_first = teacher(0, "考えた点は大切です。1と1を数える部分を確認しよう。合わせるといくつかな？")
        second = teacher(1, "数え直せましたね。1+1=2で正解です。なぜ2になるか説明できるかな？")
        dialogue = {
            "candidate_id": "v4-0000", "source_id": "q1", "problem": "1+1は？",
            "reference_solution": "2", "student_profile": {
                "id": "V4-S01", "prior_knowledge": ["整数の加法"],
            },
            "initial_emotion": "confused",
            "initial_student_state": {
                "active_misconception": "加法の数え違い", "acquired_knowledge": [],
            },
            "final_student_state": {"active_misconception": "なし", "emotion": "engaged"},
            "is_completed": False, "generation_error": None, "models": {},
            "conversation": [
                {
                    "turn": 0, "role": "student", "content": "3かな。",
                    "state_after": {"acquired_knowledge": []},
                    "knowledge_used": ["整数の加法"],
                },
                original_first,
                {
                    "turn": 1, "role": "student", "content": "数えると2かも。",
                    "state_after": {"acquired_knowledge": []},
                    "knowledge_used": ["整数の加法"],
                },
                second,
            ],
        }
        initial_audits = [
            {
                "turn_key": "v4-0000:teacher:0", "status": "completed",
                "classification": "Repair", "total_score": 40,
                "audit": audit_with(score=7, repair_instructions=["誤答追認を直す"]),
            },
            {
                "turn_key": "v4-0000:teacher:1", "status": "completed",
                "classification": "Keep", "total_score": 54, "audit": audit_with(),
            },
        ]
        repair_value = {
            "candidate_id": "v4-0000",
            "repaired_teacher_turns": [{
                "teacher_index": 0,
                **{key: repaired_first[key] for key in run_v4.TEACHER_PROPERTIES},
            }],
            "context_consistency_check": "次の生徒発話『2かも』へ自然につながる。",
        }
        reaudits = [
            {
                "turn_key": f"v4-0000:teacher:{index}", "status": "completed",
                "classification": "Keep", "total_score": 54, "audit": audit_with(),
            }
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "target_dialogues": 1, "max_candidates": 1, "max_turns": 2, "seed": 1,
                "teacher_model": "t", "teacher_reasoning_effort": "medium",
                "student_model": "s", "student_model_revision": "rev",
                "vllm_version": "0.25.1", "student_temperature": 0.6,
                "student_max_tokens": 100,
                "judge_model": "j", "judge_reasoning_effort": "high",
                "repair_model": "r", "repair_reasoning_effort": "medium",
                "questions": "q", "output_dir": directory,
            }
            file_paths = run_v4.paths(config)
            run_v4.write_jsonl(file_paths["dialogues"], [dialogue])
            run_v4.write_jsonl(file_paths["audits"], initial_audits)
            run_v4.write_jsonl(file_paths["repairs"], [{
                "candidate_id": "v4-0000", "status": "completed", "repair": repair_value,
            }])
            run_v4.write_jsonl(file_paths["reaudits"], reaudits)
            run_v4.finalize(config, file_paths)

            corpus = run_v4.read_jsonl(file_paths["corpus"])
            self.assertEqual(len(corpus), 1)
            self.assertEqual(corpus[0]["selection_path"], "repair_then_full_reaudit")
            self.assertEqual(corpus[0]["conversation"][1]["teacher_utterance"], repaired_first["teacher_utterance"])
            self.assertTrue(corpus[0]["conversation"][1]["repaired"])
            manifest = run_v4.read_json(file_paths["manifest"])
            self.assertEqual(manifest["final"]["accepted_repaired_dialogues"], 1)
            self.assertEqual(manifest["final"]["accepted_repaired_turns"], 1)

            run_v4.write_jsonl(file_paths["reaudits"], reaudits[:1])
            with contextlib.redirect_stderr(io.StringIO()):
                run_v4.finalize(config, file_paths)
            self.assertEqual(run_v4.read_jsonl(file_paths["corpus"]), [])

    def test_build_sft_messages_rejects_non_alternating_conversation(self):
        dialogue = {
            "candidate_id": "bad",
            "problem": "1+1は？",
            "conversation": [
                {"role": "student", "content": "2です。"},
                {"role": "student", "content": "合っていますか？"},
            ],
        }
        with self.assertRaises(ValueError):
            run_v4.build_sft_messages(dialogue, "system")


if __name__ == "__main__":
    unittest.main()
