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


class PipelineLogicTests(unittest.TestCase):
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

    def test_profile_emotion_assignments_are_balanced(self):
        profiles = [{"id": f"S{index}"} for index in range(4)]
        emotions = [f"e{index}" for index in range(6)]
        assignments = run_v4.stratified_assignments({"max_candidates": 120, "seed": 42}, profiles, emotions)
        counts = {}
        for profile, emotion in assignments:
            key = (profile["id"], emotion)
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(len(counts), 24)
        self.assertEqual(set(counts.values()), {5})

    def test_assignment_extension_preserves_existing_prefix(self):
        profiles = [{"id": f"S{index}"} for index in range(4)]
        emotions = [f"e{index}" for index in range(6)]
        first = run_v4.stratified_assignments({"max_candidates": 120, "seed": 42}, profiles, emotions)
        extended = run_v4.stratified_assignments({"max_candidates": 200, "seed": 42}, profiles, emotions)
        self.assertEqual(len(first), 120)
        self.assertEqual(len(extended), 200)
        self.assertEqual(first, extended[:120])

    def test_student_understanding_cannot_jump(self):
        previous = {
            "understanding_level": 1, "confidence": 0.5,
            "active_misconception": "m", "emotion": "neutral",
            "acquired_knowledge": [], "remaining_unknowns": ["x"],
        }
        value = {
            "state_after": {**previous, "understanding_level": 3},
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
            "state_update_reason": "開始時に改善した", "utterance": "できそうです。",
        }
        with self.assertRaises(ValueError):
            run_v4.validate_student_turn(value, previous, allow_emotion_change=False)

    def test_config_rejects_target_above_candidates(self):
        config = {
            "target_dialogues": 2, "max_candidates": 1, "max_turns": 1, "seed": 1,
            "teacher_model": "t", "teacher_reasoning_effort": "medium",
            "student_model": "s", "judge_model": "j", "judge_reasoning_effort": "high",
            "student_model_revision": "rev", "vllm_version": "0.25.1",
            "student_temperature": 0.6, "student_top_p": 0.95,
            "student_top_k": 20, "student_min_p": 0, "student_max_tokens": 100,
            "repair_model": "r", "repair_reasoning_effort": "medium",
            "questions": "q.jsonl", "output_dir": "out",
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
            self.assertIn("生徒発話: 3かな。", messages[1]["content"])
            self.assertIn("<analysis>", sft[0]["messages"][-1]["content"])
            manifest = run_v4.read_json(file_paths["manifest"])
            self.assertTrue(manifest["final"]["sft_format"]["tokenizer_length_audit_required"])
            self.assertEqual(
                manifest["selection_policy"],
                "keep_or_contextual_repair_with_full_dialogue_reaudit",
            )
            self.assertEqual(manifest["final"]["accepted_repaired_turns"], 0)

    def test_resume_rejects_immutable_config_change_but_allows_limit_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "target_dialogues": 1, "max_candidates": 1, "max_turns": 1, "seed": 1,
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

            extended = {**config, "target_dialogues": 2, "max_candidates": 2}
            manifest = run_v4.load_manifest(extended, file_paths)
            self.assertEqual(manifest["current_limits"]["max_candidates"], 2)

            run_v4.save_manifest(file_paths["manifest"], manifest)
            decreased = {**extended, "target_dialogues": 1, "max_candidates": 1}
            with self.assertRaises(RuntimeError):
                run_v4.load_manifest(decreased, file_paths)

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
            "reference_solution": "2", "student_profile": {"id": "V2-S01"},
            "initial_emotion": "confused",
            "initial_student_state": {"active_misconception": "加法の数え違い"},
            "final_student_state": {"active_misconception": "なし", "emotion": "engaged"},
            "is_completed": False, "generation_error": None, "models": {},
            "conversation": [
                {"turn": 0, "role": "student", "content": "3かな。"},
                original_first,
                {"turn": 1, "role": "student", "content": "数えると2かも。"},
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
