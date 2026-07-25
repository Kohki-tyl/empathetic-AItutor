from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


V4_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = V4_DIR / "scripts" / "build_problem_profile_assignments.py"
SPEC = importlib.util.spec_from_file_location("build_problem_profile_assignments", MODULE_PATH)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)

SELECTION_MODULE_PATH = V4_DIR / "scripts" / "build_balanced_problem_selections.py"
SELECTION_SPEC = importlib.util.spec_from_file_location(
    "build_balanced_problem_selections", SELECTION_MODULE_PATH,
)
assert SELECTION_SPEC and SELECTION_SPEC.loader
selection_builder = importlib.util.module_from_spec(SELECTION_SPEC)
sys.modules[SELECTION_SPEC.name] = selection_builder
SELECTION_SPEC.loader.exec_module(selection_builder)


class ProblemProfileAssignmentTests(unittest.TestCase):
    def test_order_starts_at_math_train_0_and_is_numeric(self):
        rows = [
            {"id": "math_train_10"},
            {"id": "math_train_2"},
            {"id": "not_math_train_0"},
            {"id": "math_train_0"},
            {"id": "math_train_1"},
        ]
        self.assertEqual(
            [row["id"] for row in builder.ordered_questions(rows)],
            ["math_train_0", "math_train_1", "math_train_2", "math_train_10"],
        )

    def test_checked_in_assignment_table_starts_at_zero(self):
        path = V4_DIR / "assignments" / "problem_profile_assignments.jsonl"
        with path.open(encoding="utf-8") as stream:
            first = json.loads(next(line for line in stream if line.strip()))
        self.assertEqual(first["source_id"], "math_train_0")
        self.assertEqual(first["order_index"], 0)

    def test_emotion_is_derived_from_scope_and_difficulty(self):
        self.assertEqual(builder.initial_emotion("mastered", 1)[0], "neutral")
        self.assertEqual(builder.initial_emotion("frontier", 5)[0], "confused")
        self.assertEqual(builder.initial_emotion("one_step_beyond", 5)[0], "anxious")
        self.assertEqual(builder.initial_emotion("far_beyond", 1)[0], "frustrated")

    def test_advanced_notation_sets_conservative_concept_stage(self):
        matrix = builder.detect_required_concepts(
            r"\begin{pmatrix}1&0\\0&1\end{pmatrix}", "行列を計算する",
        )
        self.assertIn("matrix", {item["id"] for item in matrix})
        self.assertEqual(builder.required_stage("algebra", "", "", matrix), 4)
        combination = builder.detect_required_concepts(r"\dbinom{15}{3}", "組合せ")
        self.assertIn("binomial_coefficient", {item["id"] for item in combination})

    def test_level_five_moves_mastered_candidate_two_relations_harder(self):
        self.assertEqual(builder.scope_relation(3, 4, 5), "one_step_beyond")

    def test_far_beyond_has_two_explicit_prior_attempts(self):
        history = builder.prior_attempt_history(
            "far_beyond", "geometry", "高校数学I・A相当",
            "三角形ABCの辺の長さを求めなさい。", ["正弦定理"],
        )
        self.assertEqual(history["attempt_count"], 2)
        self.assertEqual(
            [attempt["attempt_number"] for attempt in history["attempts"]],
            [1, 2],
        )
        self.assertEqual(
            {attempt["stopped_at"] for attempt in history["attempts"]},
            {history["repeated_stuck_point"]},
        )
        self.assertTrue(history["required_initial_disclosure"].startswith("1回目は"))
        self.assertIn("2回目は", history["required_initial_disclosure"])
        self.assertIn("2回とも", history["required_initial_disclosure"])
        self.assertIn("三角形ABC", history["required_initial_disclosure"])

    def test_far_beyond_attempt_history_is_problem_specific(self):
        first = builder.prior_attempt_history(
            "far_beyond", "geometry", "高校数学I・A相当",
            "三角形ABCの辺を求めなさい。", ["正弦定理"],
        )
        second = builder.prior_attempt_history(
            "far_beyond", "geometry", "高校数学I・A相当",
            "円Oの接線の長さを求めなさい。", ["接弦定理"],
        )
        self.assertNotEqual(
            first["required_initial_disclosure"],
            second["required_initial_disclosure"],
        )

    def test_non_far_beyond_has_no_fabricated_attempt(self):
        history = builder.prior_attempt_history(
            "one_step_beyond", "geometry", "高校数学I・A相当",
        )
        self.assertEqual(history["attempt_count"], 0)
        self.assertEqual(history["attempts"], [])
        self.assertEqual(history["required_initial_disclosure"], "なし")

    def test_first_120_include_all_scope_relations_and_hard_emotions(self):
        path = V4_DIR / "assignments" / "problem_profile_assignments.jsonl"
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()][:120]
        self.assertEqual(
            {row["scope_relation"] for row in rows},
            {"mastered", "frontier", "one_step_beyond", "far_beyond"},
        )
        emotions = {row["initial_emotion"] for row in rows}
        self.assertIn("anxious", emotions)
        self.assertIn("frustrated", emotions)

    def test_checked_in_corpus_selection_is_balanced_within_first_800(self):
        selection = json.loads(
            (V4_DIR / "assignments" / "corpus_120_selection.json").read_text(
                encoding="utf-8",
            )
        )
        self.assertEqual(selection["source_partition"], {"start": 0, "end_exclusive": 800})
        self.assertEqual(len(selection["records"]), 120)
        counts = {
            relation: sum(
                row["scope_relation"] == relation for row in selection["records"]
            )
            for relation in selection_builder.RELATIONS
        }
        self.assertEqual(set(counts.values()), {30})
        self.assertTrue(all(row["order_index"] < 800 for row in selection["records"]))
        self.assertEqual(selection["records"][0]["source_id"], "math_train_0")
        self.assertTrue(selection["knowledge_boundary_audit"]["passed"])

    def test_checked_in_test_selection_is_balanced_within_last_200(self):
        path = V4_DIR / "assignments" / "test_120_selection.json"
        selection = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            selection["source_partition"], {"start": 800, "end_exclusive": 1000},
        )
        self.assertEqual(len(selection["records"]), 120)
        counts = {
            relation: sum(
                row["scope_relation"] == relation for row in selection["records"]
            )
            for relation in selection_builder.RELATIONS
        }
        self.assertEqual(set(counts.values()), {30})
        self.assertTrue(all(row["order_index"] >= 800 for row in selection["records"]))
        self.assertTrue(selection["knowledge_boundary_audit"]["passed"])

    def test_standalone_v4_contains_all_selection_inputs(self):
        self.assertTrue((V4_DIR / "questions" / "translated_1000_math.jsonl").is_file())
        self.assertTrue((V4_DIR / "questions" / "excluded_test_question_ids.json").is_file())
        self.assertTrue((V4_DIR / "assignments" / "test_120_selection.json").is_file())
        rows = [
            json.loads(line) for line in (
                V4_DIR / "assignments" / "test_problem_profile_assignments.jsonl"
            ).read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        self.assertEqual(len(rows), 200)


if __name__ == "__main__":
    unittest.main()
