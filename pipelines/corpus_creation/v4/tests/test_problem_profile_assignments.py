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

    def test_frustrated_requires_repeated_prior_attempts(self):
        history = builder.prior_attempt_history("frustrated")
        self.assertGreaterEqual(history["attempt_count"], 2)
        self.assertNotEqual(history["repeated_stuck_point"], "なし")

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


if __name__ == "__main__":
    unittest.main()
