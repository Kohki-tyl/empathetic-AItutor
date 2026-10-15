import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_additional_math import select_additional


def raw(problem, solution="A solution."):
    return {"problem": problem, "solution": solution, "subject": "Algebra", "level": 3,
            "unique_id": "train/algebra/example.json"}


class Tests(unittest.TestCase):
    def test_continuation_excludes_existing_and_new_duplicates_and_diagrams(self):
        rows = [raw("Old problem"), raw("Old problem"), raw("As shown below, find x."),
                raw("New problem A"), raw("New problem A"), raw("New problem B")]
        existing = [{"id": "math_train_0", "original_problem": "Old problem", "original_solution": "A solution."}]
        selected, rejected = select_additional(rows, existing, 2)
        self.assertEqual([r["id"] for r in selected], ["math_train_3", "math_train_5"])
        self.assertEqual(selected[0]["type"], "Algebra")
        self.assertEqual(selected[0]["messages"][1]["content"], "New problem A")
        self.assertEqual(len(rejected), 3)

    def test_source_alignment_mismatch_fails(self):
        existing = [{"id": "math_train_0", "original_problem": "Other", "original_solution": "A solution."}]
        with self.assertRaisesRegex(ValueError, "ordering/content"):
            select_additional([raw("Changed"), raw("New")], existing, 1)


if __name__ == "__main__":
    unittest.main()
