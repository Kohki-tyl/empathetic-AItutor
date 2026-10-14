import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fix_split import fixed_split


def row(identifier, group, split):
    return {"id": identifier, "group_id": group, "split": split, "messages": [
        {"role": "system", "content": "system"}, {"role": "user", "content": "question"},
        {"role": "assistant", "content": "reply"}]}


class Tests(unittest.TestCase):
    def test_exact_split_minimal_changes_and_no_leakage(self):
        rows = [row(str(i), str(i // 2), "validation" if i < 6 else "train") for i in range(10)]
        result, changes = fixed_split(rows, 4, 42)
        self.assertEqual(sum(r["split"] == "validation" for r in result), 4)
        self.assertEqual(len(changes), 2)
        groups = {}
        for old, new in zip(rows, result):
            self.assertEqual(new["messages"], old["messages"])
            self.assertEqual(new["original_split"], old["split"])
            self.assertEqual(groups.setdefault(new["group_id"], new["split"]), new["split"])
        self.assertEqual(result, fixed_split(list(reversed(rows)), 4, 42)[0][::-1])
        self.assertEqual(sum(r["split"] == "validation" for r in rows), 6)

    def test_impossible_count_fails_instead_of_splitting_group(self):
        rows = [row(str(i), str(i // 2), "validation" if i < 2 else "train") for i in range(6)]
        with self.assertRaisesRegex(ValueError, "impossible"):
            fixed_split(rows, 3)


if __name__ == "__main__":
    unittest.main()
