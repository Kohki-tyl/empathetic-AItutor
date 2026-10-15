import json
import sys
import unittest
from pathlib import Path

FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FOLDER))
import prepare_dataset as prep


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = prep.read_jsonl(FOLDER / "data.jsonl")
        cls.plain = prep.read_jsonl(prep.PLAIN / "data.jsonl")
        cls.source = {r["source_id"]: r for r in prep.read_jsonl(prep.CORPUS)}

    def test_every_turn_maps_to_source_and_preserves_split_and_visible_text(self):
        prompt = (FOLDER / "prompts/teacher_system.txt").read_text(encoding="utf-8").strip()
        self.assertEqual(len(self.rows), 1000)
        self.assertEqual(sum(r["split"] == "train" for r in self.rows), 900)
        for old, new in zip(self.plain, self.rows):
            self.assertEqual({k: v for k, v in old.items() if k != "messages"},
                             {k: v for k, v in new.items() if k != "messages"})
            source = self.source[old["group_id"]]
            teachers = iter(t for t in source["conversation"] if t["role"] == "teacher")
            self.assertEqual(new["messages"][0]["content"], prompt)
            for index, (a, b) in enumerate(zip(old["messages"][1:], new["messages"][1:]), 1):
                if a["role"] == "user":
                    self.assertEqual(a, b)
                    continue
                turn = next(teachers)
                analysis, final = b["content"].split("</analysis>\n<final>\n")
                for field, label in zip(prep.FIELDS, ("認知状態", "感情状態", "次の一歩")):
                    self.assertIn(f"【{label}】{prep.normalize_analysis(turn[field])}", analysis)
                expected = a["content"]
                if index == len(new["messages"]) - 1 and source["is_completed"] and "[指導完了]" not in expected:
                    expected += "\n[指導完了]"
                self.assertEqual(final, expected + "\n</final>")
        self.assertEqual({r['group_id'] for r in self.rows if r['split'] == 'train'} &
                         {r['group_id'] for r in self.rows if r['split'] == 'validation'}, set())

    def test_files_hashes_and_split_copies(self):
        manifest = json.loads((FOLDER / "manifest.json").read_text(encoding="utf-8"))
        for filename, digest in manifest["files"].items():
            self.assertEqual(prep.sha256(FOLDER / filename), digest)
        for split in ("train", "validation"):
            self.assertEqual(prep.read_jsonl(FOLDER / f"{split}.jsonl"), [r for r in self.rows if r["split"] == split])
        self.assertEqual((FOLDER / "prompts/teacher_system.txt").read_bytes(), prep.PROMPT_SOURCE.read_bytes())

    def test_missing_analysis_or_misaligned_turns_fail(self):
        row = self.plain[0]
        source = self.source[row["group_id"]]
        broken = json.loads(json.dumps(source))
        next(t for t in broken["conversation"] if t["role"] == "teacher").pop("thought_process")
        with self.assertRaises(ValueError):
            prep.convert_record(row, broken, "prompt")
        broken = json.loads(json.dumps(source))
        broken["conversation"].pop(0)
        with self.assertRaises(ValueError):
            prep.convert_record(row, broken, "prompt")


if __name__ == "__main__":
    unittest.main()
