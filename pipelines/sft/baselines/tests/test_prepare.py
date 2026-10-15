import argparse
import copy
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare as p


class Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        ids = [1]
        for m in messages:
            ids += [{"system": 2, "user": 3, "assistant": 4}[m["role"]]]
            ids += [100 + ord(c) for c in m["content"]] + [5]
        return ids + ([4] if add_generation_prompt else [])


def math_row(qid="1"):
    return {"qid": qid, "question": "2+2?", "ground_truth": "SECRET ANSWER",
            "teacher_described_confusion": "SECRET LABEL",
            "conversation": "Teacher: (generic)Try it.|EOM|Student: 5|EOM|Teacher: (focus)Count again.|EOM|Student: 4"}


class Tests(unittest.TestCase):
    def test_math_roles_annotations_and_group(self):
        a, b = p.mathdial([math_row(), math_row()])
        self.assertEqual(a["split"], b["split"])
        self.assertEqual([m["role"] for m in a["messages"]], ["system", "user", "assistant", "user", "assistant"])
        self.assertEqual(a["messages"][2]["content"], "Try it.")
        self.assertNotIn("SECRET", str(a["messages"]))

    def test_math_accepts_named_student_from_profile(self):
        row = math_row()
        row["student_profile"] = "Steven is a 7th grade student."
        row["conversation"] = row["conversation"].replace("Student:", "Steven:")
        converted = p.mathdial([row])[0]
        self.assertEqual(converted["messages"][3], {"role": "user", "content": "5"})
        row["conversation"] = row["conversation"].replace("Steven:", "Unexpected:")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            p.mathdial([row])

    def test_empathy_orders_speakers_not_global_parity(self):
        rows = [{"conv_id": "c", "utterance_idx": i, "speaker_idx": s, "utterance": t,
                 "prompt": "SECRET", "context": "SECRET"}
                for i, s, t in [(3, 18, "Thanks"), (1, 18, "Sad_comma_ today"), (2, 7, "I'm here.")]]
        row = p.empathetic(rows, "validation")[0]
        self.assertEqual(row["messages"][1]["content"], "Sad, today")
        self.assertEqual(row["messages"][-1]["role"], "assistant")
        self.assertNotIn("SECRET", str(row["messages"]))
        rows.append(rows[-1])
        with self.assertRaises(ValueError):
            p.empathetic(rows, "train")

    def test_plain_mask_and_structured_default(self):
        row = p.mathdial([math_row()])[0]
        with self.assertRaises(ValueError):
            p.engine.validate_messages([row], 1)
        encoded = p.engine.encode_with_assistant_mask(Tokenizer(), row, 8192, 1, 1, "plain")
        supervised = [t for t, label in zip(encoded.input_ids, encoded.labels) if label != -100]
        expected = [100 + ord(c) for c in "Try it."] + [5]
        expected += [100 + ord(c) for c in "Count again."] + [5]
        self.assertEqual(supervised, expected)
        self.assertEqual(encoded.analysis_tokens, 0)
        with self.assertRaises(ValueError):
            p.engine.encode_with_assistant_mask(Tokenizer(), row, 3, 1, 1, "plain")

    def test_budget_is_deterministic(self):
        rows = p.mathdial([math_row(str(i)) for i in range(20)])
        a, tokens = p.select(rows, 42, 100, Tokenizer())
        self.assertLessEqual(tokens, 100)
        self.assertEqual(a, p.select(list(reversed(rows)), 42, 100, Tokenizer())[0])

    def test_build_and_structure_audit_all_conditions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            math = p.mathdial([math_row(str(i)) for i in range(100)])
            empathy = copy.deepcopy(math)
            for row in empathy:
                row["id"] = "ed:" + row["id"]
                row["source"] = "empatheticdialogues"
            p.write_rows(root / "math.jsonl", math)
            p.write_rows(root / "ed.jsonl", empathy)
            args = argparse.Namespace(math=root / "math.jsonl", empathy=root / "ed.jsonl",
                                      output=root / "run", language="en", seed=42, target_tokens=None)
            p.build(args)
            for condition in ("math", "empathy", "mixed"):
                config = args.output / f"{condition}.json"
                with contextlib.redirect_stdout(io.StringIO()):
                    report = p.engine.structure_check(config)
                self.assertGreater(report["train_records"], 0)
                self.assertGreater(report["validation_records"], 0)
                p.engine.prepare_run(config, Tokenizer())
            with self.assertRaises(FileExistsError):
                p.build(args)
            args.output = root / "leak"
            leaked = copy.deepcopy(math[0])
            leaked["id"] = "new"
            leaked["split"] = "validation" if leaked["split"] == "train" else "train"
            p.write_rows(args.math, math + [leaked])
            with self.assertRaisesRegex(ValueError, "leaks"):
                p.build(args)


if __name__ == "__main__":
    unittest.main()
