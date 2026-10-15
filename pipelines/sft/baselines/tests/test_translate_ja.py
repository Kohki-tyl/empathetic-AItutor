import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import translate_ja as t
from fetch_sources import sample


def source(identifier="a"):
    return {"id": identifier, "group_id": identifier, "source": "mathdial", "split": "train", "language": "en",
            "messages": [{"role": "system", "content": "system"},
                         {"role": "user", "content": "I think 2+2=5. It costs $12 and $8. Also \\(x=3\\)."},
                         {"role": "assistant", "content": "Try counting 2 again."}]}


class FakeClient:
    def __init__(self, fail_at=None):
        self.responses = self
        self.calls = 0
        self.fail_at = fail_at

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("interrupted")
        payload = json.loads(kwargs["input"])
        value = {"turns": [{"index": turn["index"], "text": "日本語：" + turn["text"]} for turn in payload["turns"]]}
        return SimpleNamespace(status="completed", output_text=json.dumps(value), model="gpt-6-luna",
                               id=f"r{self.calls}", usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}))


class Tests(unittest.TestCase):
    def test_formula_only_solution_preserves_latex_environment_without_japanese(self):
        row = source()
        row["messages"][2]["content"] = r"\begin{align*}x^2=\boxed{361}.\end{align*}"
        payload, mappings = t.protect(row["messages"])
        value = {"turns": [{"index": 1, "text": "日本語 " + payload["turns"][0]["text"]},
                           {"index": 2, "text": payload["turns"][1]["text"]}]}
        result = t.restore(row, value, mappings)
        self.assertEqual(result["messages"][2]["content"], row["messages"][2]["content"])

    def test_chat_typography_is_not_a_mathematical_zero(self):
        row = source()
        row["messages"][1]["content"] = "Did y0u get 2?"
        payload, mappings = t.protect(row["messages"])
        self.assertEqual(list(mappings[0].values()), ["2"])
        self.assertIn("you", payload["turns"][0]["text"])
        self.assertEqual(row["messages"][1]["content"], "Did y0u get 2?")
    def test_repairs_only_failed_turn_with_original_context(self):
        class DropOnce(FakeClient):
            def create(self, **kwargs):
                result = super().create(**kwargs)
                if self.calls == 1:
                    value = json.loads(result.output_text)
                    value["turns"][0]["text"] = value["turns"][0]["text"].replace("__KEEP_1_0__", "")
                    result.output_text = json.dumps(value)
                else:
                    request = json.loads(kwargs["input"])
                    self.assertion = len(request["turns"]) == 1 and len(request["context"]) == 2
                return result

        client = DropOnce()
        row = t.translate_one(client, source(), "gpt-6-luna", 1000)
        self.assertEqual(client.calls, 2)
        self.assertTrue(client.assertion)
        self.assertEqual(row["translation"]["repaired_turns"], {1: 1})
        self.assertEqual(row["translation"]["usage"]["total_tokens"], 24)

    def test_protection_preserves_numeric_errors_currency_and_math(self):
        row = source()
        payload, mappings = t.protect(row["messages"])
        self.assertNotIn("12", payload["turns"][0]["text"])
        self.assertNotIn("costs", str(list(mappings[0].values())))
        value = {"turns": [{"index": turn["index"], "text": "日本語 " + turn["text"]} for turn in payload["turns"]]}
        restored = t.restore(row, value, mappings)
        self.assertIn("2+2=5", restored["messages"][1]["content"])
        self.assertIn("\\(x=3\\)", restored["messages"][1]["content"])
        self.assertEqual(restored["id"], row["id"])
        self.assertEqual(restored["split"], "train")
        self.assertEqual(restored["language"], "ja")
        value["turns"][0]["text"] += " 99"
        flagged = t.restore(row, value, mappings)
        self.assertEqual(flagged["translation_qa"][0]["values"], ["99"])

    def test_rejects_missing_placeholder_and_changed_order(self):
        row = source()
        payload, mappings = t.protect(row["messages"])
        value = {"turns": [{"index": turn["index"], "text": "日本語 " + turn["text"]} for turn in payload["turns"]]}
        value["turns"][0]["text"] = "日本語"
        with self.assertRaisesRegex(ValueError, "Protected"):
            t.restore(row, value, mappings)
        value["turns"][0]["index"] = 99
        with self.assertRaisesRegex(ValueError, "order"):
            t.restore(row, value, mappings)

    def test_accepts_exact_original_literals_without_allowing_numeric_changes(self):
        row = source()
        payload, mappings = t.protect(row["messages"])
        turns = []
        for turn, mapping in zip(payload["turns"], mappings):
            text = "日本語：" + turn["text"]
            for key, literal in mapping.items():
                text = text.replace(key, literal)
            turns.append({"index": turn["index"], "text": text})
        restored = t.restore(row, {"turns": turns}, mappings)
        self.assertIn("2+2=5", restored["messages"][1]["content"])
        turns[1]["text"] = turns[1]["text"].replace("2", "9")
        with self.assertRaisesRegex(ValueError, "Protected"):
            t.restore(row, {"turns": turns}, mappings)

    def test_preserves_original_separator_between_adjacent_numeric_placeholders(self):
        row = source()
        row["messages"][1]["content"] = "Shred 2 132 contracts."
        payload, mappings = t.protect(row["messages"])
        value = {"turns": [{"index": x["index"], "text": "日本語 " + x["text"]} for x in payload["turns"]]}
        value["turns"][0]["text"] = "__KEEP_1_0____KEEP_1_1__通の契約書を裁断。"
        result = t.restore(row, value, mappings)
        self.assertIn("2 132", result["messages"][1]["content"])

    def test_random_sampling_is_reproducible_and_order_independent(self):
        rows = [source(str(i)) for i in range(20)]
        self.assertEqual(sample(rows, 10, 42), sample(list(reversed(rows)), 10, 42))
        self.assertNotEqual(sample(rows, 10, 42), rows[:10])

    def test_resume_preserves_completed_translations_and_rejects_changed_model(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            args = argparse.Namespace(input=root / "en.jsonl", output=root / "ja.jsonl", model="gpt-6-luna",
                                      max_output_tokens=1000, limit=None, dry_run=False, workers=1)
            t.p.write_rows(args.input, [source("a"), source("b"), source("c")])
            client = FakeClient(fail_at=2)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                t.run(args, lambda: client)
            self.assertFalse(args.output.exists())
            self.assertEqual(len(t.p.engine.read_jsonl(args.output.with_suffix(".partial.jsonl"))), 1)
            args.model = "different"
            with self.assertRaisesRegex(ValueError, "differ"):
                t.run(args, FakeClient)
            args.model = "gpt-6-luna"
            resumed = FakeClient()
            t.run(args, lambda: resumed)
            self.assertEqual(resumed.calls, 2)
            self.assertEqual(len(t.p.engine.read_jsonl(args.output)), 3)
            manifest = t.p.engine.read_json(args.output.with_suffix(".manifest.json"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["usage"]["total_tokens"], 36)


if __name__ == "__main__":
    unittest.main()
