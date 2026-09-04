from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_module("public_prepare_samples", BASE_DIR / "prepare_samples.py")
translate = load_module("public_translate_samples", BASE_DIR / "translate_samples.py")


class PipelineTests(unittest.TestCase):
    def test_samples_and_manifest(self) -> None:
        manifest = json.loads((BASE_DIR / "data" / "sample_manifest.json").read_text(encoding="utf-8"))
        for name in translate.DATASETS:
            path = BASE_DIR / "data" / f"{name}_500_source.jsonl"
            rows = prepare.read_jsonl(path)
            self.assertEqual(len(rows), 500)
            self.assertEqual(len({row["id"] for row in rows}), 500)
            self.assertEqual(prepare.sha256_file(path), manifest["datasets"][name]["output_sha256"])

    def test_mathdial_parser_preserves_roles_and_acts(self) -> None:
        value = "Teacher: (probing)Why?|EOM|Student: Because.|EOM|Teacher: (generic)Good!"
        turns = prepare.parse_mathdial_conversation(value)
        self.assertEqual([turn["role"] for turn in turns], ["assistant", "user", "assistant"])
        self.assertEqual(turns[0]["dialogue_act"], "probing")

    def test_sft_rows_alternate_and_end_with_assistant(self) -> None:
        for dataset in translate.DATASETS:
            source = prepare.read_jsonl(BASE_DIR / "data" / f"{dataset}_500_source.jsonl")[0]
            translated = translate.translation_payload(source, dataset)
            row = translate.sft_row(source, translated, dataset)
            roles = [message["role"] for message in row["messages"]]
            self.assertEqual(roles[0], "system")
            self.assertEqual(roles[-1], "assistant")
            self.assertTrue(all(left != right for left, right in zip(roles[1:], roles[2:])))

    def test_mathdial_sft_starts_with_real_student_turn(self) -> None:
        source = prepare.read_jsonl(BASE_DIR / "data" / "mathdial_500_source.jsonl")[0]
        translated = translate.translation_payload(source, "mathdial")
        first_student = next(turn for turn in translated["turns"] if turn["role"] == "user")
        row = translate.sft_row(source, translated, "mathdial")
        self.assertEqual(row["messages"][1], {"role": "user", "content": first_student["content"]})


if __name__ == "__main__":
    unittest.main()
