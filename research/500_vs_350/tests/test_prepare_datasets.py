from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_datasets", BASE_DIR / "prepare_datasets.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DatasetPreparationTests(unittest.TestCase):
    def test_generated_counts_hashes_and_subset(self) -> None:
        manifest = json.loads((BASE_DIR / "data" / "dataset_manifest.json").read_text(encoding="utf-8"))
        all_rows = MODULE.read_jsonl(BASE_DIR / "data" / "legacy_all_500_sft.jsonl")
        strict_rows = MODULE.read_jsonl(BASE_DIR / "data" / "legacy_research_350_sft.jsonl")
        self.assertEqual(len(all_rows), 500)
        self.assertEqual(len(strict_rows), 350)
        self.assertEqual({row["id"] for row in strict_rows} - {row["id"] for row in all_rows}, set())
        self.assertEqual(MODULE.sha256_file(BASE_DIR / "data" / "legacy_all_500_sft.jsonl"), manifest["datasets"]["all_500"]["sha256"])
        self.assertEqual(MODULE.sha256_file(BASE_DIR / "data" / "legacy_research_350_sft.jsonl"), manifest["datasets"]["research_350"]["sha256"])

    def test_strict_ids_match_research_evaluations(self) -> None:
        evaluated = MODULE.read_jsonl(MODULE.DEFAULT_EVALUATED)
        expected = {row["source_id"] for row in evaluated if MODULE.strict_research_pass(row)}
        actual = {row["id"] for row in MODULE.read_jsonl(BASE_DIR / "data" / "legacy_research_350_sft.jsonl")}
        self.assertEqual(actual, expected)

    def test_no_forbidden_control_characters_remain(self) -> None:
        for name in ("legacy_all_500_sft.jsonl", "legacy_research_350_sft.jsonl"):
            for row in MODULE.read_jsonl(BASE_DIR / "data" / name):
                for message in row["messages"]:
                    self.assertFalse(any(ord(char) < 32 and char not in "\n\r\t" for char in message["content"]))


if __name__ == "__main__":
    unittest.main()
