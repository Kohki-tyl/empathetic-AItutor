from __future__ import annotations

import importlib.util
import json
import random
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
        all_rows = MODULE.read_jsonl(BASE_DIR / "data" / "legacy_unaudited_800_sft.jsonl")
        audited_rows = MODULE.read_jsonl(BASE_DIR / "data" / "legacy_audited_500_sft.jsonl")
        self.assertEqual(len(all_rows), 800)
        self.assertEqual(len(audited_rows), 500)
        self.assertLessEqual({row["id"] for row in audited_rows}, {row["id"] for row in all_rows})
        for name, path in (("unaudited_800", "legacy_unaudited_800_sft.jsonl"), ("audited_500", "legacy_audited_500_sft.jsonl")):
            self.assertEqual(MODULE.sha256_file(BASE_DIR / "data" / path), manifest["datasets"][name]["sha256"])

    def test_audited_sample_is_deterministic_and_passes(self) -> None:
        corpus_ids = sorted(row["source_id"] for row in MODULE.read_jsonl(MODULE.DEFAULT_CORPUS))
        random.Random(42).shuffle(corpus_ids)
        evaluations = {row["source_id"]: row for row in MODULE.read_jsonl(MODULE.DEFAULT_EVALUATED)}
        expected = [source_id for source_id in corpus_ids if MODULE.strict_audit_pass(evaluations[source_id])][:500]
        actual = [row["id"] for row in MODULE.read_jsonl(BASE_DIR / "data" / "legacy_audited_500_sft.jsonl")]
        self.assertEqual(actual, expected)

    def test_no_forbidden_control_characters_remain(self) -> None:
        for name in ("legacy_unaudited_800_sft.jsonl", "legacy_audited_500_sft.jsonl"):
            for row in MODULE.read_jsonl(BASE_DIR / "data" / name):
                for message in row["messages"]:
                    self.assertFalse(any(ord(char) < 32 and char not in "\n\r\t" for char in message["content"]))


if __name__ == "__main__":
    unittest.main()
