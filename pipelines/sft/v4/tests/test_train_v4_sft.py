from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "train_v4_sft.py"
SPEC = importlib.util.spec_from_file_location("train_v4_sft", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    ROLE = {"system": 10, "user": 20, "assistant": 30}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize
        ids = [1]
        for message in messages:
            ids.append(self.ROLE[message["role"]])
            ids.extend(100 + ord(char) % 50 for char in message["content"])
            ids.append(9)
        if add_generation_prompt:
            ids.append(self.ROLE["assistant"])
        return ids


def row(identifier: str = "D-001") -> dict:
    return {
        "id": identifier,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "<analysis>state</analysis><final>hint</final>",
            },
            {"role": "user", "content": "answer"},
            {
                "role": "assistant",
                "content": "<analysis>check</analysis><final>next</final>",
            },
        ],
    }


class V4SftTests(unittest.TestCase):
    def test_validate_messages_accepts_v4_record(self) -> None:
        MODULE.validate_messages([row()], 1)

    def test_validate_messages_rejects_duplicate_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "重複"):
            MODULE.validate_messages([row(), row()], 2)

    def test_validate_messages_rejects_role_order(self) -> None:
        value = row()
        value["messages"][3]["role"] = "assistant"
        with self.assertRaisesRegex(ValueError, "role順"):
            MODULE.validate_messages([value], 1)

    def test_validate_messages_rejects_missing_marker(self) -> None:
        value = row()
        value["messages"][2]["content"] = "plain response"
        with self.assertRaisesRegex(ValueError, "analysis/final形式"):
            MODULE.validate_messages([value], 1)

    def test_validate_messages_rejects_text_outside_markers(self) -> None:
        value = row()
        value["messages"][2]["content"] += "leaked text"
        with self.assertRaisesRegex(ValueError, "analysis/final形式"):
            MODULE.validate_messages([value], 1)

    def test_validate_messages_rejects_empty_analysis(self) -> None:
        value = row()
        value["messages"][2]["content"] = "<analysis> </analysis><final>hint</final>"
        with self.assertRaisesRegex(ValueError, "analysis/final形式"):
            MODULE.validate_messages([value], 1)

    def test_validate_messages_rejects_nested_reserved_marker(self) -> None:
        value = row()
        value["messages"][2]["content"] = (
            "<analysis>state</analysis><final><analysis>leak</analysis></final>"
        )
        with self.assertRaisesRegex(ValueError, "analysis/final形式"):
            MODULE.validate_messages([value], 1)

    def test_split_is_deterministic_and_disjoint(self) -> None:
        rows = [row(f"D-{index:03d}") for index in range(100)]
        train1, validation1 = MODULE.deterministic_split(rows, 0.1, 42)
        train2, validation2 = MODULE.deterministic_split(rows, 0.1, 42)
        self.assertEqual([item["id"] for item in train1], [item["id"] for item in train2])
        self.assertEqual(
            [item["id"] for item in validation1],
            [item["id"] for item in validation2],
        )
        self.assertEqual(len(train1), 90)
        self.assertEqual(len(validation1), 10)
        self.assertFalse(
            {item["id"] for item in train1} & {item["id"] for item in validation1}
        )

    def test_assistant_mask_excludes_headers_and_user(self) -> None:
        tokenizer = FakeTokenizer()
        value = row()
        encoded = MODULE.encode_with_assistant_mask(tokenizer, value, 4096)
        self.assertEqual(len(encoded.input_ids), len(encoded.labels))
        target_positions = [
            index for index, label in enumerate(encoded.labels) if label != -100
        ]
        self.assertTrue(target_positions)
        self.assertTrue(all(encoded.labels[index] == encoded.input_ids[index] for index in target_positions))
        first_assistant_header = encoded.input_ids.index(FakeTokenizer.ROLE["assistant"])
        self.assertEqual(encoded.labels[first_assistant_header], -100)
        first_user_header = encoded.input_ids.index(FakeTokenizer.ROLE["user"])
        self.assertEqual(encoded.labels[first_user_header], -100)

    def test_overlength_aborts_without_truncation(self) -> None:
        with self.assertRaisesRegex(ValueError, "自動切り詰め禁止"):
            MODULE.encode_with_assistant_mask(FakeTokenizer(), row(), 5)

    def test_fingerprint_changes_with_config(self) -> None:
        first = MODULE.build_fingerprint({"lr": 1}, "abc", ["a"], ["b"])
        second = MODULE.build_fingerprint({"lr": 2}, "abc", ["a"], ["b"])
        self.assertNotEqual(first, second)

    def test_resolve_config_rejects_missing_revision(self) -> None:
        config = {
            "dataset": "data.jsonl",
            "expected_records": 1,
            "model_name": "model",
            "model_revision": "",
            "output_dir": "out",
            "max_length": 10,
            "validation_ratio": 0.1,
            "seed": 42,
            "num_train_epochs": 1,
            "learning_rate": 1e-4,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "warmup_ratio": 0.1,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
            "optim": "adamw_torch_fused",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_target_modules": ["q_proj"],
            "gradient_checkpointing": True,
            "bf16": True,
            "tf32": True,
            "dataloader_num_workers": 0,
            "save_total_limit": 1,
            "logging_steps": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model_revision"):
                MODULE.resolve_config(path)


if __name__ == "__main__":
    unittest.main()
