from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


BASE_DIR = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wrapper = load_module("test_v5_wrapper", BASE_DIR / "generate_in_context_dialogues.py")
core = wrapper.load_core()


class V5GPT54StudentTest(unittest.TestCase):
    def test_defaults_pin_gpt54_snapshot_and_openai_provider(self) -> None:
        argv = wrapper.apply_v5_defaults(["generate", "--teacher-model", "teacher"])
        self.assertEqual(argv[argv.index("--student-model") + 1], "gpt-5.4-2026-03-05")
        self.assertEqual(argv[argv.index("--student-revision") + 1], "gpt-5.4-2026-03-05")
        self.assertEqual(argv[argv.index("--student-api-provider") + 1], "openai")
        self.assertEqual(argv[argv.index("--student-reasoning-effort") + 1], "none")
        self.assertEqual(argv[argv.index("--student-top-k") + 1], "0")

    def test_explicit_cli_value_overrides_default(self) -> None:
        argv = wrapper.apply_v5_defaults(["generate", "--student-temperature", "0.2"])
        self.assertEqual(argv.count("--student-temperature"), 1)
        self.assertEqual(argv[argv.index("--student-temperature") + 1], "0.2")

    def test_openai_call_uses_supported_chat_completion_parameters(self) -> None:
        kwargs = core.build_call_kwargs(
            "gpt-5.4-2026-03-05",
            [{"role": "user", "content": "test"}],
            0.6,
            4096,
            core.STUDENT_TURN_SCHEMA,
            42,
            {"top_p": 0.95},
            api_provider="openai",
            reasoning_effort="none",
        )
        self.assertEqual(kwargs["max_completion_tokens"], 4096)
        self.assertNotIn("max_tokens", kwargs)
        self.assertEqual(kwargs["reasoning_effort"], "none")
        self.assertEqual(kwargs["top_p"], 0.95)
        self.assertNotIn("extra_body", kwargs)

    def test_openai_call_rejects_vllm_only_sampling_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k"):
            core.build_call_kwargs(
                "gpt-5.4-2026-03-05", [{"role": "user", "content": "test"}],
                0.6, extra_body={"top_p": 0.95, "top_k": 20},
                api_provider="openai", reasoning_effort="none",
            )

    def test_openai_model_preflight_requires_exact_snapshot(self) -> None:
        client = SimpleNamespace(models=SimpleNamespace(retrieve=Mock(
            return_value=SimpleNamespace(id="gpt-5.4-2026-03-05")
        )))
        evidence = core.validate_openai_student_model(client, "gpt-5.4-2026-03-05")
        self.assertEqual(evidence["provider"], "openai")
        self.assertEqual(evidence["snapshot"], "gpt-5.4-2026-03-05")
        with self.assertRaisesRegex(RuntimeError, "一致しません"):
            core.validate_openai_student_model(client, "gpt-5.4")


if __name__ == "__main__":
    unittest.main()
