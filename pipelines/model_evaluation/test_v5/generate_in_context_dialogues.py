"""test-v4条件を維持し、生徒だけをGPT-5.4へ置換するv5生成入口。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CORE_PATH = BASE_DIR.parent / "test_v4" / "generate_in_context_dialogues.py"
GPT54_SNAPSHOT = "gpt-5.4-2026-03-05"


def load_core():
    spec = importlib.util.spec_from_file_location("test_v5_generation_core", CORE_PATH)
    if not spec or not spec.loader:
        raise RuntimeError(f"test-v4生成器を読み込めません: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def apply_v5_defaults(argv: list[str]) -> list[str]:
    defaults = [
        ("--student-api-provider", "openai"),
        ("--student-model", GPT54_SNAPSHOT),
        ("--student-checkpoint", GPT54_SNAPSHOT),
        ("--student-revision", GPT54_SNAPSHOT),
        ("--student-base-url", "https://api.openai.com/v1"),
        ("--student-reasoning-effort", "none"),
        ("--student-request-timeout", "180"),
        ("--student-temperature", "0.6"),
        ("--student-top-p", "0.95"),
        ("--student-top-k", "0"),
        ("--student-min-p", "0.0"),
        ("--student-max-tokens", "4096"),
        ("--output", str(BASE_DIR / "data" / "gpt54_student" / "primary_60" / "dialogues.jsonl")),
    ]
    result = list(argv)
    for flag, value in reversed(defaults):
        if not any(arg == flag or arg.startswith(flag + "=") for arg in result):
            result[1:1] = [flag, value]
    return result


def main() -> None:
    sys.argv = apply_v5_defaults(sys.argv)
    load_core().main()


if __name__ == "__main__":
    main()
