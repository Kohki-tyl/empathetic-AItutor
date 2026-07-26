"""test-v4 Judgeを同一条件で用いるv5評価入口。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CORE_PATH = BASE_DIR.parent / "test_v4" / "evaluate_in_context_dialogues.py"


def load_core():
    spec = importlib.util.spec_from_file_location("test_v5_evaluation_core", CORE_PATH)
    if not spec or not spec.loader:
        raise RuntimeError(f"test-v4評価器を読み込めません: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def apply_v5_defaults(argv: list[str]) -> list[str]:
    data_dir = BASE_DIR / "data" / "gpt54_student" / "primary_60"
    defaults = [
        ("--input", str(data_dir / "dialogues.jsonl")),
        ("--output", str(data_dir / "evaluated_initial_successes.jsonl")),
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
