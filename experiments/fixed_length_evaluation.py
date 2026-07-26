"""既存対話の先頭1・3・5教師ターンを同一Judgeで再評価する。"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import statistics
import threading
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments" / "fixed_length" / "evaluations.jsonl"
PROMPT_PATH = (
    ROOT / "pipelines" / "model_evaluation" / "test_v4" / "prompts"
    / "instruction_judge_system.txt"
)
CUTOFFS = (1, 3, 5)
SCORE_FIELDS = (
    "mathematical_accuracy_score",
    "error_diagnosis_recovery_score",
    "cognitive_empathy_score",
    "emotional_support_score",
    "adaptive_scaffolding_score",
    "verification_completion_score",
)
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fixed_length_instruction_judge",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                **{
                    field: {"type": "integer", "minimum": 0, "maximum": 10}
                    for field in SCORE_FIELDS
                },
                "total_score": {"type": "integer", "minimum": 0, "maximum": 60},
                "false_affirmation_count": {"type": "integer", "minimum": 0},
                "direct_answer_without_need_count": {"type": "integer", "minimum": 0},
                "critical_failure": {"type": "boolean"},
                "analysis_reflected_in_utterance": {"type": "boolean"},
                "judge_reason": {"type": "string"},
            },
            "required": [
                *SCORE_FIELDS,
                "total_score",
                "false_affirmation_count",
                "direct_answer_without_need_count",
                "critical_failure",
                "analysis_reflected_in_utterance",
                "judge_reason",
            ],
            "additionalProperties": False,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--judge-model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def teacher_turn_count(dialogue: list[dict[str, Any]]) -> int:
    return sum(turn.get("role") == "teacher" for turn in dialogue)


def prefix_dialogue(
    dialogue: list[dict[str, Any]], cutoff: int,
) -> list[dict[str, Any]]:
    prefix: list[dict[str, Any]] = []
    count = 0
    for turn in dialogue:
        prefix.append(turn)
        if turn.get("role") == "teacher":
            count += 1
            if count == cutoff:
                return prefix
    raise ValueError(f"teacher turns are fewer than cutoff={cutoff}")


def dialogue_text(dialogue: list[dict[str, Any]]) -> str:
    blocks = []
    for turn in dialogue:
        block = f"{turn['role']}: {turn['content']}"
        if turn.get("role") == "teacher" and turn.get("analysis"):
            block += f"\n教師の明示的判断記録: {turn['analysis']}"
        blocks.append(block)
    return "\n\n".join(blocks)


def dataset_rows() -> dict[str, list[dict[str, Any]]]:
    paths = {
        "v3_base": ROOT / "experiments" / "test_v2" / "v3_build"
        / "base_swallow_8b_v0.5" / "baseline_v2_quoted.jsonl",
        "v3_sft": ROOT / "experiments" / "test_v2" / "v3_build"
        / "sft_v3_swallow_8b_v0.5" / "evaluated_results.jsonl",
        "v4_sft": ROOT / "experiments" / "test_v4" / "data"
        / "v4_sft_provisional" / "retry_60" / "evaluated_successes.jsonl",
        "v5_sft": ROOT / "experiments" / "test_v5" / "v4_sft"
        / "primary_60" / "evaluated_initial_successes.jsonl",
    }
    loaded = {name: read_jsonl(path) for name, path in paths.items()}

    base = {row["source_id"]: row for row in loaded["v3_base"]}
    sft = {row["source_id"]: row for row in loaded["v3_sft"]}
    common = sorted(
        source_id
        for source_id in base.keys() & sft.keys()
        if teacher_turn_count(base[source_id]["dialogue_log"]) >= 5
        and teacher_turn_count(sft[source_id]["dialogue_log"]) >= 5
    )
    loaded["v3_base"] = [base[source_id] for source_id in common]
    loaded["v3_sft"] = [sft[source_id] for source_id in common]
    for name in ("v4_sft", "v5_sft"):
        loaded[name] = [
            row for row in loaded[name]
            if teacher_turn_count(row["dialogue_log"]) >= 5
        ]
    return loaded


def build_tasks() -> list[dict[str, Any]]:
    tasks = []
    for dataset, rows in dataset_rows().items():
        for row in rows:
            for cutoff in CUTOFFS:
                prefix = prefix_dialogue(row["dialogue_log"], cutoff)
                text = dialogue_text(prefix)
                tasks.append({
                    "task_id": f"{dataset}:{row['source_id']}:turn-{cutoff}",
                    "dataset": dataset,
                    "source_id": row["source_id"],
                    "run_id": row.get("run_id"),
                    "cutoff": cutoff,
                    "original_teacher_turns": teacher_turn_count(row["dialogue_log"]),
                    "prefix_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "dialogue_text": text,
                })
    return tasks


def parse_content(content: str) -> dict[str, Any]:
    result = json.loads(content)
    result["total_score"] = sum(result[field] for field in SCORE_FIELDS)
    return result


_thread_local = threading.local()


def thread_client(api_key: str) -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OpenAI(api_key=api_key, timeout=180.0)
        _thread_local.client = client
    return client


def evaluate_task(
    task: dict[str, Any], api_key: str, model: str, reasoning_effort: str,
    prompt: str, retries: int,
) -> dict[str, Any]:
    seed = int(hashlib.sha256(task["task_id"].encode()).hexdigest()[:8], 16)
    user = (
        f"これは最大5教師ターンある同一対話の先頭{task['cutoff']}教師ターンです。\n"
        "【対話prefix】\n" + task["dialogue_text"]
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = thread_client(api_key).chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=2048,
                response_format=SCHEMA,
                reasoning_effort=reasoning_effort,
                seed=seed + attempt,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Judge returned empty content")
            result = parse_content(content)
            return {
                key: value for key, value in task.items() if key != "dialogue_text"
            } | {
                "judge_model": model,
                "reasoning_effort": reasoning_effort,
                "judge_attempts": attempt + 1,
                "evaluation": result,
            }
        except Exception as exc:  # APIの一時失敗を再試行する
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt + random.random(), 8))
    return {
        key: value for key, value in task.items() if key != "dialogue_text"
    } | {
        "judge_model": model,
        "reasoning_effort": reasoning_effort,
        "judge_attempts": retries,
        "error": f"{type(last_error).__name__}: {last_error}",
    }


def main() -> None:
    args = parse_args()
    load_env(ROOT / ".env")
    tasks = build_tasks()
    if args.limit is not None:
        tasks = tasks[:args.limit]
    print(json.dumps({
        "planned_tasks": len(tasks),
        "by_dataset": {
            name: sum(task["dataset"] == name for task in tasks)
            for name in sorted({task["dataset"] for task in tasks})
        },
    }, ensure_ascii=False))
    if args.dry_run:
        return

    api_key = os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("GPT_API_KEY または OPENAI_API_KEY が必要です")
    if args.workers < 1 or args.retries < 1:
        raise SystemExit("--workersと--retriesは1以上にしてください")

    prompt = PROMPT_PATH.read_text(encoding="utf-8") + """

【固定長prefix評価の追加規則】
- 入力は同一対話を所定ターンで機械的に打ち切ったprefixである。
- 打ち切り時点で指導が未完了であること自体は減点しない。
- verification_completion_scoreは、観察範囲内で必要な理解確認を行ったか、未確認なのに完了扱いしていないか、次の確認へ適切に接続したかを評価する。
- prefix内に実際に現れた誤り、誤答追認、矛盾、不要な直接解答、反復だけを数える。
- 後続ターンの存在や内容を推測してはいけない。
"""
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        args.output.write_text("", encoding="utf-8")
    existing = read_jsonl(args.output) if args.output.exists() else []
    done = {
        row["task_id"] for row in existing
        if row.get("task_id") and isinstance(row.get("evaluation"), dict)
    }
    pending = [task for task in tasks if task["task_id"] not in done]
    manifest = {
        "judge_model": args.judge_model,
        "reasoning_effort": args.reasoning_effort,
        "prompt_sha256": prompt_hash,
        "cutoffs": list(CUTOFFS),
        "fixed_cohort_rule": "same dialogue has at least 5 teacher turns",
        "v3_pair_rule": "both Base and SFT have at least 5 teacher turns",
        "planned_tasks": len(tasks),
        "already_completed": len(tasks) - len(pending),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if not pending:
        return

    with args.output.open("a", encoding="utf-8") as output:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    evaluate_task, task, api_key, args.judge_model,
                    args.reasoning_effort, prompt, args.retries,
                ): task
                for task in pending
            }
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                completed += 1
                if completed % 10 == 0 or completed == len(pending):
                    errors = sum(1 for row in existing if row.get("error"))
                    print(f"completed={completed}/{len(pending)} current_error_rows={errors}")


if __name__ == "__main__":
    main()
