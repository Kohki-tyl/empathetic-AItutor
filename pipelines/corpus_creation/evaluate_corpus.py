import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "500_empathetic_dialogues.jsonl"
DEFAULT_OUTPUT = BASE_DIR / "500_dialogue_evaluations.jsonl"
JUDGE_PROMPT = BASE_DIR / "prompts" / "corpus_quality_judge_system.txt"
ENV_FILE = BASE_DIR.parents[1] / ".env"

EVALUATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "corpus_dialogue_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "emotion_alignment_score": {"type": "integer", "minimum": 0, "maximum": 10},
                "pedagogical_empathy_score": {"type": "integer", "minimum": 0, "maximum": 10},
                "mathematical_correctness_score": {"type": "integer", "minimum": 0, "maximum": 10},
                "error_detection_recovery_score": {"type": "integer", "minimum": 0, "maximum": 10},
                "adaptive_scaffolding_score": {"type": "integer", "minimum": 0, "maximum": 10},
                "length_control_score": {"type": "integer", "minimum": 0, "maximum": 10},
                "total_score": {"type": "integer", "minimum": 0, "maximum": 60},
                "false_affirmation_count": {"type": "integer", "minimum": 0},
                "repetition_count": {"type": "integer", "minimum": 0},
                "critical_failure": {"type": "boolean"},
                "recommendation": {"type": "string", "enum": ["keep", "review", "reject"]},
                "reason": {"type": "string"},
            },
            "required": [
                "emotion_alignment_score",
                "pedagogical_empathy_score",
                "mathematical_correctness_score",
                "error_detection_recovery_score",
                "adaptive_scaffolding_score",
                "length_control_score",
                "total_score",
                "false_affirmation_count",
                "repetition_count",
                "critical_failure",
                "recommendation",
                "reason",
            ],
            "additionalProperties": False,
        },
    },
}

SCORE_FIELDS = [
    "emotion_alignment_score",
    "pedagogical_empathy_score",
    "mathematical_correctness_score",
    "error_detection_recovery_score",
    "adaptive_scaffolding_score",
    "length_control_score",
    "total_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="共感的数学対話コーパスを対話単位で評価します。")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="入力JSONL")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="評価結果JSONL")
    parser.add_argument("--model", default="gpt-5.4", help="Judgeモデル名")
    parser.add_argument("--limit", type=int, help="先頭から評価する最大件数")
    parser.add_argument("--max-retries", type=int, default=3, help="API呼び出しの最大試行回数")
    parser.add_argument("--overwrite", action="store_true", help="既存結果を削除して最初から評価")
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries, overriding stale inherited values."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("status") == "completed":
                completed.add(item["source_id"])
    return completed


def format_dialogue(session: dict) -> str:
    profile = json.dumps(session.get("student_profile", {}), ensure_ascii=False)
    lines = [
        f"【source_id】{session.get('source_id', 'unknown')}",
        f"【問題】{session.get('problem', '')}",
        f"【生徒プロフィール】{profile}",
        f"【is_completed】{session.get('is_completed', False)}",
        "【対話】",
    ]
    for turn in session.get("conversation", []):
        role = turn.get("role", "unknown")
        if role == "teacher":
            lines.extend([
                f"teacher thought_process: {turn.get('thought_process', '')}",
                f"teacher student_emotion: {turn.get('student_emotion', '')}",
                f"teacher next_step_plan: {turn.get('next_step_plan', '')}",
                f"teacher utterance: {turn.get('content', '')}",
            ])
        else:
            lines.append(f"student: {turn.get('content', '')}")
    return "\n".join(lines)


def validate_evaluation(evaluation: dict) -> None:
    calculated_total = sum(evaluation[field] for field in SCORE_FIELDS[:-1])
    if evaluation["total_score"] != calculated_total:
        raise ValueError(
            f"total_scoreが内訳と一致しません: {evaluation['total_score']} != {calculated_total}"
        )


def evaluate_session(
    client: OpenAI,
    model: str,
    system_prompt: str,
    session: dict,
    max_retries: int,
) -> dict:
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": format_dialogue(session)},
                ],
                response_format=EVALUATION_SCHEMA,
                temperature=0.0,
            )
            evaluation = json.loads(response.choices[0].message.content)
            validate_evaluation(evaluation)
            return evaluation
        except Exception as exc:
            last_error = exc
            if getattr(exc, "status_code", None) in {401, 403}:
                raise
            if attempt + 1 < max_retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"{max_retries}回の試行に失敗しました: {last_error}") from last_error


def print_summary(output_path: Path) -> None:
    rows = [row for row in load_jsonl(output_path) if row.get("status") == "completed"]
    if not rows:
        return
    print("\n=== 評価結果サマリー ===")
    print(f"評価完了: {len(rows)}件")
    for field in SCORE_FIELDS:
        values = [row["evaluation"][field] for row in rows]
        print(f"{field}: {sum(values) / len(values):.2f}")
    recommendations = Counter(row["evaluation"]["recommendation"] for row in rows)
    print("recommendation:", dict(recommendations))
    critical_count = sum(row["evaluation"]["critical_failure"] for row in rows)
    print(f"critical_failure: {critical_count}件 ({critical_count / len(rows) * 100:.1f}%)")


def main() -> None:
    args = parse_args()
    load_env_file(ENV_FILE)
    if not args.input.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {args.input}")
    if not JUDGE_PROMPT.exists():
        raise FileNotFoundError(f"Judgeプロンプトが見つかりません: {JUDGE_PROMPT}")
    if not os.getenv("GPT_API_KEY"):
        raise RuntimeError("環境変数 GPT_API_KEY を設定してください。")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limitには1以上の整数を指定してください。")

    sessions = load_jsonl(args.input)
    if args.limit is not None:
        sessions = sessions[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        args.output.write_text("", encoding="utf-8")
    completed_ids = load_completed_ids(args.output)
    pending = [session for session in sessions if session.get("source_id") not in completed_ids]

    print(f"入力: {len(sessions)}件 / 評価済み: {len(completed_ids)}件 / 今回評価: {len(pending)}件")
    if not pending:
        print_summary(args.output)
        return

    client = OpenAI(api_key=os.environ["GPT_API_KEY"])
    system_prompt = JUDGE_PROMPT.read_text(encoding="utf-8")

    for session in tqdm(pending, desc="対話品質評価"):
        source_id = session.get("source_id", "unknown")
        try:
            evaluation = evaluate_session(client, args.model, system_prompt, session, args.max_retries)
            result = {
                "source_id": source_id,
                "status": "completed",
                "is_completed": session.get("is_completed", False),
                "turn_count": sum(1 for turn in session.get("conversation", []) if turn.get("role") == "teacher"),
                "evaluation": evaluation,
            }
        except Exception as exc:
            if getattr(exc, "status_code", None) in {401, 403} or "Error code: 401" in str(exc) or "Error code: 403" in str(exc):
                raise RuntimeError(
                    "API認証に失敗しました。GPT_API_KEYを確認してから再実行してください。"
                ) from exc
            tqdm.write(f"[ERROR] {source_id}: {exc}")
            result = {"source_id": source_id, "status": "error", "error": str(exc)}

        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    print_summary(args.output)


if __name__ == "__main__":
    main()
