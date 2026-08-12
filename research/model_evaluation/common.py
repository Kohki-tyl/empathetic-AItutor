from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


AXES = (
    "mathematical_accuracy",
    "error_diagnosis_recovery",
    "instruction_completion",
    "scaffolding",
    "emotional_support",
    "emotion_recognition",
)
INSTRUCTION_AXES = AXES[:3]
EMPATHY_AXES = AXES[3:]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: JSON objectではありません")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def resolve_path(config_path: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def env_value(name: str, *, fallback: str | None = None) -> str:
    value = os.getenv(name)
    if value:
        return value
    if fallback:
        fallback_value = os.getenv(fallback)
        if fallback_value:
            return fallback_value
    raise RuntimeError(f"環境変数 {name} が設定されていません")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def import_openai():
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - 実API環境だけで通る分岐
        raise RuntimeError("openaiパッケージをインストールしてください") from exc
    return OpenAI


def openai_client(*, base_url: str, api_key_env: str, fallback_key_env: str | None = None,
                  timeout: float = 180.0, default_key: str | None = None):
    OpenAI = import_openai()
    try:
        key = env_value(api_key_env, fallback=fallback_key_env)
    except RuntimeError:
        if default_key is None:
            raise
        key = default_key
    return OpenAI(api_key=key, base_url=base_url, timeout=timeout)


def _message_text(response: Any) -> str:
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("モデル応答が空です")
    return content.strip()


def call_json_model(client: Any, *, model: str, messages: list[dict[str, str]],
                    schema: dict[str, Any], temperature: float, max_tokens: int,
                    seed: int, top_p: float = 1.0, reasoning_effort: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "max_completion_tokens": max_tokens,
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    raw = _message_text(client.chat.completions.create(**kwargs))
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("構造化応答がJSON objectではありません")
    return value


def call_text_model(client: Any, *, model: str, messages: list[dict[str, str]],
                    temperature: float, max_tokens: int, seed: int, top_p: float = 1.0) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=max_tokens,
    )
    return _message_text(response)


def retry_call(call, *, attempts: int):
    if attempts < 1:
        raise ValueError("attemptsは1以上である必要があります")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return call(attempt), attempt
        except Exception as exc:  # API・形式エラーだけを同じ規則で再試行する
            last_error = exc
    assert last_error is not None
    raise last_error


def validate_nonempty_utterance(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("utteranceは空でない文字列である必要があります")
    return value.strip()


def extract_visible_teacher_utterance(raw: str, completion_marker: str) -> tuple[str, bool]:
    """内部analysisを捨て、生徒へ提示するfinalだけを返す。"""
    text = raw.strip()
    lower = text.lower()
    final_start = lower.rfind("<final>")
    final_end = lower.rfind("</final>")
    if final_start >= 0 and final_end > final_start:
        text = text[final_start + len("<final>"):final_end].strip()
    else:
        if "<analysis" in lower or "</analysis>" in lower or "<final" in lower or "</final>" in lower:
            raise ValueError("教師のanalysis/finalタグが不完全で、可視発話を安全に分離できません")
    completed = completion_marker in text
    visible = text.replace(completion_marker, "").strip()
    if not visible:
        raise ValueError("教師の可視発話が空です")
    return visible, completed


def visible_dialogue_text(dialogue: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in dialogue:
        role = item.get("role")
        if role not in {"student", "teacher"}:
            continue
        content = validate_nonempty_utterance(item.get("content"))
        label = "生徒" if role == "student" else "教師"
        lines.append(f"{label}: {content}")
    return "\n\n".join(lines)


def axis_scores(evaluation: dict[str, Any]) -> dict[str, float | None]:
    axes = evaluation.get("axes")
    if not isinstance(axes, dict):
        raise ValueError("evaluation.axesがありません")
    result: dict[str, float | None] = {}
    for name in AXES:
        item = axes.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"評価軸{name}がありません")
        score = item.get("score")
        if score is None:
            result[name] = None
        elif isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 10:
            raise ValueError(f"評価軸{name}の得点が不正です: {score!r}")
        else:
            result[name] = float(score)
    return result


def mean_excluding_na(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return statistics.mean(valid) if valid else None


def overall_score(scores: dict[str, float | None]) -> float | None:
    mean = mean_excluding_na(scores.values())
    return None if mean is None else mean * 6.0


def group_score(scores: dict[str, float | None], names: Iterable[str]) -> float | None:
    return mean_excluding_na(scores[name] for name in names)


def rounded(value: float | None, digits: int = 4) -> float | None:
    return None if value is None or math.isnan(value) else round(value, digits)


def describe(values: Iterable[float | None]) -> dict[str, Any]:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return {"n": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "n": len(valid),
        "mean": rounded(statistics.mean(valid)),
        "median": rounded(statistics.median(valid)),
        "minimum": rounded(min(valid)),
        "maximum": rounded(max(valid)),
    }
