"""長時間の生成前にJudge APIの認証付き疎通を確認する。"""

from __future__ import annotations

import argparse
import os

import httpx
from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", default=os.getenv("JUDGE_PROXY"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--model", default=os.getenv("JUDGE_MODEL_NAME", "gpt-5.4"))
    args = parser.parse_args()
    api_key = os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("GPT_API_KEY または OPENAI_API_KEY を設定してください。")
    http_client = httpx.Client(proxy=args.proxy, timeout=args.timeout) if args.proxy else httpx.Client(timeout=args.timeout)
    try:
        client = OpenAI(api_key=api_key, http_client=http_client)
        client.models.list()
        client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_completion_tokens=16,
        )
    except Exception as exc:
        raise SystemExit(f"Judge API接続確認に失敗しました: {type(exc).__name__}: {exc}") from exc
    finally:
        http_client.close()
    print("Judge API connection check passed")


if __name__ == "__main__":
    main()
