#!/bin/bash
# ファイル名: serve_finetuned.sh

# 仮想環境の有効化
source .venv/bin/activate

# オリジナルモデル（マージ済み）を指定してvLLMを起動
python -m vllm.entrypoints.openai.api_server \
    --model ./models/Swallow-70B-MathTutor-v1 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --port 8000 > vllm_eval.log 2>&1 &

echo "ファインチューニング済みモデルのvLLMサーバーをバックグラウンドで起動しました。"
echo "ログを確認するには: tail -f vllm_eval.log"