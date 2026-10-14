# math 日本語SFT

全1,000対話（学習900・検証100）。`data.jsonl`に明示splitを持ち、学習処理はこの区分を維持する。
`train.jsonl`と`validation.jsonl`は同じ内容の分割コピー。データの追加生成や再翻訳はしていない。
発話本文を通常応答形式で学習し、lossはassistantのみに適用する。3条件の比較では教師の内部推論を学習対象に含めない。

リポジトリルートから実行する（共通の `pipelines/sft/v4` も必要）。

```bash
python research/corpus/sft_1000/math/train.py --structure-only
python research/corpus/sft_1000/math/train.py --preflight-only
python research/corpus/sft_1000/math/train.py --resume auto
```

依存関係は `pipelines/sft/v4/requirements.txt`。基盤モデル・revision・LoRA等は既存v4設定を引き継ぐ。
実学習はCUDA/BF16対応GPUが必要。tokenizerでの長さ検査はpreflightで行い、長すぎる対話は暗黙に切り詰めず停止する。
1,000件は学習・検証の合計。対話数をそろえても教師token数は一致しない。
来歴・入力hash・採用IDは `manifest.json` を参照。機械翻訳の意味の忠実性は全件の人手レビュー前。
