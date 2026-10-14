# eduempathy 日本語SFT

全1,000対話（学習900・検証100）。`data.jsonl`に明示splitを持ち、学習処理はこの区分を維持する。
`train.jsonl`と`validation.jsonl`は同じ内容の分割コピー。データの追加生成や再翻訳はしていない。
発話本文を通常応答形式で学習し、lossはassistantのみに適用する。3条件の比較では教師の内部推論を学習対象に含めない。

リポジトリルートから実行する（共通の `pipelines/sft/v4` も必要）。

```bash
python research/corpus/sft_1000/eduempathy/train.py --structure-only
python research/corpus/sft_1000/eduempathy/train.py --preflight-only
python research/corpus/sft_1000/eduempathy/train.py --resume auto
```

依存関係は `pipelines/sft/v4/requirements.txt`。基盤モデル・revision・LoRA等は既存v4設定を引き継ぐ。
実学習はCUDA/BF16対応GPUが必要。tokenizerでの長さ検査はpreflightで行い、長すぎる対話は暗黙に切り詰めず停止する。
1,000件は学習・検証の合計。対話数をそろえても教師token数は一致しない。
来歴・入力hash・採用IDは `manifest.json` を参照。機械翻訳の意味の忠実性は全件の人手レビュー前。

## 生成対話の選定条件

元の1,300対話から、内容監査keep・完了・重大失敗なし・誤答肯定0の1,106対話を候補とした。
前回の厳格な採用候補772件を全件含め、表記条件を緩めた228件を追加して1,000件にした。
要確認・rejectという内容判定の対話は採用していない。教師の内部フィールドは学習入力から除外する。
初回user入力には問題文と最初の生徒発話をまとめて含める。

制御文字のある本文は既知のLaTeXエスケープを復元したうえで残存C0/DELを除去した。
修正前後は`surface_changes.json`、元の品質・変更有無は`quality_metadata.json`に記録した。
これは形式の正規化であり、数式の意味や元表記の復元を保証しない。正規化後の内容再監査は未実施。
生成対話の元データと監査結果は変更していない。翻訳の注意書きは元問題に関するもので、対話本文は生成データ。
