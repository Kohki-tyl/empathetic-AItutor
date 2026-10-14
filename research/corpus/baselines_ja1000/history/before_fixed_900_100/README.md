# 日本語比較学習用コーパス（各1,000対話）

| SFT条件 | 学習入力 | 学習対話 | 検証対話 | 教師発話数 |
|---|---|---:|---:|---:|
| Math-SFT | [math_ja_1000.jsonl](math_ja_1000.jsonl) | 894 | 106 | 6,407 |
| Empathy-SFT | [empathy_ja_1000.jsonl](empathy_ja_1000.jsonl) | 882 | 118 | 2,052 |

GPT-6 Luna (`gpt-6-luna`) で機械翻訳した各1,000対話。非復元ランダム抽出の seed は42。1,000件は学習・検証の合計であり、公式 test は含めない。原文の `id`・`group_id`・`split`・role を保持している。学習には各行の `messages` を使い、`split` に従って分割する。教師発話は通常の応答形式で、既存 v3 の analysis/final 形式とは異なる。

原文との対応、数値・保護対象、対話構造を機械検査済み。意味の忠実性は全件の人手レビュー前である。数値表記等の確認候補は MathDial 60対話、EmpatheticDialogues 3対話で、必ずしも誤訳を意味しない。

## 原文・来歴・検査資料

- `math_en_1000.jsonl` / `empathy_en_1000.jsonl`: 抽出した英語原文（SFT用に正規化した対話）
- `source_manifest.json`: 取得元、MathDial commit、入力hash、除外理由、抽出ID
- `math_ja_1000.manifest.json` / `empathy_ja_1000.manifest.json`: 翻訳設定、処理件数、最終出力hash
- `empathy_typography_ja.manifest.json`: チャット表記 `y0u` を数値と誤認した1対話の再翻訳記録
- `math_audit.json` / `empathy_audit.json`: 機械検査結果と確認候補
- `math_review.md` / `empathy_review.md`: 確認候補とランダム10対話の対訳
- `empathy_numeric_review.json`: 数字表記の変化3対話についての確認記録

作業出力 `data/baselines/ja1000/` から複製し、各ファイルのSHA-256一致を確認した。マニフェスト内の作業パスは翻訳実行時の履歴である。最終コーパスはこのフォルダーの JSONL を利用する。

再実行・学習設定生成は [翻訳手順](../../../pipelines/sft/baselines/TRANSLATION.md) と [学習手順](../../../pipelines/sft/baselines/README.md) を参照する。`prepare.py build` の `--math` / `--empathy` へ、このフォルダーの日本語JSONLを指定できる。
