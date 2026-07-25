# v4 SFT入力

v4コーパス作成の`finalize`で生成された`v4_sft.jsonl`をこのディレクトリへコピーする。

```bash
cp /path/to/corpus_creation/v4/data/run_100/v4_sft.jsonl data/v4_sft.jsonl
```

学習スクリプトは次を開始前に検証する。

- 100レコードである
- IDが欠損・重複していない
- `system → user → assistant`の交互順である
- 全assistant発話に`<analysis>`と`<final>`が一組ずつある
- chat template適用後の系列長が8,192トークン以内である
- assistant以外のトークンが損失対象から除外されている

入力ファイルは生成物であるため、このREADMEだけをリポジトリで管理する。

