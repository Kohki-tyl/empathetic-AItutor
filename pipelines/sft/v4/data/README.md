# v4 SFT入力

このフォルダーには、表面修正後の全体再監査とSFT構造ゲートを通過した104対話を`v4_sft.jsonl`として同梱している。元データは`corpus_creation/v4`の`v4_all_keep_sft_eligible.jsonl`である。

- レコード: 104対話
- 教師学習ターゲット: 711ターン
- SHA-256: `09bda3da746079639f48146c7fecfdc7c40480d34caaceb3ff9ed229700c4b9d`
- 学習・検証分割: seed 42で94件 / 10件

`source_corpus_manifest.json`、`SOURCE_CORPUS_REPORT.md`、`SOURCE_CORPUS_REQUIREMENTS.md`に、選択経路、監査結果、採択要件、元ファイルhashを保存している。

学習スクリプトは次を開始前に検証する。

- 104レコードであり、configのSHA-256と一致する
- IDが欠損・重複していない
- `system → user → assistant`の交互順である
- 全assistant発話に`<analysis>`と`<final>`が一組ずつある
- chat template適用後の系列長が8,192トークン以内である
- assistant以外のトークンが損失対象から除外され、`<analysis>` 0.25・`<final>` 1.0の重み境界を決定できる

入力を置き換える場合は`expected_records`と`dataset_sha256`を意図的に更新し、既存の`output_dir`を再利用しない。
