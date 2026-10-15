# 既存800対話の続き：追加500対話

`pipelines/corpus_creation/800_empathetic_dialogues.jsonl` を元に、追加翻訳した500問から対話801〜1300を生成した。元の800対話は変更していない。

| 項目 | 件数 |
|---|---:|
| 追加対話 | 500 |
| 教師モデルが完了と判定 | 484 |
| 最大15往復で未完了 | 16 |
| 追加分の教師発話 | 2,167 |
| 既存分との合計 | 1,300 |

今回返されたmodel名は教師 `gpt-5.4-2026-03-05`、生徒 `gpt-5.4-mini-2026-03-17`。元800対話との一致、新旧の問題ID非重複、追加500問との一対一対応、対話のrole順と発話数を検査済み。

- `additional_500_empathetic_dialogues_801_1300.jsonl`: 追加500対話
- `1300_empathetic_dialogues.jsonl`: 既存800対話＋追加500対話
- `translated_additional_500_math_gpt6_luna.jsonl`: 今回の対話生成に使った問題・参照解答
- `generation_manifest.json`: 入力hash、生成設定、応答モデル名、件数、実行環境、token使用量
- `structural_audit.json`: 件数・問題対応・role順・元800件との一致・重複検査
- `prompts/`: 使用した教師・生徒プロンプト、生徒プロフィールのスナップショット

## 生成条件

既存 `generate_corpus.py` のプロンプト、教師JSONスキーマ、対話履歴の渡し方を引き継ぐ。教師は `gpt-5.4`・temperature 0.2、生徒は `gpt-5.4-mini`・temperature 0.8。両方で reasoning effort を既定相当の `none` に明示し、追加の出力上限は4,096 tokens、最大15往復とする。既存800対話の実行時モデルは各行に記録されていないため、過去の実モデルsnapshotと完全に同一であることまでは保証しない。今回の実際の応答model名・応答IDは新規各行に保存する。

生徒プロフィールは通し位置800を起点に10種類を巡回し、初期感情も同じ位置を起点に Frustrated / Confusion / Engaged の3種類を巡回する。追加問題の順序は翻訳ファイルの順序を保持する。教師・生徒へ参照解答は渡さない。過去の1対話を延長するのではなく、新しい問題について500対話を追加している。

各JSONL行は既存と同じ `problem`・`student_profile`・`is_completed`・`conversation`・`source_id` を持ち、新規行には `generation_metadata` を追加している。`conversation` は student / teacher の交互で、teacher には既存と同じ `thought_process`・`student_emotion`・`roadmap_breakdown`・`next_step_plan`・`content` が入る。

教師が完了と判断するか15往復へ到達すると1対話を確定する。15往復で未完了の場合は `is_completed=false` のまま残す。API中断で止まった途中の対話は確定コーパスに含めず、発話単位のチェックポイントから再開する。JSONで `\frac` などのエスケープが制御文字へ変換された場合は既知のLaTeXコマンドへ復元し、非表示の区切り文字は除去する。修正したフィールドと修正前文字列のhashをAPI記録へ残す。最終検査で追加の表記修正を行ったフィールドは115件で、各行の `generation_metadata.surface_postprocessing` に記録した。修正前の追加分は作業ディレクトリの `additional_500_before_surface_cleanup.jsonl` に保存している。

生成後、追加500対話全件の内容監査を実施した。keep 413件、review 23件、reject 64件で、制御文字の残存も除外したSFT採用候補は390件。詳細は[監査報告](audit/README.md)を参照。SFT学習ファイルや分割の作成は未実施。`is_completed=true` は教師モデルの判断であり、独立した正答検査の結果ではない。今後、必要な監査・修正を経てSFT用の正例を選ぶ。

元の翻訳1,000問を使う既存評価用プールと今回の追加500問には問題IDの重複がない。追加問題自体の元MATHにおけるtrain/test由来は[基本翻訳の記録](../math_additional_500_gpt6_luna/README.md)を参照する。

## 実行・再開

```bash
python pipelines/corpus_creation/continue_corpus.py --workers 64
python pipelines/corpus_creation/finalize_continuation.py
```

`OPENAI_API_KEY` または `GPT_API_KEY` が必要。同じ設定・入力・プロンプトで再開すれば、保存済みの発話や完成した対話を再生成しない。初回の動作確認には `--limit 1 --workers 1` を使用できる。作業記録は `data/corpus_continuation_500/` に保存する。採用応答のtoken使用量は集計するが、API失敗や保存前の中断で失われた応答の課金総量ではない。

## 監査の補足

2026-10-14の追加検査では、従来のASCII制御文字検査の対象外だったDEL（U+007F）が29対話・44フィールド・151文字見つかった。上記の表記修正で全問題を除去できたわけではない。元データは監査後も変更せず、該当IDを監査結果に記録した。
