# 各1,000対話の日本語機械翻訳

モデルは指定された `gpt-6-luna`。公式 [モデル仕様](https://developers.openai.com/api/docs/models/gpt-6-luna) と [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) に沿って Responses API を使用する。reasoning effort は `none`、store は false。翻訳専用依存関係には学習用の torch は不要。

```bash
pip install -r pipelines/sft/baselines/requirements-translation.txt
python pipelines/sft/baselines/fetch_sources.py --output data/baselines/ja1000 --sample-size 1000 --seed 42
```

MathDial は公式 train、EmpatheticDialogues は公式 train と valid を母集団とし、各1,000の完全な対話を非復元ランダム抽出する。test は含めない。形式不正の対話は抽出前に除外し、理由を `source_manifest.json` に記録する。MathDial には Student の代わりにプロフィールの生徒名を使う対話もあり、同じ生徒の発話として解析する。MathDial の qid 単位分割、EmpatheticDialogues の公式 split を抽出後も保持する。したがって1,000件は学習・検証の合計で、教師発話1,000件や学習専用1,000件ではない。

`OPENAI_API_KEY` または既存の `GPT_API_KEY` を環境変数として設定する。キーはログ・manifest に保存しない。`.env` は自動で読まない。

```bash
python pipelines/sft/baselines/translate_ja.py --input data/baselines/ja1000/math_en_1000.jsonl --output data/baselines/ja1000/math_ja_1000.jsonl --model gpt-6-luna --workers 32
python pipelines/sft/baselines/translate_ja.py --input data/baselines/ja1000/empathy_en_1000.jsonl --output data/baselines/ja1000/empathy_ja_1000.jsonl --model gpt-6-luna --workers 32
```

`--dry-run` は件数・文字数の確認だけを行う。`--limit 10` は試験用の先頭10対話を別出力へ翻訳する。ランダム抽出は前段で実施済み。API制限が厳しい場合は `--workers 8` 等へ下げる。失敗後は同じコマンドで再開でき、保存済みの対話は再課金の呼び出しをしない。中断時に処理中だったリクエストや検証失敗での再試行は追加のAPI使用量が発生しうる。

翻訳には対話の全ターンをまとめて渡し、本文中の指示は翻訳対象データとして扱う。英語の system prompt は両条件共通の固定日本語文へ置換する。数値と明示的な数式はプレースホルダーで保護し、生成後に元の文字列へ戻す。ターン順・role・id・group_id・split を保持する。学習者の誤答・教師の不適切な対応も修正せず訳し、共感や足場かけを追加しない。

JSONL は各対話の完了時に `.partial.jsonl` へ原子的に保存し、全件が完了してから最終ファイル名へ変更する。manifest に原文hash・抽出条件・model・prompt・処理件数を記録する。出力各行には応答ID・実際に返されたmodel名・使用token数・原文レコードhashを付与する。token使用量の集計は採用した応答分で、失敗した試行や停止時に処理中だった応答の課金総量ではない。モデルaliasの将来の挙動まで固定することはできない。コード修正後の再開は入力・モデル・prompt・設定がすべて同じ場合だけ許可し、旧コードhashを履歴に残す。

“a quarter” →「25セント」のように原文の英単語が数字表記に変わる場合は、自動的に誤訳とせず `translation_qa` に記録する。原文にある数字の欠落や保護対象の欠損はエラーとする。機械検査は意味の忠実性を保証しないため、学習前に特に誤答維持、通貨・単位、感情の強さを対訳で確認する。

```bash
python pipelines/sft/baselines/audit_translation.py --source data/baselines/ja1000/math_en_1000.jsonl --target data/baselines/ja1000/math_ja_1000.jsonl --report data/baselines/ja1000/math_audit.json --review data/baselines/ja1000/math_review.md
python pipelines/sft/baselines/audit_translation.py --source data/baselines/ja1000/empathy_en_1000.jsonl --target data/baselines/ja1000/empathy_ja_1000.jsonl --report data/baselines/ja1000/empathy_audit.json --review data/baselines/ja1000/empathy_review.md
python pipelines/sft/baselines/prepare.py build --math data/baselines/ja1000/math_ja_1000.jsonl --empathy data/baselines/ja1000/empathy_ja_1000.jsonl --language ja --output pipelines/sft/baselines/runs/ja1000
```

並列リクエストの成功結果は出力名に対応する `.cache/` へも保存する。一部の対話で検証が失敗しても、後続の成功した翻訳を再開時に再利用できる。`.failure.json` は調査用の不採用応答であり、学習入力には含めない。数値プレースホルダーの対応表もモデルへ渡し、序数や単位の自然な日本語化を支援する。

現在の保存済みコーパスは、翻訳後に学習900対話／検証100対話へ固定している。抽出済みの1,000対話と翻訳本文を保持し、同じ group_id を分割しない範囲で変更対話数を最小にしている。元の区分は `original_split` に残す。EmpatheticDialogues の現在の区分は公式 train/valid と完全には一致しない。

新しい翻訳出力へ同じ分割調整を適用する場合:

```bash
python pipelines/sft/baselines/fix_split.py --corpus-dir data/baselines/ja1000 --output data/baselines/fixed_900_100 --validation-count 100 --seed 42
```

出力先は新しいディレクトリを指定する。調整した英語・日本語JSONLと `fixed_split_manifest.json` が生成される。調整後の日本語JSONLを `prepare.py build` へ渡す。現在の確定版は `research/corpus/baselines_ja1000/` に保存している。

この件数指定では両条件の教師token数は一致しない。token数を統制する追加実験では `build --target-tokens N` を使用する。ただしtoken予算で学習対話を抽出し直すと900対話の条件は維持されない。3条件の学習設定は `build` コマンドで生成される。
