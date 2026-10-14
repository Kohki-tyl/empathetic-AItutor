# MATH 追加500問の基本翻訳（GPT-6 Luna）

既存の `translated_1000_math.jsonl` に重複しない500問の問題文・解答を、日本語へ機械翻訳したデータ。対話生成前の問題・解答ペアであり、教師・生徒対話のSFTコーパスではない。

## 使用するファイル

- `translated_additional_500_math_gpt6_luna.jsonl`: 新規翻訳した500問
- `translated_1500_math_mixed_translation.jsonl`: 既存1,000問と追加500問を結合した別ファイル
- `manifest.json`: 件数、重複検査、最終ファイルhash、翻訳モデル
- `selection_manifest.json`: 固定したデータセットrevision、既存原文との一致検証、選択ID、除外理由
- `math_additional_500_en.normalized.jsonl`: 翻訳対象の英語原文と出典情報
- `math_additional_500_ja.normalized_v2.jsonl` と対応する `.manifest.json`: 翻訳の作業形式、設定、API応答ID、token使用量
- `translation_audit.json` / `translation_review.md`: 機械検査と対訳確認資料

同じ新規500問と結合版は `pipelines/corpus_creation/questions/` にも保存している。既存1,000問は上書きしていない。結合版の先頭1,000問は既存データそのもので、その実行時の翻訳モデルは記録されていない。追加500問のみ GPT-6 Luna の翻訳である。

## 抽出条件

元データは [nlile/hendrycks-MATH-benchmark](https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark) の配布 `train`、revision `465bcdb36f5962aa3512891498966df785fc3c18`。既存1,000問すべての原文・解答と当該revisionの一致を確認したうえで、既存最大index 1097の続きから順番に抽出した。図を要する問題、Asymptoteを含む問題・解答、既存および追加分内の原文重複を除外して500問にしている。ランダム抽出ではない。

選択IDは `math_train_1098` から `math_train_1693` の範囲。配布側の train は元MATHのtrain/testを再編したものであり、追加500問の由来は元train 312問・元test 188問。元MATHの区分を `original_math_split`、元の問題IDを `source_unique_id` に保存している。評価データとの分離は元IDで確認する。

## 翻訳条件・形式

[GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna) (`gpt-6-luna`) の Responses API、reasoning effort `none`、temperature未指定、最大出力16,384 tokens、32並列で実行。モデルから返されたmodel名・応答ID・採用した応答のtoken使用量・原文hashを各行へ記録する。失敗したリクエストの課金総量はtoken集計に含まれない。

対話翻訳と共通の `pipelines/sft/baselines/translate_ja.py` を使用し、問題文と解答を一緒に渡して文脈を保持する。新たに解く、説明を増やす、原文の誤りを修正することを指示していない。数値、ドル記号で囲まれた数式、LaTeXの環境などをプレースホルダーで保護・復元する。数式だけの解答には不要な日本語説明を付け加えない。

基本翻訳のため、旧スキーマとの互換用 `thought_process` は空文字。既存のGPT-5.4翻訳スクリプトが使用する専門プロンプト・temperature 0.3とは同一設定ではない。今回のプロンプト本文・hash・設定は翻訳manifestに記録している。

数式・数値・原文対応とLaTeXコマンドの欠落がないことを機械検査済みで、意味の忠実性は全件の人手レビュー前である。数字表記の変化による確認候補は6問で、英単語から数字への表記変換等を含む。`translation_review.md` で確認候補とランダム抽出した10問の対訳を確認できる。

対訳確認で `math_train_1120` の日付の語順を日本語の年・月・日へ並べ替えた。数値・数式の内容は変更していない。実際の置換規則は当該行の `translation.postprocessing` とmanifestに記録している。

## 再実行

リポジトリルートで `openai>=2.40,<3` と `pyarrow` を導入し、`OPENAI_API_KEY` または `GPT_API_KEY` を環境変数へ設定する。

```bash
python pipelines/corpus_creation/prepare_additional_math.py --output-dir data/math_additional_500 --count 500
python pipelines/sft/baselines/translate_ja.py --input data/math_additional_500/math_additional_500_en.normalized.jsonl --output data/math_additional_500/math_additional_500_ja.normalized_v2.jsonl --model gpt-6-luna --workers 32
python pipelines/corpus_creation/finalize_additional_math.py --work-dir data/math_additional_500 --translated data/math_additional_500/math_additional_500_ja.normalized_v2.jsonl
```

新しい実験では別の作業ディレクトリ・出力先を指定する。翻訳の途中再開は同じコマンドで行う。既存の完成済み出力は上書きしない。
