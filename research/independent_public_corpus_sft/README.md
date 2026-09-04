# 公開対話コーパス別・独立SFTデータ準備

EmpatheticDialoguesから汎用的な共感応答を学習するモデルと、MathDialから数学教育対話を学習するモデルを混ぜずに作るための前処理である。公式train splitから各500対話をseed 42で固定抽出し、`gpt-5.6-luna`で日本語へ機械翻訳して、別々のSFT JSONLを生成する。

## データ源と利用条件

- EmpatheticDialogues: `facebook/empathetic_dialogues` train split。CC BY-NC 4.0（非商用条件あり）。
- MathDial: `eth-nlped/mathdial` の `data/train.jsonl`。CC BY-SA 4.0。

ライセンス条件は派生翻訳データと学習成果物の利用・配布前に必ず再確認する。rawデータはGit管理対象外である。抽出済み原文、選択ID、入力hashは `data/sample_manifest.json` に保存する。

## 現在の状態

公式trainデータは `raw/` に取得済みで、抽出済み原文は次の2ファイルである。

- `data/empathetic_dialogues_500_source.jsonl`
- `data/mathdial_500_source.jsonl`

翻訳APIはまだ実行していない。翻訳結果は1対話ごとに追記され、停止後の再実行では完了済みIDをスキップする。

## セットアップと検査

リポジトリルートの `.env` に `OPENAI_API_KEY` または `GPT_API_KEY` を設定する。

```bash
cd research/independent_public_corpus_sft
python -m pip install -r requirements.txt
python prepare_samples.py
python -m unittest discover -s tests -v
python translate_samples.py empathetic_dialogues --dry-run
python translate_samples.py mathdial --dry-run
```

## 翻訳実行

まず各コーパス1件でAPI疎通・品質・JSON形式を確認する。

```bash
python translate_samples.py empathetic_dialogues --limit 1
python translate_samples.py mathdial --limit 1
```

出力を目視確認後、残りを再開実行する。

```bash
python translate_samples.py empathetic_dialogues
python translate_samples.py mathdial
python validate_outputs.py
```

既定では8並列で実行する。レート制限が厳しい環境では、たとえば `--workers 2` を指定する。各完了結果はmain threadから1行ずつ追記されるため、途中停止後も同じコマンドで再開できる。

生成物は次のとおり。

| 用途 | EmpatheticDialogues | MathDial |
| --- | --- | --- |
| API応答・来歴 | `data/empathetic_dialogues_500_ja.jsonl` | `data/mathdial_500_ja.jsonl` |
| SFT入力 | `data/empathetic_dialogues_500_ja_sft.jsonl` | `data/mathdial_500_ja_sft.jsonl` |

EmpatheticDialoguesの末尾に応答のないuser発話がある場合、翻訳記録には保持し、SFT入力からだけ除く。MathDial原文は全件teacher開始なので、収録時の会話開始用teacher発話を翻訳記録には保持しつつSFT入力から除き、最初の実際のstudent発話から開始する。同一話者の連続発話はSFT入力で結合し、末尾の未応答student発話はSFT入力からだけ除く。MathDialの問題・正答解法・生徒プロフィール・初期誤答は教師向け事前情報としてsystem promptに配置する。

## 翻訳方針

- 発話数・role・順序を固定したStructured Outputsを使用する。
- EmpatheticDialoguesでは感情、共感度、口調、対人距離を保存する。
- MathDialでは数式、数値、単位、誤答、教師方略、dialogue actを保存し、原文の誤りを勝手に訂正しない。
- `reasoning_effort=low`。モデル、retry、最大出力tokenは `config.json` で固定する。

500件完了後は、翻訳品質の標本監査とSFT token長監査を行ってから、既存 `pipelines/sft/v4/train_v4_sft.py` 用の2つの学習configを確定する。

全件の原文hash・ID・API来歴・翻訳turn構造・SFT構造と成果物hashは、`python validate_outputs.py` が `data/translation_manifest.json` に記録する。
