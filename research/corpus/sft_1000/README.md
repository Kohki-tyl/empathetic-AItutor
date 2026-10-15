# 日本語SFT用の3条件・各1,000対話

analysis付きEduEmpathyの別条件は [eduempathy_analysis](eduempathy_analysis/README.md) を参照。同じ1,000対話・900/100分割を維持し、800対話版と同じプロンプトとanalysis/final形式で学習する。

2026-10-15作成。リポジトリ内の既存コーパスから作成し、元データは変更していない。

| 条件 | フォルダ | 学習 | 検証 | assistant発話数（全体） |
|---|---|---:|---:|---:|
| Math-SFT | [math](math/README.md) | 900 | 100 | 6,407 |
| Empathy-SFT | [empathy](empathy/README.md) | 900 | 100 | 2,052 |
| EduEmpathy-SFT | [eduempathy](eduempathy/README.md) | 900 | 100 | 3,633 |

1,000件は学習・検証の合計である。教師token数は統一していない。
math・empathyは既存の日本語翻訳1,000件と固定splitを保持した。
eduempathyは `1300_empathetic_dialogues.jsonl` から選定し、seed 42で900/100に分割した。
ユーザー指定の「1300_empathetic_dialogues2」は、該当名のファイルがないため既存の1,300件の生成コーパスを指すものとして扱った。

## 共通の学習形式

通常応答の `messages` 形式、assistant-only loss。教師のthought_process・感情ラベル・roadmap・next_step_planは学習入力に含めない。
3条件とも同じsystem promptと基盤モデルを使い、それぞれ元モデルから独立に学習する。
基盤モデルは `tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5`、revisionは既存v4と同じ固定値。
LoRA・4 epochs・学習率5e-5・最大長8192等も既存v4設定を引き継いだ。

各フォルダに以下を保存している。

- `data.jsonl`: 学習器が読み込む全1,000件。明示splitを持つ。
- `train.jsonl` / `validation.jsonl`: 学習900件／検証100件の分割コピー。
- `config.json`: データhash、件数、学習設定。
- `manifest.json`: 出典hash・採用ID・分割・件数。
- `train.py`: 共通学習器への実行入口。リポジトリの `pipelines/sft/v4` が必要。

eduempathyには `quality_metadata.json` と `surface_changes.json` も保存している。
内容監査keep・完了・重大な問題なし・誤答肯定なしの1,106件から、厳格候補772件と表記条件を緩めた228件を選んだ。
本文の制御文字を正規化した214件は、数式の意味の復元を保証できず、正規化後の内容再監査も未実施である。
ユーザーが品質条件を緩めて1,000件を選ぶ方針を指定したことを来歴に記録した。

## 実行

リポジトリルートから実行する。最初にGPU環境用の依存関係を用意する。

```bash
pip install -r pipelines/sft/v4/requirements.txt
python research/corpus/sft_1000/math/train.py --preflight-only
python research/corpus/sft_1000/math/train.py --resume auto
```

empathy・eduempathyもフォルダ名を置き換えてそれぞれ実行する。
各フォルダの `outputs/` に学習結果を出力する。`--structure-only` でGPUなしの構造検査を再実行できる。
ABCIでは既存 `pipelines/sft/baselines/run_abci.pbs` の `SFT_CONFIG` に各フォルダのconfigの絶対パスを渡せる。

## 確認済みと未実施

全3条件のrole順・本文非空・制御文字なし・ID一意性・groupの学習検証間非重複・900/100分割・データhashを確認した。
共通学習器を実際の入口から起動した構造検査結果を [structure_checks.json](structure_checks.json) に保存した。
ローカル環境にはtransformersがないためtokenizerの最大長検査は未実施。実学習も未実施。
GPU環境でpreflightを行い、最大長超過がないことを確認してから学習する。暗黙の切り詰めは行わない。
翻訳・生成品質には機械処理の限界があり、全件の人手レビュー済みデータではない。
