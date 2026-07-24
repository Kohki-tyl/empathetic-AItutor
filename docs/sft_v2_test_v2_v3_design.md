# SFT v2・テストv2/v3分離型評価設計

## 目的

v2では、品質評価が `keep` の教師対話だけでSFTを行う。また、教師モデルの変更が生徒役へ波及する交絡をなくすため、評価対象の教師と固定した生徒シミュレーターを別モデル・別推論サーバーに分離する。

主比較は次の3条件とする。

| 条件 | 教師モデル | 生徒モデル |
| --- | --- | --- |
| Base | SFT前モデル | 未SFT Swallow 8B |
| v1 | v1 SFTモデル | 未SFT Swallow 8B |
| v2 | Keep-only SFTモデル | 未SFT Swallow 8B |

問題、類似問題、Student Simulatorのcheckpoint、プロンプト、profile割当、seed、temperature、最大ターン数、Judgeを全条件で固定し、教師checkpointだけを変更する。

## ファイル

- `pipelines/sft/v2/data/`: SFT v2の学習データとManifest
- `pipelines/sft/v2/prompts/`: SFT v2のCoT教師プロンプト
- `pipelines/model_evaluation/test_v2/`: プロファイル更新テストのスクリプトとプロンプト
- `pipelines/model_evaluation/test_v3/`: インコンテキスト学習テストのスクリプトとプロンプト

## Student Simulator

各ターンで生徒モデルは、`state_after`、`state_update_reason`、`utterance`をJSONで返す。評価スクリプトは教師へ`utterance`だけを渡し、内部状態は次の生徒ターンまで保持する。

内部状態は次を含む。

- 理解度（0〜4）
- 確信度（0〜1）
- 現在の誤概念
- 感情状態
- 獲得した知識
- 未解決事項

Phase 2は別々のテストバージョンとして実施する。

| テスト | 引き継ぐ情報 | 測定対象 |
| --- | --- | --- |
| テストv2 | 元プロファイルとPhase 1終了時の学習状態 | 明示的に更新された生徒状態による転移 |
| テストv3 | 元問題とPhase 1の対話全文 | 対話例からのインコンテキスト学習 |

テストv2では対話全文を渡さず、テストv3では更新後の内部状態を明示的に渡さない。教師モデル、固定生徒モデル、問題、profile割当、seed、生成パラメータを揃えて比較する。

## CoT付きv2 SFT

CoT付き条件では、元コーパスの教師ターンから次の3項目を学習する。

- 認知状態: 生徒の理解、つまずき、誤概念
- 感情状態: 10種類の感情ラベルと判断根拠
- 次の一歩: 今回提示する足場かけ

教師出力は `<analysis>...</analysis><final>...</final>` とし、評価時に生徒へ渡すのは `<final>`だけとする。CoTなしv2とCoT付きv2は別checkpointとして学習し、CoTの有無による効果を比較する。

```bash
python pipelines/sft/v2/prepare_v2_cot_sft_dataset.py
```

## ABCIでの実行

### 1. v2 SFTデータの再生成と確認

```bash
python pipelines/sft/v2/prepare_keep_only_sft_dataset.py
```

SFT v2の学習には `pipelines/sft/v2/data/v2_keep_only_sft_train.jsonl` を指定する。

### 2. 推論サーバー

教師と生徒を別ポートで起動する。GPU割当や起動オプションはABCIのジョブ構成に合わせる。

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/teacher-checkpoint \
  --served-model-name teacher-under-test \
  --port 8000
```

```bash
python -m vllm.entrypoints.openai.api_server \
  --model tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5 \
  --served-model-name tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5 \
  --port 8001
```

Student Simulatorには未SFTの`tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5`を使用する。教師用SFT checkpointは使わず、全教師条件で同一のcheckpointを使う。

### 3. 接続確認用パイロット

Judge APIを呼ばず、2問だけで対話形式を確認する。

```bash
python pipelines/model_evaluation/test_v2/generate_profile_update_dialogues.py \
  --teacher-model teacher-under-test \
  --teacher-system-prompt pipelines/sft/v2/prompts/v2_cot_teacher_system.txt \
  --limit 2 \
  --overwrite
```

```bash
python pipelines/model_evaluation/test_v3/generate_in_context_dialogues.py \
  --teacher-model teacher-under-test \
  --teacher-system-prompt pipelines/sft/v2/prompts/v2_cot_teacher_system.txt \
  --limit 2 \
  --overwrite
```

### 4. 20問パイロット

ABCIの外部接続でproxyが必要な場合は、`JUDGE_PROXY`を設定する。

```bash
export GPT_API_KEY="..."
export JUDGE_PROXY="http://proxy.abci.local:3128"

python pipelines/model_evaluation/test_v2/generate_profile_update_dialogues.py \
  --teacher-model teacher-under-test \
  --limit 20 \
  --seed 42 \
  --overwrite
```

生成ログを確認した後、Judge評価を別に実行する。

```bash
python pipelines/model_evaluation/test_v2/evaluate_profile_update_dialogues.py \
  --overwrite
```

テストv3も`generate_in_context_dialogues.py`と`evaluate_in_context_dialogues.py`で同様に実行する。

### 5. 主実験

各教師条件で出力先だけを変え、同一seedで実行する。

```bash
python pipelines/model_evaluation/test_v2/generate_profile_update_dialogues.py \
  --teacher-model base-teacher \
  --seed 42 \
  --output experiments/test_v2/base/dialogues.jsonl
```

```bash
python pipelines/model_evaluation/test_v2/evaluate_profile_update_dialogues.py \
  --input experiments/test_v2/base/dialogues.jsonl \
  --output experiments/test_v2/base/evaluated_results.jsonl
```

```bash
python pipelines/model_evaluation/test_v3/generate_in_context_dialogues.py \
  --teacher-model base-teacher \
  --seed 42 \
  --output experiments/test_v3/base/dialogues.jsonl
```

中断時は同じコマンドを再実行すれば、既存の`run_id`を読み取り完了済み問題をスキップする。最初からやり直す場合だけ`--overwrite`を付ける。各出力の隣に`.manifest.json`が作成され、モデル名、URL、temperature、seedなどが保存される。

複数seedでは、`--seed 42`、`--seed 43`、`--seed 44`のように変更し、出力ファイルもseedごとに分ける。

## パイロットの合格条件

20問のログを人手でも確認し、次を満たしてから主実験へ進む。

- 教師口調の混入がほぼない
- 未提示の高度な公式を突然使わない
- 一度の弱いヒントで不自然に完全理解しない
- 生徒発話が原則1〜2文で、過度にラフでない
- `state_after`と実際の発話に矛盾がない
- テストv2ではPhase 2入力に対話全文が含まれていない
- テストv3ではPhase 2入力に更新後の内部状態が直接含まれていない

教師の対話品質は次の2タスクに分けて独立採点する。

| タスク | 評価項目 | 配点 |
| --- | --- | ---: |
| 共感指導 | 感情認識、認知的共感、情緒的支援と心理的安全性 | 30点 |
| 数学的指導 | 数学的正確性、誤りの診断と回復、適応的足場かけ、理解確認、認知負荷制御 | 50点 |

合計80点に加え、誤答追認回数と答えの直接提示回数を診断値として保存する。Phase 2の数学的正答判定と、生徒の教師口調、知識違反、文体違反、不自然な状態更新、無条件同意は別評価として記録する。

## 解釈上の注意

使用中の問題には中学校の学習範囲を超えるものが含まれる。教師モデル間の対応比較は可能だが、「中学生に対する絶対的な指導性能」として解釈する場合は、別途、学習指導要領に合う問題だけを抽出した評価セットを用意する。
