# v2 SFT・分離型評価設計

## 目的

v2では、品質評価が `keep` の教師対話だけでSFTを行う。また、教師モデルの変更が生徒役へ波及する交絡をなくすため、評価対象の教師と固定した生徒シミュレーターを別モデル・別推論サーバーに分離する。

主比較は次の3条件とする。

| 条件 | 教師モデル | 生徒モデル |
| --- | --- | --- |
| Base | SFT前モデル | 固定Student Simulator v2 |
| v1 | v1 SFTモデル | 固定Student Simulator v2 |
| v2 | Keep-only SFTモデル | 固定Student Simulator v2 |

問題、類似問題、Student Simulatorのcheckpoint、プロンプト、profile割当、seed、temperature、最大ターン数、Judgeを全条件で固定し、教師checkpointだけを変更する。

## ファイル

- `v2_keep_only_sft_train.jsonl`: v2の教師SFTデータ
- `v2_keep_only_sft_manifest.json`: 抽出条件と採用ID
- `evaluate_v2_by_simulator.py`: 分離型評価スクリプト
- `prompts/v2_teacher_system.txt`: 評価時の教師プロンプト
- `prompts/v2_student_system.txt`: Phase 1の状態駆動型生徒プロンプト
- `prompts/v2_phase2_student_system.txt`: Phase 2の生徒プロンプト
- `prompts/v2_student_profiles.json`: 固定生徒profile
- `prompts/v2_student_realism_judge_system.txt`: 生徒らしさの評価基準

## Student Simulator v2

各ターンで生徒モデルは、`state_after`、`state_update_reason`、`utterance`をJSONで返す。評価スクリプトは教師へ`utterance`だけを渡し、内部状態は次の生徒ターンまで保持する。

内部状態は次を含む。

- 理解度（0〜4）
- 確信度（0〜1）
- 現在の誤概念
- 感情状態
- 獲得した知識
- 未解決事項

Phase 2にはPhase 1の対話全文を渡さない。最終内部状態だけを渡すことで、会話のコピーではなく、状態として保持された学習内容の転移を測る。

## ABCIでの実行

### 1. v2 SFTデータの再生成と確認

```bash
python pipelines/model_evaluation/prepare_keep_only_sft_dataset.py
```

v2の学習には `pipelines/model_evaluation/v2_keep_only_sft_train.jsonl` を指定する。

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
  --model /path/to/frozen-student-model \
  --served-model-name fixed-student-v2 \
  --port 8001
```

Student Simulatorには教師用SFT checkpointを使わない。全教師条件で、同一の固定checkpointを使う。

### 3. 接続確認用パイロット

Judge APIを呼ばず、2問だけで対話形式を確認する。

```bash
python pipelines/model_evaluation/evaluate_v2_by_simulator.py \
  --teacher-model teacher-under-test \
  --student-model fixed-student-v2 \
  --limit 2 \
  --skip-judges \
  --output experiments/v2_test/pilot/dialogues.jsonl \
  --overwrite
```

### 4. 20問パイロット

ABCIの外部接続でproxyが必要な場合は、`JUDGE_PROXY`を設定する。

```bash
export GPT_API_KEY="..."
export JUDGE_PROXY="http://proxy.abci.local:3128"

python pipelines/model_evaluation/evaluate_v2_by_simulator.py \
  --teacher-model teacher-under-test \
  --student-model fixed-student-v2 \
  --limit 20 \
  --seed 42 \
  --output experiments/v2_test/pilot/evaluation_results.jsonl \
  --overwrite
```

### 5. 主実験

各教師条件で出力先だけを変え、同一seedで実行する。

```bash
python pipelines/model_evaluation/evaluate_v2_by_simulator.py \
  --teacher-model base-teacher \
  --student-model fixed-student-v2 \
  --seed 42 \
  --output experiments/v2_test/base/evaluation_results.jsonl
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
- Phase 2入力に対話全文が含まれていない

自動評価では、Pedagogical Empathyと数学的正確性に加え、tutor leak、知識違反、文体違反、不自然な状態更新、無条件同意を記録する。

## 解釈上の注意

使用中の問題には中学校の学習範囲を超えるものが含まれる。教師モデル間の対応比較は可能だが、「中学生に対する絶対的な指導性能」として解釈する場合は、別途、学習指導要領に合う問題だけを抽出した評価セットを用意する。
