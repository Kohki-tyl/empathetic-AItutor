# Test v4：v4生徒準拠インコンテキスト転移評価

## 目的

v4コーパス作成時と同じ生徒モデル、プロフィール、初期感情、状態更新制約を用いて教師モデルを評価する。Phase 1では教師との対話を生成し、Phase 2ではPhase 1の自然言語対話だけをインコンテキストの学習例として与えて類似問題を解かせる。

Phase 2へは`final_student_state`、理解度、確信度、誤概念、感情、獲得知識などの構造化状態を渡さない。このため、test_v2のプロファイル更新型転移ではなく、対話からのインコンテキスト学習による近接転移を測定する。

## v4コーパスとの共通条件

| 項目 | 設定 |
| --- | --- |
| 生徒モデル | `tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2` |
| revision | `496cd5558fef4af1d426e96327d7a74681063280` |
| vLLM | 0.25.1、`--reasoning-parser qwen3` |
| 生徒sampling | temperature 0.6、top-p 0.95、top-k 20、min-p 0 |
| 生徒最大生成長 | 4,096 token |
| プロフィール | V2-S01〜V2-S04の4種類 |
| 初期感情 | neutral、engaged、curious、confused、frustrated、anxious |
| 条件割当 | 4プロフィール×6初期感情を24件単位でseed付き層化 |
| Phase 1 | 最大10ターン |

初期感情はプロフィールと独立に与える。初回発話では感情を変更できず、その後もv4の感情サイクルに従う。理解度は1ターン最大1段階、確信度は最大0.25だけ変化でき、獲得知識の削除は禁止する。

## Phase 2へ渡す情報

渡す情報：

- 元問題
- Phase 1の`student`と`teacher`の自然言語発話
- 新しい類似問題
- 変更されない元の生徒プロフィール

渡さない情報：

- 教師の`<analysis>`
- 各ターンの`state_after`と`state_update_reason`
- Phase 1終了時の構造化学習状態
- 模範解答

## ABCIでの実行

リポジトリ全体をコピーするか、少なくとも`pipelines/model_evaluation/test_v4/`と`pipelines/model_evaluation/shared/`を同じ階層でコピーする。

```bash
pip install -r pipelines/model_evaluation/test_v4/requirements-abci.txt
```

### 1. 生徒モデルの起動

Qwen3-Swallowはthinkingの無効化を行わず、reasoning parserを使用する。

```bash
python -m vllm.entrypoints.openai.api_server \
  --model tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2 \
  --revision 496cd5558fef4af1d426e96327d7a74681063280 \
  --served-model-name tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2 \
  --reasoning-parser qwen3 \
  --dtype bfloat16 \
  --port 8001
```

### 2. 教師モデルの起動

教師は生徒と異なるポートで起動する。BaseとSFT条件では、教師checkpoint以外を変更しない。

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/teacher-model-or-merged-checkpoint \
  --served-model-name teacher-under-test \
  --dtype bfloat16 \
  --port 8000
```

### 3. 2問の接続確認

```bash
python pipelines/model_evaluation/test_v4/generate_in_context_dialogues.py \
  --teacher-model teacher-under-test \
  --teacher-checkpoint /path/to/base-checkpoint \
  --teacher-adapter /path/to/adapter \
  --limit 2 \
  --output experiments/test_v4/pilot/dialogues.jsonl \
  --overwrite
```

生成ログについて、次を確認する。

- `transfer_mode`が`v4_in_context`
- `initial_emotion`と初回生徒発話が整合する
- Phase 1の生徒発話が原則1〜2文である
- Phase 2入力に構造化状態や教師analysisが含まれない
- Phase 2出力が`\boxed{...}`または`\boxed{わからない}`である

### 4. 本評価の対話生成

```bash
python pipelines/model_evaluation/test_v4/generate_in_context_dialogues.py \
  --teacher-model teacher-under-test \
  --teacher-checkpoint /path/to/base-checkpoint \
  --teacher-adapter /path/to/adapter \
  --seed 42 \
  --output experiments/test_v4/v4_sft/dialogues.jsonl
```

同じコマンドを再実行すると、正常終了済みの`run_id`をスキップする。モデル、prompt、問題、seed、samplingなどが既存manifestと一致しない場合は停止する。条件を変更する場合は別の出力先を使う。`--overwrite`は既存結果を意図的に破棄して再生成する場合だけ指定する。

### 5. Judge評価

```bash
export GPT_API_KEY="..."

python pipelines/model_evaluation/test_v4/evaluate_in_context_dialogues.py \
  --input experiments/test_v4/v4_sft/dialogues.jsonl \
  --output experiments/test_v4/v4_sft/evaluated_results.jsonl
```

教師対話はv4コーパスと同じ6項目を各10点、計60点で評価する。

1. 数学的正確性
2. 誤りの診断と回復
3. 認知的共感
4. 感情認識・情緒的支援
5. 適応的足場かけ
6. 理解確認・完了判定

併せて誤答追認、不要な直接解答、Critical failure、明示的判断記録の最終発話への反映、生徒らしさ、Phase 2近接転移正答を記録する。

## 比較時に固定する項目

Base、v3 SFT、v4 SFTなどの教師条件間で次を固定する。

- 問題・類似問題と共通除外18問
- 生徒モデルとrevision
- プロフィール・初期感情割当
- 全sampling設定とseed
- 最大ターン数
- 教師・生徒・Phase 2・Judgeのprompt
- Judgeモデルとreasoning effort

比較条件ごとに出力先を分け、同じseedを指定する。

## ローカルテスト

```bash
python -m unittest pipelines/model_evaluation/test_v4/test_v4.py
```
