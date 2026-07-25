# Test v4：v4生徒準拠インコンテキスト転移評価

## 目的

v4コーパス作成時と同じ生徒モデル、プロフィール、初期感情、状態更新制約を用いて教師モデルを評価する。Phase 1では教師との対話を生成し、Phase 2ではPhase 1の自然言語対話だけをインコンテキストの学習例として与えて類似問題を解かせる。

Phase 2へは`final_student_state`、理解度、確信度、誤概念、感情、獲得知識などの構造化状態を渡さない。このため、test_v2のプロファイル更新型転移ではなく、対話からのインコンテキスト学習による近接転移を測定する。Phase 2では最終解答に加え、利用したプロフィール既習知識またはPhase 1教師発話の完全一致引用を内部記録し、許可されない知識源による「賢すぎる正答」を検証エラーまたは生徒らしさ違反として扱う。採点へ渡す最終解答は従来どおり`phase2_student_answer`だけである。

## v4コーパスとの共通条件

| 項目 | 設定 |
| --- | --- |
| 生徒モデル | `tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2` |
| revision | `496cd5558fef4af1d426e96327d7a74681063280` |
| vLLM | 0.25.1、`--reasoning-parser qwen3` |
| 生徒sampling | temperature 0.6、top-p 0.95、top-k 20、min-p 0 |
| 生徒最大生成長 | 4,096 token |
| プロフィール | E2/E3仕様を持つV4-S01〜V4-S08（中学1年〜高校2年、分野別習熟差を含む） |
| 初期感情 | neutral、engaged、curious、confused、frustrated、anxious |
| 条件割当 | 問題ごとの事前対応表でプロフィール・範囲関係・初期感情・誤概念を固定 |
| 問題選択 | 後半200件から固定した120問を、各範囲15件の一次60問・確認60問へ排他的に分割 |
| Phase 1 | 最大10ターン |

初期感情は、問題の必要範囲とプロフィールの分野別習得段階の関係、およびMATH難度から事前決定する。`far_beyond`には教師との対話前に行った異なる2回の試行と共通の停止箇所を`prior_attempt_history.attempts`へ保存する。初回発話は対応表の`required_initial_disclosure`から開始して両試行を明示し、その後に問題固有の援助要請を一つだけ行う。教師介入前には理解度、獲得知識、未習範囲、誤概念を変更できず、確信度変化も0.1以内とする。その後はv4の感情サイクルに従う。

プロフィールの`prior_knowledge`を使用可能知識の完全な一覧とし、生徒は各ターンで`response_stage`と`knowledge_used`を返す。獲得知識は生成スクリプト側を正本とし、モデルは今回増えた`newly_acquired_knowledge`だけを返す。過去の全`acquired_knowledge`を再出力させないため、長い対話での習得済み知識の脱落を防ぐ。E2としてカリキュラム範囲を超えた知識使用を検査し、E3として事前指定した誤概念・部分手続きが回答へ一貫して現れるかをJudgeで確認する。範囲外の場合、初回は条件整理または具体的な援助要請に限定する。

評価では各問題のプロフィール、初期感情、E3誤概念を全教師条件で共通にする。初回の正誤そのものは`natural_profile_consistent`とするが、指定された認識状態と無関係な誤答や未習知識による正答は許可しない。

`prompts/test_120_selection.json`は、コーパス候補に使用しない後半200件から、共通除外18件を除いた後にseed 42で固定した親集合である。この120問を別の固定層化分割によって、`test_60_primary_selection.json`と`test_60_confirmation_selection.json`へ分ける。両方とも4種類の`scope_relation`が各15件で、重複せず、和集合が元の120問と一致する。

一次評価では全教師条件に`test_60_primary_selection.json`を使用する。一次結果を見て問題を選び直してはならない。効果が小さい、信頼区間が広い、数学的正確性の低下が疑われる、または低頻度failureを判断できない場合に、事前固定済みの確認用60問を追加する。条件別15件の結果は探索的分析とし、scope別の優劣を確証的に主張しない。

選択表を再生成する必要がある場合だけ次を実行する。通常の評価前には再生成しない。

```bash
python pipelines/model_evaluation/test_v4/build_staged_selections.py
```

対応表は概念単位の必要範囲をプロフィールの自然言語`prior_knowledge`と照合する。選択時と実行時に、`mastered`への未習概念混入、範囲関係不整合、要人手確認問題を拒否する。教師へは全条件で初回だけ事前割当済みの初期感情ラベルを渡し、現在感情は渡さない。

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

ABCIでは次の配置を前提とする。

```text
SFT_abci/test/
├── shared/
└── test_v4/
    ├── abci/run_generation.pbs
    ├── generate_in_context_dialogues.py
    ├── evaluate_in_context_dialogues.py
    └── prompts/
```

以下のコマンドは`SFT_abci/test/test_v4`をカレントディレクトリとして実行する。リポジトリ内の正本は`pipelines/model_evaluation/test_v4`だが、ABCIへコピーした後にそのパスを参照しない。

```bash
cd /path/to/empathetic-AItutor/SFT_abci/test/test_v4
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-abci.txt
python -m unittest test_v4.py
```

prompt同期テストは、リポジトリ全体がある場合はコーパス正本と直接比較する。テストフォルダーだけをコピーした環境では`prompts/corpus_prompt_sync.json`の固定hashと比較し、skipしない。正本が別の場所にある場合は`V4_CORPUS_PROMPT_DIR`を設定する。

### 推奨：PBSで教師・生徒・生成を一括実行

`abci/run_generation.pbs`は`rt_HF`のノード占有8GPUを要求し、教師をGPU 0、生徒をGPU 1で起動する。`rt_HG`はGPU 1基のためこの構成には使用しない。2つのvLLMを別GPU・動的ポートで起動し、`/v1/models`の準備完了後に生成する。LoRA条件ではvLLMへadapterを実際にロードし、生成スクリプトもLoRA model cardの`parent`と`root`を検証する。

```bash
cd /path/to/empathetic-AItutor/SFT_abci/test/test_v4
export TEST_CONDITION=lora
export TEACHER_MODEL_PATH=tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5
export TEACHER_REVISION=b1f8317099a97e790ec872c1225ca155979b4816
export TEACHER_LORA_PATH=/absolute/path/to/final_adapter
export TEACHER_SERVED_MODEL=v4-sft
export OUTPUT_FILE="$PWD/data/v4_sft/primary_60/dialogues.jsonl"
qsub -P YOUR_ABCI_GROUP abci/run_generation.pbs
```

Base条件ではadapterを指定しない。

```bash
export TEST_CONDITION=base
export TEACHER_MODEL_PATH=tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5
export TEACHER_REVISION=b1f8317099a97e790ec872c1225ca155979b4816
export TEACHER_SERVED_MODEL=teacher-base
export OUTPUT_FILE="$PWD/data/base/primary_60/dialogues.jsonl"
unset TEACHER_LORA_PATH
qsub -P YOUR_ABCI_GROUP abci/run_generation.pbs
```

確認用60問では、一次評価と同じモデル条件のまま次だけ変更する。

```bash
export SELECTION_FILE="$PWD/prompts/test_60_confirmation_selection.json"
export OUTPUT_FILE="$PWD/data/v4_sft/confirmation_60/dialogues.jsonl"
qsub -P YOUR_ABCI_GROUP abci/run_generation.pbs
```

### 手動起動する場合

#### 1. 生徒モデル

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

#### 2. LoRA教師モデル

`--teacher-adapter`だけではadapterは適用されない。vLLMを次のように起動し、`v4-sft`を実際のリクエストmodelとして使用する。

```bash
python -m vllm.entrypoints.openai.api_server \
  --model tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5 \
  --revision b1f8317099a97e790ec872c1225ca155979b4816 \
  --served-model-name teacher-base \
  --enable-lora \
  --lora-modules '{"name":"v4-sft","path":"/absolute/path/to/final_adapter","base_model_name":"tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"}' \
  --max-lora-rank 16 \
  --dtype bfloat16 \
  --port 8000
```

#### 3. 2問の接続確認

```bash
python generate_in_context_dialogues.py \
  --teacher-model v4-sft \
  --teacher-checkpoint tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5 \
  --teacher-revision b1f8317099a97e790ec872c1225ca155979b4816 \
  --teacher-adapter /absolute/path/to/final_adapter \
  --teacher-serving-mode lora \
  --limit 2 \
  --output data/pilot/dialogues.jsonl \
  --overwrite
```

`/v1/models`に`v4-sft`がない、model cardにLoRAの`parent`がない、または`root`が指定adapterと一致しない場合は、生成開始前に停止する。Baseやmerged条件で`--teacher-adapter`を渡した場合も停止する。

生成ログについて、次を確認する。

- `transfer_mode`が`v4_in_context`
- `initial_emotion`と初回生徒発話が整合する
- 教師の最初のuser入力に`初期感情ラベル`があり、2ターン目以降にはない
- Phase 1の生徒発話が原則1〜2文である
- Phase 2入力に構造化状態や教師analysisが含まれない
- Phase 2出力が`\boxed{...}`または`\boxed{わからない}`である

#### 4. 一次60問の対話生成

```bash
python generate_in_context_dialogues.py \
  --teacher-model v4-sft \
  --teacher-checkpoint tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5 \
  --teacher-revision b1f8317099a97e790ec872c1225ca155979b4816 \
  --teacher-adapter /absolute/path/to/final_adapter \
  --teacher-serving-mode lora \
  --seed 42 \
  --output data/v4_sft/primary_60/dialogues.jsonl
```

`--problem-selection`を省略すると一次60問が使われる。Base、v3 SFT、v4 SFTの全条件で同じ選択表、seed、samplingを使用する。

同じコマンドを再実行すると、正常終了済みの`run_id`をスキップする。モデル、prompt、問題、seed、samplingなどが既存manifestと一致しない場合は停止する。条件を変更する場合は別の出力先を使う。`--overwrite`は既存結果を意図的に破棄して再生成する場合だけ指定する。

### Judge評価

```bash
export GPT_API_KEY="..."

python evaluate_in_context_dialogues.py \
  --input data/v4_sft/primary_60/dialogues.jsonl \
  --output data/v4_sft/primary_60/evaluated_results.jsonl
```

Judgeは入力全件について`generation_error`が空、Phase 1対話が存在し、Phase 2回答と構造記録が存在することを先に検査する。1件でも未完了ならAPIを呼ばずに停止するため、生成を再実行して全件成功させてから評価する。

教師対話はv4コーパスと同じ6項目を各10点、計60点で評価する。

1. 数学的正確性
2. 誤りの診断と回復
3. 認知的共感
4. 感情認識・情緒的支援
5. 適応的足場かけ
6. 理解確認・完了判定

併せて誤答追認、不要な直接解答、Critical failure、明示的判断記録の最終発話への反映、生徒らしさ、Phase 2近接転移正答を記録する。

### 必要時のみ確認用60問を追加

```bash
python generate_in_context_dialogues.py \
  --teacher-model v4-sft \
  --teacher-checkpoint tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5 \
  --teacher-revision b1f8317099a97e790ec872c1225ca155979b4816 \
  --teacher-adapter /absolute/path/to/final_adapter \
  --teacher-serving-mode lora \
  --problem-selection prompts/test_60_confirmation_selection.json \
  --seed 42 \
  --output data/v4_sft/confirmation_60/dialogues.jsonl

python evaluate_in_context_dialogues.py \
  --input data/v4_sft/confirmation_60/dialogues.jsonl \
  --output data/v4_sft/confirmation_60/evaluated_results.jsonl
```

追加実行する場合、一次60問だけの結果を最終結果として置き換えず、一次・確認を合わせた120問を主な最終推定に使う。一次結果と確認結果も別々に残し、効果の再現性を確認する。

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
python -m unittest test_v4.py
```
