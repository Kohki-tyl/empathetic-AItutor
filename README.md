# Empathetic AI Tutor

学習者の認知状態と感情に応じて、数学の理解を段階的に支援する共感的AIチューターの研究リポジトリです。日本語の数学対話コーパス作成、教師ありファインチューニング（SFT）用データ整形、対話シミュレーションによるモデル評価を扱います。

## 概要

このプロジェクトでは、LLMを単に答えを返すSolverではなく、学習者のつまずきを推定し、次に考えるべき一歩だけを提示するTutorとして評価します。主な要素は次の3つです。

- MATHベンチマークのテキスト問題を日本語化し、図への依存がある問題を除外
- 生徒・教師のロールプレイによる、感情ラベル付きマルチターン対話の合成
- 正答到達率、類似問題への転移、LLM-as-a-judgeによる共感性の評価

教師応答では、生徒の状態を `Engaged`、`Curious`、`Neutral`、`Confusion`、`Frustrated`、`Bored`、`Anxious`、`Eureka`、`Proud`、`Relieved` の10クラスで追跡します。

## リポジトリ構成

```text
.
├── pipelines/
│   ├── corpus_creation/       # 翻訳、フィルタリング、対話コーパス生成
│   │   ├── prompts/
│   │   └── questions/
│   └── model_evaluation/      # SFTデータ整形、評価問題生成、シミュレーション評価
│       ├── v1/                # v1 SFT・旧評価
│       ├── v2/                # Keep-only/CoT SFT・分離型評価
│       └── shared/            # 共通質問セット・Judgeプロンプト
├── experiments/               # テスト版別の質問、評価結果、分析コード
│   ├── v0_test/               # 40問で実施したv0テスト
│   │   ├── questions/
│   │   ├── baseline_swallow_70b_v0.3/
│   │   ├── baseline_swallow_8b_v0.5/
│   │   └── sft_swallow_8b_v0.5/
│   └── v1_test/               # 200問で実施したv1テスト
│       ├── questions/
│       └── sft_swallow_8b_v0.5/
├── docs/                      # SFT方針と関連資料
└── TODO.md                    # 今後の作業
```

各パイプラインのプロンプト、入力データ、生成物は、そのパイプラインのディレクトリ内にまとめています。`experiments/`には再現用コードだけでなく、実行済みJSONLとレポートも保存しています。

## セットアップ

Python 3.12以降を想定しています。

```bash
python -m venv .venv
```

仮想環境を有効化した後、必要なパッケージをインストールします。

```bash
python -m pip install openai datasets tqdm
```

OpenAI APIを使用する処理では `GPT_API_KEY` を設定してください。

PowerShell:

```powershell
$env:GPT_API_KEY = "your-api-key"
```

Bash:

```bash
export GPT_API_KEY="your-api-key"
```

APIキーをソースコードやコミット対象のファイルへ保存しないでください。

## コーパス作成

コマンドはリポジトリのルートから実行します。

1. MATHデータセットを日本語へ翻訳します。

   ```bash
   python pipelines/corpus_creation/translate_train_dataset.py
   ```

   出力: `pipelines/corpus_creation/questions/translated_1000_math.jsonl`

2. 生徒シミュレータと教師モデルの対話を生成します。

   ```bash
   python pipelines/corpus_creation/generate_corpus.py
   ```

   出力: `pipelines/corpus_creation/500_empathetic_dialogues.jsonl`

各スクリプトの件数、モデル名、温度、最大ターン数は、現時点ではファイル内の定数として管理されています。API呼び出しを伴うため、実行前に設定と想定コストを確認してください。

## SFTデータの作成

生成済み対話を `pipelines/model_evaluation/v1/data/math_tutor_corpus.jsonl` に配置し、次を実行します。

```bash
python pipelines/model_evaluation/v1/prepare_sft_dataset.py
```

`is_completed: true` の対話だけを抽出し、固定シードでシャッフルして全件を訓練用データへ変換します。

- `pipelines/model_evaluation/v1/data/sft_train.jsonl`

### v2 Keep-only SFTデータの作成

500件のコーパス品質評価で `recommendation: keep` と判定され、かつ指導完了している対話だけをv2学習データへ変換します。

```bash
python pipelines/model_evaluation/v2/prepare_keep_only_sft_dataset.py
```

出力:

- `pipelines/model_evaluation/v2/data/v2_keep_only_sft_train.jsonl`
- `pipelines/model_evaluation/v2/data/v2_keep_only_sft_manifest.json`

Manifestにはフィルタ条件、シャッフルシード、採用件数、出力順の`source_id`を保存します。

CoTを含めて学習する場合は、教師ターンを `<analysis>` と `<final>` に分離したデータを作成します。

```bash
python pipelines/model_evaluation/v2/prepare_v2_cot_sft_dataset.py
```

出力:

- `pipelines/model_evaluation/v2/data/v2_keep_only_cot_sft_train.jsonl`
- `pipelines/model_evaluation/v2/data/v2_keep_only_cot_sft_manifest.json`

`<analysis>`には認知状態、感情状態、次の一歩を収録し、`<final>`には生徒へ提示する短い発話だけを収録します。CoTなし版は比較実験用に残します。

現在のSFT方針は [docs/SFT_Strategy.md](docs/SFT_Strategy.md) を参照してください。

## 評価

評価は、元問題での対話学習と、会話履歴を消去した後の類似問題テストで構成されます。

### 学習コーパスの品質評価

500件の生成対話を、感情認識、教育的共感、数学的正確性、エラー回復、適応的足場かけ、発話長の6軸で評価します。まず少数件で出力を確認できます。

```bash
python pipelines/corpus_creation/evaluate_corpus.py --limit 5
```

全件を評価する場合は次を実行します。

```bash
python pipelines/corpus_creation/evaluate_corpus.py
```

結果は `pipelines/corpus_creation/500_dialogue_evaluations.jsonl` に逐次保存されます。再実行時は評価済みの `source_id` をスキップするため、中断した位置から再開できます。最初から評価し直す場合のみ `--overwrite` を指定してください。

### 評価問題の準備

```bash
python pipelines/model_evaluation/shared/translate_test_dataset.py
python pipelines/model_evaluation/shared/questions/generate_similar_questions.py
```

生成物は `pipelines/model_evaluation/shared/questions/` に保存されます。

### ベースライン評価

```bash
python pipelines/model_evaluation/v1/evaluate_baseline_by_simulator.py
python pipelines/model_evaluation/v1/analyze_model.py
```

評価スクリプトは、既定ではOpenAI互換APIを `http://localhost:8000/v1` で提供しているローカルモデルを利用し、JudgeにはOpenAI APIを利用します。モデル名、エンドポイント、最大ターン数は実行前にスクリプト先頭の設定を確認してください。

### v2 分離型評価

v2では評価対象の教師と固定した生徒シミュレーターを、別モデル・別エンドポイントで実行します。Phase 2には対話全文ではなく、Phase 1終了時の生徒の学習状態だけを引き継ぎます。対話生成とJudge評価は別スクリプトで実行します。

接続確認はJudgeを呼ばずに実行できます。

```bash
python pipelines/model_evaluation/v2/generate_v2_dialogues.py \
  --teacher-model teacher-under-test \
  --teacher-system-prompt pipelines/model_evaluation/v2/prompts/v2_cot_teacher_system.txt \
  --limit 2 \
  --output experiments/v2_test/pilot/dialogues.jsonl \
  --overwrite
```

生徒には既定で、SFTしていない`tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5`を使用します。全教師条件で同じcheckpoint、プロンプト、temperature、seedを固定します。

生成後にJudge評価を実行します。

```bash
python pipelines/model_evaluation/v2/evaluate_v2_dialogues.py \
  --input experiments/v2_test/pilot/dialogues.jsonl \
  --output experiments/v2_test/pilot/evaluated_results.jsonl \
  --overwrite
```

ABCIでのサーバー構成、20問パイロット、主実験の固定条件は [v2 SFT・分離型評価設計](docs/v2_sft_evaluation_design.md) を参照してください。

主な評価指標は次のとおりです。

- 指導完了率: 元問題の対話で学習者が正解へ到達した割合
- Near Transfer Accuracy: 履歴を消した類似問題に正解した割合
- Emotion Alignment: 生徒の感情状態を適切に捉えたか
- Pedagogical Empathy: 安心感と適切な足場かけを両立したか
- Length Control: 生徒への発話が簡潔か

## 実験結果

評価ログ、分析スクリプト、レポートは `experiments/` 以下にテスト版ごとに保存しています。各テストフォルダーの `questions/` は、そのテストで使用した元問題と類似問題のスナップショットです。

```text
experiments/
├── v0_test/
│   ├── questions/                    # v0の元問題・類似問題（40問）
│   └── <条件>_<モデル規模>_<モデル版>/
└── v1_test/
    ├── questions/                    # v1の元問題・類似問題（200問）
    └── <条件>_<モデル規模>_<モデル版>/
```

例: `experiments/v0_test/baseline_swallow_8b_v0.5`、`experiments/v1_test/sft_swallow_8b_v0.5`

比較時は、データ版、モデル規模、ベースライン/SFTの条件が一致しているかを確認してください。

## データ形式

対話コーパスは1行1セッションのJSONLです。主なフィールドは次のとおりです。

| フィールド | 内容 |
| --- | --- |
| `source_id` | 元問題の識別子 |
| `problem` | 日本語の数学問題 |
| `student_profile` | 学年、既習範囲、苦手分野など |
| `is_completed` | 指導完了の判定 |
| `conversation` | 生徒と教師の発話列 |

教師ターンには、生成時のスキーマに応じて `student_emotion`、`thought_process`、`roadmap_breakdown`、`next_step_plan` などが含まれます。SFT向けデータではChatML互換の `messages` 配列へ変換されます。

## 注意事項

- JSONLにはモデル出力や内部推論相当のフィールドが含まれるため、公開・再利用時は用途とライセンスを確認してください。
- 生成系スクリプトは出力ファイルを初期化するものがあります。既存結果を残す場合は事前にコピーしてください。
- モデル名やAPI仕様は固定値です。利用環境に合わせて各スクリプトの設定を確認してください。

## 関連資料

- [研究の現状・実施内容・ToDo](docs/research_status_and_todo.md)
- [SFT戦略](docs/SFT_Strategy.md)
- [500件コーパス品質評価レポート](docs/corpus_quality_report.md)
- [v2 SFT・分離型評価設計](docs/v2_sft_evaluation_design.md)
- [今後の作業](TODO.md)
