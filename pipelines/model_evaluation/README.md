# Model Evaluation

モデル評価関連のファイルは、テストバージョンと共通資産で分けている。SFTバージョンは`pipelines/sft/`で独立して管理する。

```text
model_evaluation/
├── test_v1/
│   ├── prompts/    # v1生徒シミュレーター用プロンプト
│   ├── archive/    # 旧40問テスト
│   └── *.py        # テストv1の評価・分析
├── test_v2/
│   ├── prompts/    # v2教師・生徒・生徒評価プロンプト
│   └── *.py        # プロファイル更新テストの生成・評価
├── test_v3/
│   ├── prompts/    # v3インコンテキスト学習用プロンプト
│   └── *.py        # インコンテキスト学習テストの生成・評価
├── test_v4/
│   ├── prompts/    # v4コーパス準拠の生徒・教師・Judgeプロンプト
│   └── *.py        # Qwen3-Swallow生徒によるインコンテキスト転移評価
└── shared/
    ├── questions/  # テストv1/v2で固定して使用する200問と類似問題
    └── prompts/    # 共通の教師・Judgeプロンプト
```

テストv1とテストv2を比較するときは、`shared/questions/`の問題セットを変更しない。v2評価では教師モデルだけを変更し、生徒モデル、Judge、プロンプト、seedを固定する。

テストv2/v3の固定生徒モデルには、未SFTの`tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5`を使用する。テストv4では、v4コーパス生成条件に合わせて`tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2`の固定revisionを使用する。いずれも既定の接続先は`http://localhost:8001/v1`である。

テストv2は`generate_profile_update_dialogues.py`、テストv3は`generate_in_context_dialogues.py`で生成する。テストv4はv4コーパスと同じQwen3-Swallow生徒、4プロフィール、6初期感情、状態遷移制約を使い、Phase 1の自然言語対話だけをPhase 2へ渡す。各テスト固有の評価スクリプトで保存済みログを評価し、結果をテストバージョン別に保存する。

テストv2/v3の教師評価は、共感指導タスク（30点）と数学的指導タスク（50点）に分ける。旧`eval_empathy_judge_system.txt`はテストv1の再現用として残し、新評価では`eval_empathic_instruction_judge_system.txt`と`eval_mathematical_instruction_judge_system.txt`を使用する。

テストv4の教師評価は、v4コーパス採択と同じ6項目を各10点、計60点で採点し、誤答追認、不要な直接解答、Critical failure、明示的判断記録と最終発話の対応も評価する。
