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
└── shared/
    ├── questions/  # テストv1/v2で固定して使用する200問と類似問題
    └── prompts/    # 共通の教師・Judgeプロンプト
```

テストv1とテストv2を比較するときは、`shared/questions/`の問題セットを変更しない。v2評価では教師モデルだけを変更し、生徒モデル、Judge、プロンプト、seedを固定する。

固定生徒モデルには、未SFTの`tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5`を使用する。既定の接続先は`http://localhost:8001/v1`である。

テストv2は`generate_profile_update_dialogues.py`、テストv3は`generate_in_context_dialogues.py`で生成する。各テスト固有の評価スクリプトで保存済みログを評価し、結果を`experiments/test_v2/`と`experiments/test_v3/`へ分けて保存する。

テストv2/v3の教師評価は、共感指導タスク（30点）と数学的指導タスク（50点）に分ける。旧`eval_empathy_judge_system.txt`はテストv1の再現用として残し、新評価では`eval_empathic_instruction_judge_system.txt`と`eval_mathematical_instruction_judge_system.txt`を使用する。
