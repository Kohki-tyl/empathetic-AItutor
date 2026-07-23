# Model Evaluation

モデル評価関連のファイルは、実験バージョンと共通資産で分けている。

```text
model_evaluation/
├── v1/
│   ├── data/       # v1 SFTデータと生成元コーパス
│   ├── prompts/    # v1生徒シミュレーター用プロンプト
│   ├── archive/    # 旧40問テスト
│   └── *.py        # v1データ作成・評価・分析
├── v2/
│   ├── data/       # Keep-onlyおよびCoT付きSFTデータ
│   ├── prompts/    # v2教師・生徒・生徒評価プロンプト
│   └── *.py        # v2データ作成・対話生成・Judge評価
└── shared/
    ├── questions/  # v1/v2で固定して使用する200問と類似問題
    └── prompts/    # 共通の教師・Judgeプロンプト
```

v1とv2を比較するときは、`shared/questions/`の問題セットを変更しない。v2評価では教師モデルだけを変更し、生徒モデル、Judge、プロンプト、seedを固定する。

固定生徒モデルには、未SFTの`tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5`を使用する。既定の接続先は`http://localhost:8001/v1`である。

v2は`generate_v2_dialogues.py`で対話と類似問題解答を生成し、`evaluate_v2_dialogues.py`で保存済みログを評価する。生成ログと評価結果は別ファイルへ保存する。
