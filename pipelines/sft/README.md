# SFT

SFT関連ファイルは、評価テストのバージョンとは独立して管理する。

```text
sft/
├── shared/
│   └── prompts/    # 複数SFT版で共有する教師プロンプト
├── v1/
│   ├── data/       # v1学習データと生成元コーパス
│   ├── prompts/    # v1固有プロンプト
│   └── *.py        # v1学習データ作成
└── v2/
    ├── data/       # Keep-only・CoT付き学習データ
    ├── prompts/    # v2 CoT教師プロンプト
    └── *.py        # v2学習データ作成
```

モデル名や実験結果では、`sft_v1`、`sft_v2`のようにSFT版を明記する。評価結果は`experiments/test_vN/`へ保存し、どのテスト版で評価したかを別に記録する。
