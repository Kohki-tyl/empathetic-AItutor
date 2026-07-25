# SFT

SFT関連ファイルは、評価テストのバージョンとは独立して管理する。

```text
sft/
├── shared/
│   └── prompts/    # v0・v1で使用した共通教師プロンプト
├── v0/
│   └── data/       # v0の学習・検証データ
├── v1/
│   ├── data/       # v1学習データと生成元コーパス
│   ├── prompts/    # v1固有プロンプト
│   └── *.py        # v1学習データ作成
├── v2/
│   ├── data/       # Keep-only・CoT付き学習データ
│   ├── prompts/    # v2 CoT教師プロンプト
│   └── *.py        # v2学習データ作成
├── v3/
│   ├── data/       # ターン監査・修正済みの単一SFTデータ
│   ├── prompts/    # 数学的検証を含むv3 CoT教師プロンプト
│   └── *.py        # v3学習データ作成
└── v4/
    ├── data/       # 厳格監査後のv4_sft.jsonl配置先
    ├── config.json # 再検討したBF16 LoRA設定
    ├── *.md        # 実行手順とハイパーパラメータ根拠
    ├── *.pbs       # ABCI 3.0実行ジョブ
    └── *.py        # 監査・分割・学習スクリプト
```

`shared/prompts/sft_teacher_system.txt`は、v0およびv1で使用した教師プロンプトである。v2 Keep-only（CoTなし）のデータ作成では、v1との比較条件を維持するため同じプロンプトを再利用する。v2 Keep-only＋CoTでは、`v2/prompts/v2_cot_teacher_system.txt`を使用する。

モデル名や実験結果では、`sft_v1`、`sft_v2`のようにSFT版を明記する。評価結果は`experiments/test_vN/`へ保存し、どのテスト版で評価したかを別に記録する。
