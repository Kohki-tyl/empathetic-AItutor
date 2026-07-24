# Experiments

実験結果はテストバージョン単位で保存する。

```text
experiments/
├── test_v0/
├── test_v1/
├── test_v2/
└── test_v3/
```

- `test_vN`: 質問セット、生徒シミュレーター、評価指標などのテスト版
- `sft_vM_*`: 評価対象となる教師モデルのSFT版
- `baseline_*`: SFTしていない教師モデル

例として、`test_v2/sft_v1_swallow_8b_v0.5/`は「プロファイル更新を使うテストv2でSFT v1のSwallow 8Bを評価した結果」を表す。
