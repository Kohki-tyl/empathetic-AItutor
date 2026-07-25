# v4 SFT実行フォルダー

v4厳格監査コーパスを`tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5`へBF16 LoRAで学習する。フォルダーをABCIへコピーし、コーパス生成後の`v4_sft.jsonl`を`data/`へ置けば実行できる。

## 構成

```text
v4/
├── config.json                 # 固定した学習条件
├── HYPERPARAMETER_DESIGN.md    # v2設定から変更した理由
├── train_v4_sft.py             # 監査・分割・学習・再開・manifest保存
├── run_abci.pbs                # ABCI 3.0 rt_HG用ジョブ
├── requirements.txt            # 検証済み対象バージョン
├── data/
│   └── README.md               # v4_sft.jsonlの配置方法
└── tests/
    └── test_train_v4_sft.py
```

## 1. ABCIへコピー

```bash
cd /path/to/pipelines/sft/v4
cp /path/to/corpus_creation/v4/data/run_100/v4_sft.jsonl data/v4_sft.jsonl
```

`config.json`は100件を要求する。100件未満で学習する場合は、採択不足を確認した上で`expected_records`を意図的に変更する。

## 2. 環境作成

```bash
source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

依存関係は2026年7月25日時点の安定版へ固定している。ABCI側のCUDAドライバとの組合せでPyTorch導入に失敗する場合は、PyTorchだけABCI推奨手順で導入し、残りのバージョンは維持する。

## 3. 事前監査

```bash
python train_v4_sft.py --config config.json --preflight-only
```

事前監査では、レコード数、role順、`<analysis>/<final>`、モデル固有chat template適用後の系列長、assistant-only maskを確認する。8,192トークンを超えるレコードは切り詰めず、終了する。

## 4. PBS設定と投入

`run_abci.pbs`の`CHANGE_TO_YOUR_ABCI_GROUP`を実際のABCIグループ名へ変更する。

```bash
qsub run_abci.pbs
qstat
```

既定設定は、H200 1基、BF16 LoRA、最大8,192トークン、学習90件、検証10件、4 epochである。各epochを保存し、検証損失が最小のcheckpointを`outputs/swallow8b_v4_lora/final_adapter/`へ保存する。

## 5. 再開

PBSは`--resume auto`を指定している。中断後に同じ入力、設定、モデルrevisionで再投入すると、最後のcheckpointから再開する。入力または設定のSHA-256が既存manifestと異なる場合は停止する。

最初からやり直す場合は、既存出力を削除せず別の`output_dir`へ変更する。

## 出力

```text
outputs/swallow8b_v4_lora/
├── checkpoint-*/
├── final_adapter/
├── preflight_report.json
├── run_manifest.json
├── split_manifest.json
└── trainer_state.json
```

`run_manifest.json`には、入力SHA-256、モデルrevision、環境、ハイパーパラメータ、系列長統計、最良checkpoint、最終metricsを記録する。

## 設計上の注意

- test-v4はcheckpoint選択に使わず、学習完了後の最終評価だけに使う。
- Base、v3 SFT、v4 SFTは同一の基盤モデルrevisionと評価条件で比較する。
- v4の効果とデータ量を分ける追加実験では、v3から同数の100件を抽出した対照条件を用意する。
- 詳細な判断理由は[HYPERPARAMETER_DESIGN.md](./HYPERPARAMETER_DESIGN.md)を参照する。

