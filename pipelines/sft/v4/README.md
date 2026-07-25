# v4 SFT実行フォルダー

v4の最終SFT構造ゲートを通過した104対話・711教師ターンを、`tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5`へBF16 LoRAで学習するABCI用自己完結フォルダーである。フォルダー全体をABCIへコピーすれば、データを別途配置せず実行できる。

## 構成

```text
v4/
├── config.json                       # データhash、モデルrevision、学習条件
├── train_v4_sft.py                   # 構造・token長・mask監査、学習、再開
├── setup_abci.sh                     # ABCI module・venv・依存関係の準備
├── run_abci.pbs                      # ABCI 3.0 H200単一GPUジョブ
├── requirements.txt                  # Python依存関係
├── HYPERPARAMETER_DESIGN.md          # 設定根拠
├── data/
│   ├── v4_sft.jsonl                  # 104対話の学習入力
│   ├── source_corpus_manifest.json   # コーパスの選択・評価・hash
│   ├── SOURCE_CORPUS_REPORT.md       # コーパス評価
│   ├── SOURCE_CORPUS_REQUIREMENTS.md # v4採択要件のスナップショット
│   └── README.md
└── tests/
    └── test_train_v4_sft.py
```

## 学習条件

| 項目 | 設定 |
|---|---|
| 基盤モデル | `tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5` |
| revision | `b1f8317099a97e790ec872c1225ca155979b4816` |
| データ | 104対話、教師711ターン |
| 分割 | seed 42、学習94件・検証10件 |
| 学習方式 | BF16 LoRA、単一H200 |
| LoRA | rank 16、alpha 32、dropout 0.05 |
| 対象層 | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| epoch | 4 |
| 学習率 | `5e-5`、cosine、warmup 0.1 |
| 実効バッチ | 4 |
| 最大系列長 | 8,192 tokens、暗黙の切り詰めなし |
| loss | assistantのみ。`<analysis>`重み0.25、`<final>`重み1.0 |

94学習対話、実効バッチ4では1 epochあたり24 optimizer step、4 epochで約96 stepとなる。小規模データで数学能力を過度に変化させないため、量子化せず、attention projectionだけへ低学習率でLoRAを適用する。構造化CoTは数学的検証・認知状態・感情・支援方針を学習するため残すが、生徒に提示する教師発話を主目的とするため`<final>`へ4倍の相対token重みを与える。実データではCoT 211,208 tokens、最終発話68,462 tokensであり、この設定で重み付き教師信号は概算43.5%対56.5%となる。

## 1. ABCIへコピー

リポジトリ全体をコピーする場合は`pipelines/sft/v4`へ移動する。フォルダー単体をコピーする場合も、`data/v4_sft.jsonl`を含めてコピーする。

```bash
cd /path/to/pipelines/sft/v4
```

## 2. 環境作成

```bash
chmod +x setup_abci.sh
./setup_abci.sh
```

このスクリプトはABCIの`python/3.12/3.12.9`と`cuda/13.0/13.0.1`をロードし、`.venv`を作成して依存関係を導入した後、モデルをダウンロードせずに入力hash、104件、role順、制御文字、analysis/final形式、94/10分割を検査する。

PyTorchのwheelをABCI推奨手順で別途導入する必要がある環境では、先にPyTorchを導入し、残りの依存関係のバージョンを維持する。

## 3. 事前監査

### モデル不要の構造監査

```bash
source .venv/bin/activate
python train_v4_sft.py --config config.json --structure-only
python -m unittest discover -s tests -p 'test_*.py'
```

### tokenizerを使う完全監査

```bash
python train_v4_sft.py --config config.json --preflight-only
```

完全監査は固定revisionのtokenizerとchat templateを適用し、全104件のtoken長、assistant-only mask、`<analysis>`と`<final>`のtoken境界を確認する。8,192 tokensを超える対話は自動的に切り詰めず停止する。その場合は`max_length`だけを安易に変更せず、GPUメモリを確認するか、教師ターン境界で明示的に分割した別データ・別`output_dir`を作る。

## 4. PBS投入

ABCIグループはPBSへ埋め込まない。

```bash
export ABCI_GROUP=YOUR_ABCI_GROUP
qsub -P "${ABCI_GROUP}" run_abci.pbs
qstat
```

PBSは投入時にも構造監査を行い、その後`--resume auto`で学習する。walltimeは12時間、queueは`rt_HG`、GPUは単一H200を想定する。

## 5. 再開

同じデータ、config、split、モデルrevisionでPBSを再投入すると、最後の`checkpoint-*`から再開する。データhashまたは設定が変わるとrun fingerprintが一致せず停止する。

最初から別条件で実行する場合は、既存出力を削除せず`config.json`の`output_dir`を新しい名前へ変更する。`--resume none`で既存outputへ上書きしてはいけない。

## 出力

```text
outputs/swallow8b_v4_lora_104/
├── checkpoint-*/
├── final_adapter/
├── preflight_report.json
├── split_manifest.json
├── run_manifest.json
└── trainer_state.json
```

各epochで検証し、検証lossが最小のcheckpointを読み戻して`final_adapter/`へ保存する。`run_manifest.json`には入力SHA-256、モデルrevision、環境、GPU、CUDA、ライブラリ、系列長統計、split、最良checkpoint、最終metricsを保存する。

## 実験上の注意

- test-v4をcheckpoint選択へ使用せず、学習完了後の最終評価だけに使う。
- Base、v3 SFT、v4 SFTで同一の基盤モデルrevisionと評価条件を使う。
- v4の効果とデータ量を分ける対照実験では、比較コーパスから同じ104対話を固定抽出する。
- コーパス生成・Repair・監査は同じモデル系列への適合を含みうるため、SFT結果はtest-v4の数学的指導・共感指導・誤答追認・直接解答・転移成績で判断する。
- 推論時はモデル出力を`<analysis>...</analysis><final>...</final>`として検証し、生徒UIや評価対象には`<final>`本文だけを渡す。解析失敗時に全文を生徒へフォールバック表示しない。
- 詳細な設定根拠は[HYPERPARAMETER_DESIGN.md](./HYPERPARAMETER_DESIGN.md)を参照する。
