# 旧500件 vs research採択350件 SFT比較

同じ旧500対話を母集団とし、全500件条件とresearch基準通過350件条件を、同じ基盤モデル・LoRA・seed・epochで比較する。データ品質フィルタの有無とデータ量が同時に変わる実用的な比較であり、品質だけの因果効果を分離する実験ではない。

## 条件

| 条件 | コーパス | 内部train/validation |
| --- | ---: | ---: |
| all500 | 500 | 450 / 50 |
| research350 | 350 | 315 / 35 |

350件の選択条件は、数学的正確性10点、適用可能な全軸8点以上、重大失敗なしである。未解決の修正指示と確定評価セット漏洩は未監査のため、manifestにも未適用ゲートとして記録する。

両条件とも旧コーパスに保存された `thought_process`、`student_emotion`、`next_step_plan` を `<analysis>`、可視教師発話を `<final>` として学習する。research評価は可視発話だけを評価しているため、350件の内部CoT自体がresearch監査済みという意味ではない。

旧コーパスにはANSI装飾と、数式境界として使われた非表示ASCII制御文字が含まれる。変換時にANSI装飾を除去し、その他の非空白制御文字を可視な `$` 区切りへ置換する。対象IDはdataset manifestへ保存する。

## データ再生成と構造検査

```bash
python prepare_datasets.py
python ../v4/train_v4_sft.py --config config.all500.json --structure-only
python ../v4/train_v4_sft.py --config config.research350.json --structure-only
```

完全なtoken長監査はモデル取得可能な環境で行う。

```bash
python ../v4/train_v4_sft.py --config config.all500.json --preflight-only
python ../v4/train_v4_sft.py --config config.research350.json --preflight-only
```

## ABCI実行

`pipelines/sft/v4` と本フォルダーを同じ相対配置でABCIへコピーする。

```bash
module load python/3.12/3.12.9
module load cuda/13.0/13.0.1
python -m venv .venv
source .venv/bin/activate
python -m pip install -r ../v4/requirements.txt

qsub -P "${ABCI_GROUP}" -v CONFIG_FILE=config.all500.json run_abci.pbs
qsub -P "${ABCI_GROUP}" -v CONFIG_FILE=config.research350.json run_abci.pbs
```

別々のoutput directoryを使うため、2ジョブは独立に実行できる。各条件は固定seed 42で10%を内部validationへ分け、validation lossが最小のcheckpointを最終adapterにする。

## 比較方法

学習後はbase、all500、research350をresearchの固定100ケースで同一条件評価する。主要指標は対話生成成功率と全体得点で、数学的正確性、誤り診断と回復、指導完了判定、足場かけ、情緒的支援、感情把握、重大失敗率を必ず併記する。

比較の優先順位は次のとおりとする。

1. research350がall500より重大失敗率と数学的正確性を改善するか
2. フィルタによるデータ減少で共感・足場かけ・生成成功率が悪化しないか
3. 両SFT条件がbaseを上回るか

固定評価100問はcheckpoint選択やハイパーパラメータ調整に使用しない。
