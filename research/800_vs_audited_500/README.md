# 未監査800件 vs 監査済み500件 SFT比較

旧500対話に新規合成300対話を加えた800対話を母集団とし、全800件を監査結果でゲートしない条件と、Research監査の厳格基準を通過した558件から固定seedで500件を抽出した条件を比較する。

## 条件

| 条件 | コーパス | 内部train/validation |
| --- | ---: | ---: |
| unaudited800 | 800 | 720 / 80 |
| audited500 | 500 | 450 / 50 |

監査通過条件は、`evaluation_status=evaluated`、数学的正確性10点、NAを除く全軸8点以上、重大失敗なしである。通過558件をsource IDでソートしてseed 42でシャッフルし、先頭500件を採択する。未解決の修正指示、確定評価セット漏洩、教師内部CoT品質は未適用ゲートである。

両条件とも保存済みの `thought_process`、`student_emotion`、`next_step_plan` を `<analysis>`、可視教師発話を `<final>` として学習する。ANSI装飾を除去し、非空白ASCII制御文字を可視な `$` へ置換する。未監査800件の `math_train_692` は末尾が未応答の生徒発話だったため、その末尾だけを除いてassistant終了のSFT構造にした。全処理対象IDはmanifestに記録している。

## データ再生成と検査

```bash
python prepare_datasets.py
python -m unittest discover -s tests -v
python ../../pipelines/sft/v4/train_v4_sft.py --config config.unaudited800.json --structure-only
python ../../pipelines/sft/v4/train_v4_sft.py --config config.audited500.json --structure-only
```

モデルを取得できる環境では `--structure-only` を `--preflight-only` に替えてtoken長も監査する。

## ABCI実行

`pipelines/sft/v4` と本フォルダーを、ABCI上でそれぞれ `v4` と本実験フォルダーが兄弟になる配置へコピーする。

```bash
module load python/3.12/3.12.9
module load cuda/13.0/13.0.1
python -m venv .venv
source .venv/bin/activate
python -m pip install -r ../v4/requirements.txt

qsub -P "${ABCI_GROUP}" -v CONFIG_FILE=config.unaudited800.json run_abci.pbs
qsub -P "${ABCI_GROUP}" -v CONFIG_FILE=config.audited500.json run_abci.pbs
```

基盤モデル、LoRA、seed、epochなどは `500_vs_350` と同一で、出力先だけを分離している。データ量も変わる実用比較なので、監査だけの因果効果を分離する実験ではない。学習後はbase、unaudited800、audited500を固定100ケースで同一条件評価する。
