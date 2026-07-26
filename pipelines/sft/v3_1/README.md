# v3.1 SFT

v3.1は、v3の397対話・1626教師ターン、順序、user発話、教師の最終発話、7:1の学習・検証比、QLoRA設定を維持し、system promptと教師CoTの分析区分だけをv4形式へ整合した比較条件である。

変換は次の対応だけを行う。

| v3 | v3.1/v4形式 |
| --- | --- |
| `数学的検証` | `数学的評価` |
| `認知状態`＋`感情状態` | `生徒状態` |
| `次の一歩` | `支援判断` |

`final`発話は変更しない。したがって、v3.1はprompt・出力区分の整合効果を見る条件であり、v4の104対話コーパスや重み付き損失を導入するものではない。

```bash
python SFT_abci/sft/v3_1/prepare_v3_1_sft_dataset.py
python SFT_abci/sft/validate_sft_dataset.py \
  SFT_abci/sft/v3_1/data/v3_1_cot_sft.jsonl \
  --manifest SFT_abci/sft/v3_1/data/v3_1_cot_sft_manifest.json \
  --require-cot
qsub SFT_abci/jobs/sft/v3_1/submit_v3_1_cot_sft.sh
```

学習後のadapterは`SFT_abci/LLaMA-Factory/saves/Swallow-8B/lora/v3_1_cot_sft`へ保存する。test-v5では教師配信名`v3.1-sft`、出力先`data/gpt54_student/v3_1_sft/primary_60/dialogues.jsonl`を使う。
