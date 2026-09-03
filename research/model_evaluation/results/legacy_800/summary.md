# 教師モデル評価サマリー

## 条件別主要結果

| 条件 | 予定 | 対話生成成功率 | 評価数 | 全体得点平均 / 60 | 全適用軸8点以上 | 重大失敗率 | 指導完了率 | 類似問題正答率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy-500-corpus | 500 | 100.0% | 500 | 52.32 | 73.2% | 16.6% | 70.8% | NA |
| legacy-800-additional-corpus | 300 | 100.0% | 300 | 51.32 | 70.7% | 18.3% | 67.3% | NA |

## legacy-500-corpus の6軸

| 評価軸 | n | 平均 | 中央値 | NA |
| --- | ---: | ---: | ---: | ---: |
| mathematical_accuracy | 500 | 8.708 | 10.0 | 0 |
| error_diagnosis_recovery | 443 | 8.2799 | 10.0 | 57 |
| instruction_completion | 500 | 8.134 | 10.0 | 0 |
| scaffolding | 497 | 8.8612 | 10.0 | 3 |
| emotional_support | 499 | 9.3046 | 10.0 | 1 |
| emotion_recognition | 500 | 8.87 | 9.0 | 0 |

## legacy-800-additional-corpus の6軸

| 評価軸 | n | 平均 | 中央値 | NA |
| --- | ---: | ---: | ---: | ---: |
| mathematical_accuracy | 300 | 8.6267 | 10.0 | 0 |
| error_diagnosis_recovery | 237 | 7.9198 | 10.0 | 63 |
| instruction_completion | 300 | 7.8967 | 10.0 | 0 |
| scaffolding | 288 | 8.7535 | 10.0 | 12 |
| emotional_support | 279 | 8.8315 | 9.0 | 21 |
| emotion_recognition | 300 | 8.82 | 9.0 | 0 |

## 対応比較

- legacy-800-additional-corpus - legacy-500-corpus: 計画共通0件、生成成功率差=None (95%区間=[None, None])。評価共通0件、全体得点差=None (95%区間=[None, None])
