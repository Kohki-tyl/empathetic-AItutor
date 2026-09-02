# Baseline と v3 CoT SFT の比較表

## A. 対話全体得点

| 項目 | Baseline | v3 CoT SFT |
| --- | ---: | ---: |
| 対話全体得点（平均、60点満点） | 29.24 | 36.88 |

対応差は +7.65点、95%信頼区間は4.23〜11.06点。

## B. 6評価軸

| 日本語項目名 | 英語項目名 | Baseline | v3 CoT SFT |
| --- | --- | ---: | ---: |
| 数学的正確性 | Mathematical accuracy | 4.83 | 7.01 |
| 誤り診断と回復 | Error diagnosis and recovery | 3.73 | 5.71 |
| 指導完了判定 | Instruction completion | 4.71 | 5.31 |
| 足場かけ | Scaffolding | 4.09 | 5.33 |
| 情緒的支援 | Emotional support | 5.80 | 6.60 |
| 感情把握 | Emotion recognition | 5.73 | 6.72 |

## C. 主要な割合指標

| 日本語項目名 | 英語項目名 | Baseline | v3 CoT SFT |
| --- | --- | ---: | ---: |
| 対話生成成功率 | Dialogue generation success rate | 99.0% | 100.0% |
| 全適用軸8点以上率 | All applicable axes at least 8 | 17.2% | 18.0% |
| 重大失敗率 | Critical failure rate | 72.7% | 40.0% |
| 指導完了率 | Instruction completion rate | 39.4% | 34.0% |
| 類似問題正答率 | Near-transfer accuracy | 61.6% | 59.0% |
