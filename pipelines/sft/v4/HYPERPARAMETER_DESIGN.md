# v4 SFTハイパーパラメータ設計

## 結論

v4の主比較では、v3と同じ`tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5`を基盤モデルとして使う。ABCI 3.0の`rt_HG`はH200を1基割り当てるため、8Bモデルを4-bit量子化する必然性は低い。数学能力の低下要因を増やさないよう、基盤モデルはBF16のまま凍結し、注意機構の4投影だけへ小容量LoRAを適用する。

## 採用設定

| 項目 | v2設計値 | v4採用値 | 判断 |
| --- | ---: | ---: | --- |
| 学習方式 | 4-bit QLoRA | BF16 LoRA | H200のメモリを利用し、量子化を比較上の交絡にしない |
| 最大系列長 | 4,096 | 8,192 | 1対話1レコードの判断記録と履歴を保持する |
| epoch数 | 3 | 4 | 94件の訓練集合で約96 optimizer stepを確保する |
| 学習率 | `2e-4` | `5e-5` | 少量データでの過大更新を抑える |
| 実効バッチサイズ | 16 | 4 | 1 epochあたりのoptimizer stepを確保する |
| LoRA rank | 64 | 16 | 104対話に対する過剰な適応容量を抑える |
| LoRA alpha | 128 | 32 | `alpha / rank = 2`を維持する |
| LoRA dropout | 0.05 | 0.05 | 小規模データの過学習を抑える |
| 適用層 | all-linear | q/k/v/o projection | MLPへの更新を避け、数学知識への干渉を抑える |
| 学習・検証分割 | 実行時 | 94/10 | 104対話をseed 42で固定し、最良epochを検証損失で選ぶ |
| packing | 未確定 | 無効 | 対話境界を維持し、マスク検証を単純化する |
| 損失対象 | assistantのみ | 構造化CoT 0.25、教師発話 1.0 | system、user、assistant role headerを除外し、最終発話を総lossでも重視する |

## 学習率とepoch数

104対話を94件の学習用と10件の検証用へ分ける。バッチサイズ1、gradient accumulation 4では、1 epochあたり24 optimizer step、4 epochで約96 stepとなる。v2設定の実効バッチ16では4 epochでも約24 stepしかなく、少量データでは学習率スケジュールとcheckpoint比較が粗くなる。

学習率はLoRAで一般的な`2e-4`より低い`5e-5`とする。本研究の目的は新しい数学知識の獲得ではなく、既存能力を保ちながら数学的検証、状態推定、支援選択の順序を学ぶことである。全4 epochを保存し、最後のepochではなく検証損失が最小のcheckpointを最終adapterとして保存する。

## LoRA容量

rank 64とall-linearの組合せは、429件のv2より少ない104件のv4には容量が大きい。v4ではrank 16とし、`q_proj`、`k_proj`、`v_proj`、`o_proj`だけを対象にする。これにより、教師としての注意配分と対話方略は調整しつつ、MLPを通じた広範な知識表現への介入を避ける。

PEFTはQLoRA型の設定として`target_modules="all-linear"`を提供しているが、これは必須ではない。本実験では性能最大化よりも、過去実験で低下した数学的正確性を保つことを優先する。rank 16では高rank向けのRank-Stabilized LoRAを使わず、通常の`alpha / rank`スケーリングを用いる。

## 系列長と損失マスク

Swallow 8Bのモデル設定は最大131,072位置を持つが、v4では計算量と対話長の実態から8,192を初期上限とする。学習スクリプトは自動切り詰めを行わない。1件でも上限を超えれば学習前に停止するため、実データの監査結果を見て、上限変更または教師ターン境界での明示的な分割を別条件として判断する。

TRLの`assistant_only_loss`は、chat templateがassistantマスクを返せる場合に限って利用できる。本フォルダーではライブラリ依存の暗黙マスクを使わず、各assistant発話の直前・直後のchat templateを比較して教師信号の位置を検証する。`<analysis>`と`<final>`およびassistant終端だけを損失対象とし、system、user、padding、assistant role headerは除外する。

さらにassistant範囲を`<analysis>`と`<final>`のtoken境界で分け、前者を0.25、後者を1.0としてtoken単位の重み付き交差エントロピーを計算する。構造化CoTを完全に除外すると、数学的検証・生徒状態の推定・支援選択の対応を学びにくい。一方、同じ重みにすると長い内部分析が学習信号を支配する。固定Swallow tokenizerによる実測はCoT 211,208 tokens、最終発話68,462 tokensであり、0.5ではCoTの総重みがまだ大きい。0.25なら重み付き寄与は概算43.5%対56.5%となるため、教師発話を主目的にしながらCoTも保持できる。重みは学習率を変えず、各ミニバッチの有効token重み合計で正規化する。

## 採用しなかった設定

- **4-bit QLoRA**：H200 1基でBF16 LoRAが可能であり、v4の主比較へ量子化条件を追加する必要がない。
- **rank 64 / all-linear**：104対話では表面的な方略やJudge固有表現を過学習する危険が高い。
- **学習率`2e-4`**：v2・v3で観測された数学的指導の低下を踏まえると、初回条件としては更新が強い。
- **packing**：サンプル効率よりも、対話境界とassistant-only maskの監査可能性を優先する。
- **検証なし**：最終epochを無条件で使うと過学習を検出できない。外部のtest-v4は最終評価専用とし、checkpoint選択には使わない。

## 実行後に必ず記録する値

- 実際の採択対話数、学習94件・検証10件のID
- 最大・平均・95パーセンタイル系列長
- assistant教師信号のトークン比率
- 各epochのtrain lossとvalidation loss
- 最良checkpointとそのepoch
- GPU、CUDA、PyTorch、Transformers、PEFTのバージョン
- 基盤モデルrevisionと入力JSONLのSHA-256

## 参照資料

- Swallowモデルカード: https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5
- Swallowモデル設定: https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5/blob/main/config.json
- PEFT LoRA: https://huggingface.co/docs/peft/main/package_reference/lora
- Transformers Trainer: https://huggingface.co/docs/transformers/main_classes/trainer
- TRL SFT Trainer: https://huggingface.co/docs/trl/main/sft_trainer
- ABCI 3.0ジョブ実行: https://docs.abci.ai/v3/ja/job-execution/
