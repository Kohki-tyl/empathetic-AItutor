# QLoRAによるSFT（教師あり微調整）実行戦略

本ドキュメントは、LLaMA-Factoryを用いたSwallow-70Bモデルに対するQLoRAファインチューニングの実行戦略と、最適なハイパーパラメーターの設定根拠をまとめたものである。

## 1. 基本アプローチとフォーマット
* **ChatML形式の採用**: 特殊トークン（`<|im_start|>`および`<|im_end|>`）を利用して文脈の境界を明示し、「system」「user」「assistant」という役割を構造的に分離するChatML形式を採用する[cite: 8]。これにより、モデルはどこからが自身の回答を生成すべき箇所かをトークンレベルで正確に認識可能となる[cite: 8]。
* **QLoRAアーキテクチャ**: ベースモデルの重みを4ビットのNormalFloat（NF4）形式に量子化して凍結し、学習可能なLoRAアダプターのみを16ビット精度（BF16など）で保持する[cite: 8]。これによりメモリ消費を抜本的に削減する[cite: 8]。

## 2. ハイパーパラメーターの最適化戦略
先行研究のベストプラクティスに基づき、過学習（丸暗記）を防ぎつつ汎化性能を最大化するための設定を行う[cite: 8]。

* **LoRAの適用範囲 (Target Modules)**: アテンション機構だけでなく多層パーセプトロン（MLP）のゲート層や出力層を含む「すべての線形層（all-linear）」を対象とする[cite: 8]。これによりFFT（フルファインチューニング）に最も肉薄する性能が得られることが確認されている[cite: 8]。
* **Rank ($r$) と Alpha ($\alpha$)**: 一般的なフォーマット適応やトーンの調整において十分な効果が得られる $r=16$ を採用する[cite: 8]。アルファ（$\alpha$）は一般的なヒューリスティックに従い、$r$ と等倍の $\alpha=16$ に設定する[cite: 8]。
* **学習率 (Learning Rate)**: LoRA/QLoRAでは、新設された行列に対して十分な勾配信号を与える必要があるため、FFTよりも1桁高い $1 \times 10^{-4}$ から $2 \times 10^{-4}$ が推奨される[cite: 8]。本設定ではその中間値である `1.5e-4` を採用する。
* **エポック数 (Epochs)**: モデルが訓練データの表面的なパターンや特定のプロンプトを丸暗記してしまう過学習を防ぐため、エポック数は1から3の間に留めることが強く推奨されており、本設定では上限の `3.0` とする[cite: 8]。
* **バッチサイズ**: 低い学習率と極めて大きな実効バッチサイズ（Effective Batch Size）の組み合わせが、ベンチマークにおける汎化性能を高める[cite: 8]。本構成では勾配累積を用いて実効バッチサイズを引き上げる定石を採用する[cite: 8]。
* **スケジューラーとウォームアップ**: 全体の数パーセントを線形に増加させる「ウォームアップ」により学習初期の急激な勾配の変化を防ぐ[cite: 8]。その後「コサイン減衰（Cosine Decay）」を用いて、最適化の最終段階での滑らかな収束を保証する[cite: 8]。

## 3. 設定ファイル (`qlora_sft.yaml`)
上記戦略を反映したLLaMA-Factory用の設定ファイル構成である。

```yaml
# ===== Model & Template =====
model_name_or_path: tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3
template: chatml        # ChatML形式を用いることで、特殊トークンによる構造的境界を明確化
quantization_bit: 4     # QLoRAによる4ビット量子化（NF4）でメモリ消費を抜本的に削減

# ===== Hardware Optimization =====
bf16: true
pure_bf16: true
flash_attn: fa2

# ===== Dataset =====
dataset: math_tutor_train
eval_dataset: math_tutor_val
cutoff_len: 2048

# ===== LoRA Settings =====
stage: sft
do_train: true
finetuning_type: lora
lora_target: all        # 「すべての線形層（all-linear）」を対象とし、FFTに肉薄する性能を引き出す
lora_rank: 16           # ベースライン確立のため、軽量で標準的な 16 を設定
lora_alpha: 16          # ヒューリスティックに従い、アルファ（α）は r と等倍の 16 に設定
lora_dropout: 0.05

# ===== Training Hyperparameters =====
per_device_train_batch_size: 2
gradient_accumulation_steps: 8 # 勾配累積を用いて実効バッチサイズを巨大化（4GPUで計64）
learning_rate: 1.5e-4          # LoRAアダプターに十分な勾配信号を与える 1.5e-4 を採用
num_train_epochs: 3.0          # 過学習を防ぐため、推奨される 1〜3 エポックの上限で学習を停止
lr_scheduler_type: cosine      # 最終段階での滑らかな収束を保証するコサイン減衰を適用
warmup_ratio: 0.05             # 学習初期の急激な勾配の変化を防ぐため、全体の5%のウォームアップ期間を設定
weight_decay: 0.05

# ===== Evaluation & Logging =====
eval_strategy: steps
eval_steps: 50
logging_steps: 10
save_steps: 50
plot_loss: true
overwrite_output_dir: true

# ===== Output =====
output_dir: saves/Swallow-70B/lora/sft