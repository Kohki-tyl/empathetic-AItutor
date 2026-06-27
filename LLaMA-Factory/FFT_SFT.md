# フルファインチューニング（FFT）によるSFT実行戦略

本ドキュメントは、LLaMA-Factoryを用いたSwallow-70Bモデルに対するフルファインチューニング（全パラメーター更新）の実行戦略と、最適なハイパーパラメーターの設定根拠をまとめたものである。

## 1. 基本アプローチとフォーマット
* **ChatML形式の採用**: 特殊トークンを利用して文脈の境界を明示し、「system」「user」「assistant」という役割を構造的に分離する。これにより、モデルはどこからが自身の回答を生成すべき箇所かを正確に認識可能となる。
* **FFT（Full Fine-Tuning）アーキテクチャ**: 数十億から数百億に及ぶLLMの全パラメーターを勾配降下法によって直接更新するアプローチである[cite: 8]。モデルの根源的な挙動を修正する際に必須となる手法であり、品質面での上限が最も高い[cite: 8]。

## 2. ハイパーパラメーターの最適化戦略
FFTでは、LoRA以上に「破局的忘却（Catastrophic Forgetting）」と「過学習」を防ぐためのシビアな調整が必要となる。

* **学習率 (Learning Rate)**: 事前学習で獲得した汎用的な知識の表現を破壊しないよう、比較的小さな学習率が求められる[cite: 8]。概ね $1 \times 10^{-5}$ から $2 \times 10^{-5}$ が安全なデフォルト値とされるため[cite: 8]、本設定ではその中間値である `1.5e-5` を採用する。
* **エポック数 (Epochs)**: モデルが訓練データの表面的なパターンを丸暗記してしまう過学習を防ぐため、エポック数を1から3の間に留めることが強く推奨される[cite: 8]。
* **バッチサイズ**: 低い学習率と極めて大きな実効バッチサイズ（Effective Batch Size）の組み合わせが、ベンチマークにおける汎化性能を高める[cite: 8]。VRAMの制約で物理的なバッチサイズを拡大できない場合は、勾配累積を用いて実効バッチサイズを引き上げる定石を採用する[cite: 8]。
* **スケジューラーとウォームアップ**: 全体の数パーセントを線形に増加させる「ウォームアップ」により学習初期の急激な勾配の変化を防ぐ[cite: 8]。その後「コサイン減衰（Cosine Decay）」を用いて、最適化の最終段階での滑らかな収束を保証する[cite: 8]。

## 3. 設定ファイル (`fft_sft.yaml`)
LoRA固有の設定を排除し、全パラメーターをBFloat16精度で直接更新するための設定ファイル構成である。

```yaml
# ===== Model & Template =====
model_name_or_path: tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3
template: chatml        # ChatML形式を用いることで、特殊トークンによる構造的境界を明確化[cite: 8]

# ===== Hardware Optimization =====
bf16: true              # BFloat16精度で学習を実行
pure_bf16: true
flash_attn: fa2         # VRAM消費の大幅削減と計算の高速化に必須

# ===== Dataset =====
dataset: math_tutor_train
eval_dataset: math_tutor_val
cutoff_len: 2048

# ===== FFT Settings =====
stage: sft
do_train: true
finetuning_type: full   # 全パラメーターの更新を指定 (LoRA関連のパラメータは削除)

# ===== Training Hyperparameters =====
per_device_train_batch_size: 1 # FFTはVRAM消費が激しいため、物理バッチサイズは1に制限
gradient_accumulation_steps: 16 # 勾配累積を増やし、実効バッチサイズを巨大化（4GPUで計64を維持）[cite: 8]
learning_rate: 1.5e-5          # 既存知識の破壊を防ぐため、LoRAより1桁低い 1.5e-5 を採用[cite: 8]
num_train_epochs: 3.0          # 過学習を防ぐため、推奨される 1〜3 エポックの上限で学習を停止[cite: 8]
lr_scheduler_type: cosine      # 最終段階での滑らかな収束を保証するコサイン減衰を適用[cite: 8]
warmup_ratio: 0.05             # 学習初期の急激な勾配の変化を防ぐため、全体の5%のウォームアップ期間を設定[cite: 8]
weight_decay: 0.05             # 過学習防止の正則化

# ===== Evaluation & Logging =====
eval_strategy: steps
eval_steps: 50
logging_steps: 10
save_steps: 50
plot_loss: true
overwrite_output_dir: true

# ===== Output =====
output_dir: saves/Swallow-70B/full/sft