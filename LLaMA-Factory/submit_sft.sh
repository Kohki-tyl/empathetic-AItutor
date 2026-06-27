#!/bin/bash
#$ -l rt_HF=1           # 【必須】H200ノードを1台（GPU 4基）要求
#$ -l h_rt=24:00:00     # ジョブの最大実行時間 (24時間)
#$ -j y                 # 標準出力と標準エラー出力を同じログファイルに結合
#$ -cwd                 # 現在のディレクトリを作業ディレクトリとして実行
#$ -N QLoRA_Swallow     # ジョブの名前（qstatで表示されます）

# ==========================================
# 1. 環境のセットアップ
# ==========================================
# ABCIのモジュールシステムを読み込み
source /etc/profile.d/modules.sh

# 必要なモジュールをロード (※環境に合わせてバージョンは調整してください)
module load python/3.12/3.12.9
module load cuda/12.1/12.1.1
module load cudnn/8.9/8.9.2

# 仮想環境の有効化（パスはご自身の環境に合わせてください）
# プロジェクトルートに .venv がある想定の相対パスです
source ../.venv/bin/activate

# ==========================================
# 2. 学習の実行
# ==========================================
echo "====================================="
echo "Starting QLoRA Fine-tuning on H200x4"
echo "Date: $(date)"
echo "====================================="

# LLaMA-FactoryのマルチGPU学習コマンド
# FORCE_TORCHRUN=1 を付けることで、1ノード内の4つのGPUに自動で処理が分散されます
FORCE_TORCHRUN=1 llamafactory-cli train qlora_sft.yaml

echo "====================================="
echo "Training Completed at: $(date)"
echo "====================================="