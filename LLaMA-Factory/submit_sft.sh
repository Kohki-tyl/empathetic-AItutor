#!/bin/bash
#$ -l rt_HF=1           # H200ノードを1台（GPU 4基）要求
#$ -l h_rt=24:00:00     # 最大実行時間を24時間に設定 (余裕を持って)
#$ -j y                 # 標準出力と標準エラー出力を同じファイルにまとめる
#$ -cwd                 # 現在のディレクトリで実行

# モジュールと環境のロード
source /etc/profile.d/modules.sh
module load python/3.12/3.12.9
module load cuda/12.1/12.1.1
module load cudnn/8.9/8.9.2

# 仮想環境の有効化（パスはご自身の環境に合わせてください）
source ../.venv/bin/activate

# LLaMA-Factoryのディレクトリに移動
cd ~/empathetic-AItutor/LLaMA-Factory

# マルチGPU (H200 x 4) をフル活用して学習を開始
# FORCE_TORCHRUN=1 を付けることで、4つのGPUに自動で分散処理されます
FORCE_TORCHRUN=1 llamafactory-cli train qlora_sft.yaml