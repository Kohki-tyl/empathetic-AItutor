#!/bin/bash
#PBS -P gcc50435
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N QLoRA_Swallow

# ==========================================
# ABCI 3.0 (PBS) の環境設定
# ==========================================
# PBSでは実行開始時ホームディレクトリに飛ばされるため、ジョブを投入したディレクトリに戻る
cd $PBS_O_WORKDIR

# ABCIのモジュールシステムを読み込み
source /etc/profile.d/modules.sh
module load python/3.12/3.12.9
module load cuda/12.1/12.1.1
module load cudnn/8.9/8.9.2

# 仮想環境の有効化
source ../.venv/bin/activate

# ==========================================
# 学習の実行
# ==========================================
echo "====================================="
echo "Starting QLoRA Fine-tuning on H200x4 (ABCI 3.0)"
echo "Date: $(date)"
echo "====================================="

FORCE_TORCHRUN=1 llamafactory-cli train qlora_sft.yaml

echo "====================================="
echo "Training Completed at: $(date)"
echo "====================================="