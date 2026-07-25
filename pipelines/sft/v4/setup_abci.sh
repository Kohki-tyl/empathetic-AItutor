#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
source /etc/profile.d/modules.sh
module load python/3.12/3.12.9
module load cuda/13.0/13.0.1

if [ ! -d .venv ]; then
  python -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_v4_sft.py --config config.json --structure-only

echo "ABCI environment and v4 SFT bundle are ready."
