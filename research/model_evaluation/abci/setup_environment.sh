#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/.venv}"

source /etc/profile.d/modules.sh
module load python/3.12/3.12.9
module load cuda/13.0/13.0.1

python -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip
"${VENV_PATH}/bin/python" -m pip install -r "${PROJECT_DIR}/requirements-abci.txt"

echo "Environment ready: ${VENV_PATH}"
