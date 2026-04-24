#!/bin/bash
# Set up a clean Python environment for bitflip-aware LoRA fine-tuning with FSDP2.
#
# Usage:
#   bash setup_env.sh
#
# This creates a venv at .venv/ and installs all dependencies.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "============================================="
echo "Setting up environment for BitFlip LoRA FSDP2"
echo "============================================="

# ---- Create venv ----
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
else
    echo "Virtual environment already exists at ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

# ---- Install PyTorch (nightly for FSDP2 support) ----
echo ""
echo "Installing PyTorch nightly (CUDA 12.4)..."
echo "If you need a different CUDA version, edit this script."
pip install --upgrade pip
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124

# ---- Install triton ----
echo ""
echo "Installing Triton..."
pip install triton

# ---- Install torchtitan (editable, from cloned repo) ----
echo ""
echo "Installing torchtitan (editable)..."
pip install -e "${SCRIPT_DIR}/torchtitan"

# ---- Install other dependencies ----
echo ""
echo "Installing remaining dependencies..."
pip install \
    transformers \
    datasets \
    safetensors \
    tokenizers \
    accelerate \
    sentencepiece \
    tomli

# ---- Verify installation ----
echo ""
echo "============================================="
echo "Verifying installation..."
echo "============================================="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')

import triton
print(f'Triton version: {triton.__version__}')

# Verify FSDP2
from torch.distributed.fsdp import fully_shard
print('FSDP2 (fully_shard) available: True')

# Verify torchtitan
from torchtitan.models.llama3 import llama3_configs
print(f'torchtitan Llama3 configs: {list(llama3_configs.keys())}')

print()
print('All checks passed!')
"

echo ""
echo "============================================="
echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "To run training:"
echo "  bash run.sh                  # 8-GPU, Llama 3.1 70B"
echo "  bash run.sh config_debug.toml  # 1-GPU, debug model"
echo "============================================="
