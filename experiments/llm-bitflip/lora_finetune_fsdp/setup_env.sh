#!/bin/bash
# Set up a clean Python environment for bitflip-aware LoRA fine-tuning with FSDP2.
#
# Usage:
#   bash setup_env.sh
#
# This uses `uv` to create a venv at .venv/ pinned to a Python version that has
# the stdlib `tomllib` module (>=3.11, required by train.py / eval.py), then
# installs all dependencies. `uv` is the same tool used by the parent
# NewComputeBench project. Install it from https://docs.astral.sh/uv/ if missing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
# train.py / eval.py `import tomllib`, which is stdlib only on Python >= 3.11.
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

echo "============================================="
echo "Setting up environment for BitFlip LoRA FSDP2"
echo "============================================="

# ---- Check for uv ----
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' not found on PATH."
    echo "Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "or see https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi
echo "Using $(uv --version)"

# ---- Create venv with a pinned Python (uv fetches it if not present) ----
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment at ${VENV_DIR} (Python ${PYTHON_VERSION})..."
    uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
else
    echo "Virtual environment already exists at ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
# Install into the venv created above (uv pip targets $VIRTUAL_ENV).
echo "Active Python: $(python3 --version)"

# ---- Install PyTorch (nightly for FSDP2 support) ----
echo ""
echo "Installing PyTorch nightly (CUDA 12.4)..."
echo "If you need a different CUDA version, edit this script."
uv pip install --upgrade pip
uv pip install --prerelease=allow torch --index-url https://download.pytorch.org/whl/nightly/cu124

# ---- Install triton ----
echo ""
echo "Installing Triton..."
uv pip install triton

# ---- Install torchtitan (editable, from the submodule) ----
# torchtitan/ is a git submodule pinned to commit 0e0590c1. If it's empty,
# initialise it before installing.
if [ ! -f "${SCRIPT_DIR}/torchtitan/pyproject.toml" ]; then
    echo ""
    echo "torchtitan submodule not initialised — running git submodule update..."
    (cd "${SCRIPT_DIR}" && git submodule update --init torchtitan)
fi
echo ""
echo "Installing torchtitan (editable)..."
uv pip install -e "${SCRIPT_DIR}/torchtitan"

# ---- Install other dependencies ----
echo ""
echo "Installing remaining dependencies..."
uv pip install \
    transformers \
    datasets \
    safetensors \
    tokenizers \
    accelerate \
    sentencepiece \
    'mase-triton>=0.0.7'

# ---- Verify installation ----
echo ""
echo "============================================="
echo "Verifying installation..."
echo "============================================="
python3 -c "
import sys
print(f'Python version: {sys.version.split()[0]}')

# train.py / eval.py rely on stdlib tomllib (Python >= 3.11)
import tomllib
print('tomllib (stdlib): available')

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
echo "  bash run.sh                  # 4-GPU, Llama 3 70B (config_70b.toml)"
echo "============================================="
