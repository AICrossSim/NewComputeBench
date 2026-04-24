#!/bin/bash
#SBATCH --job-name=bitflip-lora-70b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Adjust the above for your cluster. For multi-node:
#   --nodes=2 --ntasks-per-node=8 --gpus-per-node=8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/config.toml}"

# Create logs dir
mkdir -p logs

# Activate environment
source "${SCRIPT_DIR}/.venv/bin/activate"

# NCCL settings (tune for your cluster)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Get master node info
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "============================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "GPUs per node: ${SLURM_GPUS_PER_NODE}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Config: ${CONFIG}"
echo "============================================="

srun torchrun \
    --nproc_per_node="${SLURM_GPUS_PER_NODE}" \
    --nnodes="${SLURM_NNODES}" \
    --node_rank="${SLURM_NODEID}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/train.py" \
    --config "${CONFIG}"
