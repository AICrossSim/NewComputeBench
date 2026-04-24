#!/bin/bash
# Launch bitflip-aware LoRA fine-tuning with FSDP2
#
# Usage:
#   Single node, 8 GPUs:
#     bash run.sh
#
#   Custom config:
#     bash run.sh config_debug.toml
#
#   Multi-node (2 nodes, 8 GPUs each):
#     # On node 0:
#     MASTER_ADDR=<node0_ip> MASTER_PORT=29500 NNODES=2 NODE_RANK=0 bash run.sh
#     # On node 1:
#     MASTER_ADDR=<node0_ip> MASTER_PORT=29500 NNODES=2 NODE_RANK=1 bash run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/config.toml}"

# Distributed settings (override via env vars)
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "============================================="
echo "Bitflip-aware LoRA Fine-tuning with FSDP2"
echo "============================================="
echo "Config:         ${CONFIG}"
echo "GPUs per node:  ${NPROC_PER_NODE}"
echo "Nodes:          ${NNODES}"
echo "Node rank:      ${NODE_RANK}"
echo "Master:         ${MASTER_ADDR}:${MASTER_PORT}"
echo "============================================="

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/train.py" \
    --config "${CONFIG}"
