#!/bin/bash

# Evaluation-only run on train/validation splits with baseline LoRA (no bitflip, no trainable params).
# Usage: ./eval-lora-baseline.sh [num_processes] [model_name_or_path] [per_device_batch_size] [block_size] [eval_max_steps]

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_SCRIPT="${SCRIPT_DIR}/run_clm_no_trainer.py"
TRANSFORM_CFG="${SCRIPT_DIR}/transform_cfg_baseline.toml"

NUM_PROCESSES=${1:-8}
MODEL_NAME_OR_PATH=${2:-"unsloth/Llama-3.1-8B"}
PER_DEVICE_BATCH_SIZE=${3:-1}
BLOCK_SIZE=${4:-2048}
EVAL_MAX_STEPS=${5:-64}

OUTPUT_DIR="${SCRIPT_DIR}/output/$(basename ${MODEL_NAME_OR_PATH})-lora-baseline-eval"
WANDB_TAGS="${MODEL_NAME_OR_PATH},baseline,eval"

echo "============================================"
echo "Evaluation Only (LoRA Baseline, No Bitflip):"
echo "============================================"
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Number of Processes: ${NUM_PROCESSES}"
echo "Per Device Batch Size: ${PER_DEVICE_BATCH_SIZE}"
echo "Block Size: ${BLOCK_SIZE}"
if [ "${EVAL_MAX_STEPS}" -gt 0 ]; then
  echo "Eval Max Steps per split: ${EVAL_MAX_STEPS}"
else
  echo "Eval Max Steps per split: full dataset"
fi
echo "Output Directory: ${OUTPUT_DIR}"
echo "Wandb Tags: ${WANDB_TAGS}"
echo "============================================"

uv run accelerate launch --num_processes=${NUM_PROCESSES} \
  "${RUN_SCRIPT}" \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --dataset_name Cheng98/fineweb-edu-1.25B \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type linear \
    --output_dir ${OUTPUT_DIR} \
    --preprocessing_num_workers 32 \
    --trust_remote_code \
    --with_tracking \
    --report_to wandb \
    --transform_cfg "${TRANSFORM_CFG}" \
    --block_size ${BLOCK_SIZE} \
    --eval_only \
    --eval_max_steps ${EVAL_MAX_STEPS} \
    --wandb_tags ${WANDB_TAGS}
