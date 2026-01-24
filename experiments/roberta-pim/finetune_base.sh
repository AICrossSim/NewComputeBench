#!/bin/bash
# Finetuning script for BERT/RoBERTa on GLUE tasks with PIM simulation

set -euo pipefail  # Exit on error, undefined vars, and pipe failures

# =================================================================================================
# Configuration
# =================================================================================================

# Cache directories
readonly SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Training parameters
readonly BATCH_SIZE=16
readonly SEED=42
readonly MAX_LENGTH=128
readonly NUM_EPOCHS=3
readonly MASTER_PORT=14402

# Task configuration
USE_SINGLE_TASK=false
MODEL_NAME="FacebookAI/roberta-base"
TASK_NAME="mrpc"
# LR=2e-5  # For single task

# Multi-task configuration
# Default tasks for fine-tuning
TASK_LIST="cola mnli mrpc qnli qqp rte sst2 stsb"
LR_LIST="2e-5"

# =================================================================================================

# Functions
# =================================================================================================

# Run finetuning for a specific task
finetune() {
    local task_name="$1"
    local model_name="$2"
    local learning_rate="$3"
    local output_dir="$4"

    local run_name="ft_${task_name}_${learning_rate}"

    echo "--------------------------------------------------------------------------------"
    echo "Starting finetuning: Task=$task_name, LR=$learning_rate, Model=$model_name"
    echo "Output directory: $output_dir"
    echo "--------------------------------------------------------------------------------"

    # Using 1 GPU/process. run_glue.py uses Accelerator.
    # To use GPU, ensure CUDA_VISIBLE_DEVICES is set correctly in your environment.
    python -u "${SCRIPT_DIR}/run_glue.py" \
        --model_name_or_path "$model_name" \
        --task_name "$task_name" \
        --do_train \
        --do_eval \
        --max_length "$MAX_LENGTH" \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --per_device_eval_batch_size "$BATCH_SIZE" \
        --learning_rate "$learning_rate" \
        --num_train_epochs "$NUM_EPOCHS" \
        --output_dir "$output_dir" \
        --seed "$SEED" \
        --pim \
        --pim_config_path "./experiments/llm-pim/configs/sram.yaml" \
        --with_tracking \
        --report_to wandb
}

# Run single task finetuning
run_single_task() {
    if [[ -z "${LR:-}" ]]; then
        echo "ERROR: LR must be set for single task mode (e.g., LR=2e-5 bash finetune_base.sh)" >&2
        exit 1
    fi
    local output_dir="${SCRIPT_DIR}/ckpt/roberta/finetune/${TASK_NAME}/lr_${LR}"
    mkdir -p "$output_dir"
    finetune "$TASK_NAME" "$MODEL_NAME" "$LR" "$output_dir"
}

# Run multi-task finetuning
run_multi_task() {
    echo "Multi-task finetuning process started..."
    for lr in $LR_LIST; do
        for task in $TASK_LIST; do
            local task_output_dir="${SCRIPT_DIR}/ckpt/roberta/finetune/${task}/lr_${lr}"
            mkdir -p "$task_output_dir"

            finetune "$task" "$MODEL_NAME" "$lr" "$task_output_dir"

            if [[ $? -ne 0 ]]; then
                echo "ERROR: Finetuning failed for task=$task with LR=$lr" >&2
                # Continue with next task
            fi
        done
    done
}

# =================================================================================================

# Main execution
# =================================================================================================

main() {
    if [[ "$USE_SINGLE_TASK" == true ]]; then
        run_single_task
    else
        run_multi_task
    fi

    echo "Fine-tuning process completed!"
}

# Run main function
main "$@"
