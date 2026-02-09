#!/bin/bash
# Finetuning script for BERT/RoBERTa on GLUE tasks with PIM simulation
# Optimized version supporting multiple PIM configs and improved GPU control

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

# Task configuration
WITH_TRACKING="${WITH_TRACKING:-false}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MODEL_NAME="FacebookAI/roberta-base"

# Tasks to finetune
TASK_LIST="cola mnli mrpc qnli qqp rte sst2 stsb"
LR_LIST="2e-5"

# PIM configurations to evaluate (from experiments/llm-pim/configs/)
PIM_CONFIG_LIST="sram pcm reram"

# =================================================================================================

# Functions
# =================================================================================================

# Run finetuning for a specific task
finetune() {
    local task_name="$1"
    local model_name="$2"
    local learning_rate="$3"
    local pim_config="$4"
    local output_dir="$5"

    local pim_config_path="./experiments/llm-pim/configs/${pim_config}.yaml"
    local run_name="ft_${task_name}_${pim_config}_${learning_rate}"

    echo "--------------------------------------------------------------------------------"
    echo "Starting finetuning: Task=$task_name, PIM=$pim_config, LR=$learning_rate"
    echo "Model: $model_name"
    echo "Output directory: $output_dir"
    echo "--------------------------------------------------------------------------------"

    local tracking_args=()
    if [[ "$WITH_TRACKING" == "true" ]]; then
        tracking_args+=(--with_tracking --report_to wandb)
    else
        tracking_args+=(--report_to none)
    fi

    # Using python instead of uv run for better compatibility
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python -u "${SCRIPT_DIR}/run_glue.py" \
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
        --pim_config_path "$pim_config_path" \
        "${tracking_args[@]}"
}

# =================================================================================================

# Main execution
# =================================================================================================

main() {
    echo "Starting multi-task finetuning process using pre-trained task-specific models..."

    for pim_config in $PIM_CONFIG_LIST; do
        for lr in $LR_LIST; do
            for task in $TASK_LIST; do
                local task_output_dir="${SCRIPT_DIR}/ckpt/roberta/${pim_config}/${task}/lr_${lr}"
                mkdir -p "$task_output_dir"
                
                # Use task-specific pre-trained models as a starting point, matching eval.sh logic
                local current_model="JeremiahZ/roberta-base-${task}"

                finetune "$task" "$current_model" "$lr" "$pim_config" "$task_output_dir"

                if [[ $? -ne 0 ]]; then
                    echo "ERROR: Finetuning failed for task=$task with PIM=$pim_config, LR=$lr" >&2
                    # Continue with next task
                fi
            done
        done
    done

    echo "All finetuning processes completed!"
}

# Run main function
main "$@"
