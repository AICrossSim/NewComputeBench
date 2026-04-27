#!/bin/bash
# Evaluation script for BERT/RoBERTa on GLUE tasks with PIM simulation

set -euo pipefail  # Exit on error, undefined vars, and pipe failures

# =================================================================================================
# Configuration
# =================================================================================================

# Cache directories
readonly SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Evaluation parameters
readonly BATCH_SIZE=32
readonly SEED=42
readonly MAX_LENGTH=128
readonly MASTER_PORT=14402

# Task configuration
WITH_TRACKING="${WITH_TRACKING:-false}"
USE_CPU="${USE_CPU:-false}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

# Tasks to evaluate (GLUE benchmark)
TASK_LIST="cola mnli mrpc qnli qqp rte sst2 stsb"

# PIM configurations to evaluate (from experiments/roberta-pim/configs/)
PIM_CONFIG_LIST="sram pcm reram"

# =================================================================================================

# Functions
# =================================================================================================

# Run evaluation for a specific task and PIM config
evaluate() {
    local task_name="$1"
    local model_name="$2"
    local pim_config="$3"
    local output_dir="$4"

    local pim_config_path="./experiments/roberta-pim/configs/${pim_config}.yaml"
    local run_name="eval_${task_name}_${pim_config}"

    echo "--------------------------------------------------------------------------------"
    echo "Starting evaluation: Task=$task_name, PIM=$pim_config, Model=$model_name"
    echo "Output directory: $output_dir"
    echo "--------------------------------------------------------------------------------"

    # Default: use one GPU. Set USE_CPU=true to force CPU.
    local tracking_args=()
    if [[ "$WITH_TRACKING" == "true" ]]; then
        tracking_args+=(--with_tracking --report_to wandb)
    fi

    if [[ "$USE_CPU" == "true" ]]; then
        CUDA_VISIBLE_DEVICES="" uv run python -u "${SCRIPT_DIR}/run_glue.py" \
            --model_name_or_path "$model_name" \
            --task_name "$task_name" \
            --do_eval \
            --max_length "$MAX_LENGTH" \
            --per_device_eval_batch_size "$BATCH_SIZE" \
            --output_dir "$output_dir" \
            --seed "$SEED" \
            --pim \
            --pim_config_path "$pim_config_path" \
            "${tracking_args[@]}"
    else
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" uv run python -u "${SCRIPT_DIR}/run_glue.py" \
            --model_name_or_path "$model_name" \
            --task_name "$task_name" \
            --do_eval \
            --max_length "$MAX_LENGTH" \
            --per_device_eval_batch_size "$BATCH_SIZE" \
            --output_dir "$output_dir" \
            --seed "$SEED" \
            --pim \
            --pim_config_path "$pim_config_path" \
            "${tracking_args[@]}"
    fi
}

# =================================================================================================

# Main execution
# =================================================================================================

main() {
    echo "Starting evaluation process..."

    for pim_config in $PIM_CONFIG_LIST; do
        for task in $TASK_LIST; do
            local task_output_dir="${SCRIPT_DIR}/eval_results/${pim_config}/${task}"
            mkdir -p "$task_output_dir"
            MODEL_NAME="JeremiahZ/roberta-base-${task}"

            evaluate "$task" "$MODEL_NAME" "$pim_config" "$task_output_dir"

            if [[ $? -ne 0 ]]; then
                echo "ERROR: Evaluation failed for task=$task with PIM=$pim_config" >&2
                # Continue with next task/config
            fi
        done
    done

    echo "All evaluations completed successfully!"
}

# Run main function
main "$@"
