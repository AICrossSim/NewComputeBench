#!/bin/bash
# Finetuning script for BERT/RoBERTa on GLUE tasks

set -euo pipefail  # Exit on error, undefined vars, and pipe failures

# =================================================================================================
# Configuration
# =================================================================================================

# Cache directories
readonly SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Training parameters
readonly BATCH_SIZE=16
readonly SEED=42
readonly MAX_SEQ_LENGTH=512
readonly NUM_EPOCHS=10
readonly MAX_GRAD_NORM=0.5
readonly MASTER_PORT=14402


# Task configuration
USE_SINGLE_TASK=false
MODEL_NAME="FacebookAI/roberta-base"
TASK_NAME="mrpc"
# LR=1e-5  # For single task

# Multi-task configuration
TASK_LIST="stsb"
LR_LIST="1e-3"

# Optional features
PUSH_TO_HUB=false

# =================================================================================================

# Functions
# =================================================================================================

# Get the best metric for a given GLUE task
get_best_metric() {
    local task_name="$1"
    case "$task_name" in
        "cola")
            echo "matthews_correlation"
            ;;
        "stsb")
            echo "spearmanr"
            ;;
        "mnli"|"mrpc"|"sst2"|"rte"|"qnli"|"qqp")
            echo "accuracy"
            ;;
        *)
            echo "accuracy"  # Default metric
            ;;
    esac
}

# Run finetuning for a specific task
finetune() {
    local task_name="$1"
    local model_name="$2"
    local learning_rate="$3"
    local output_dir="$4"

    local run_name="ft_${task_name}_${learning_rate}"
    local best_metric
    best_metric="$(get_best_metric "$task_name")"

    echo "Starting finetuning: Task=$task_name, LR=$learning_rate, Model=$model_name"
    echo "Output directory: $output_dir"
    echo "Best metric: $best_metric"

    python -u -m torch.distributed.launch \
        --master_port="$MASTER_PORT" \
        --nproc_per_node=1 \
        --nnodes=1 \
        --node_rank=0 \
        --use_env \
        run_glue.py \
            --seed "$SEED" \
            --model_name_or_path "$model_name" \
            --task_name "$task_name" \
            --save_strategy best \
            --save_total_limit 2 \
            --eval_strategy epoch \
            --metric_for_best_model "$best_metric" \
            --load_best_model_at_end \
            --do_train \
            --do_eval \
            --fp16 \
            --ddp_timeout 180000 \
            --max_seq_length "$MAX_SEQ_LENGTH" \
            --per_device_train_batch_size "$BATCH_SIZE" \
            --per_device_eval_batch_size "$BATCH_SIZE" \
            --learning_rate "$learning_rate" \
            --num_train_epochs "$NUM_EPOCHS" \
            --output_dir "$output_dir" \
            --report_to wandb \
            --run_name "$run_name" \
            --wandb-tags "$task_name $learning_rate" \
            --transform_config "./transform_cfg.yaml" \
            --overwrite_output_dir \
            --max_grad_norm "$MAX_GRAD_NORM"
}

# Run single task finetuning
run_single_task() {
    local output_dir="${SCRIPT_DIR}/ckpt/roberta/finetune/${TASK_NAME}/lr_${LR}"

    echo "Single task finetuning"
    echo "Model: $MODEL_NAME"

    finetune "$TASK_NAME" "$MODEL_NAME" "$LR" "$output_dir"
}

# Run multi-task finetuning
run_multi_task() {
    echo "Multi-task finetuning"

    for lr in $LR_LIST; do
        for task in $TASK_LIST; do
            local output_dir="${SCRIPT_DIR}/ckpt/roberta/finetune/${task}/lr_${lr}"

            finetune "$task" "$MODEL_NAME" "$lr" "$output_dir"

            if [[ $? -ne 0 ]]; then
                echo "ERROR: Finetuning failed for task=$task with LR=$lr" >&2
                exit 1
            fi
        done
    done
}

# =================================================================================================


# Main execution
# =================================================================================================

main() {
    if [[ "$USE_SINGLE_TASK" == true ]]; then
        if [[ -z "${LR:-}" ]]; then
            echo "ERROR: LR must be set for single task mode" >&2
            exit 1
        fi
        run_single_task
    else
        run_multi_task
    fi

    echo "Finetuning completed successfully!"
}

# Run main function
main "$@"
