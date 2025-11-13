#!/bin/bash

# Parameterized fine-tuning script with proper max_train_steps calculation
# Usage: ./parameterized-fine-tune-clm-v2.sh [num_processes] [model_name_or_path] [per_device_train_batch_size] [learning_rate] [weight_decay] [gradient_accumulation_steps] [block_size]

# Default parameters
NUM_PROCESSES=${1:-1}
MODEL_NAME_OR_PATH=${2:-"AICrossSim/clm-60m"}
PER_DEVICE_TRAIN_BATCH_SIZE=${3:-8}
LEARNING_RATE=${4:-"2e-4"}
WEIGHT_DECAY=${5:-"0.01"}
GRADIENT_ACCUMULATION_STEPS=${6:-2}
BLOCK_SIZE=${7:-1024}
ATTN_IMPL=${8:-"eager"}  # New parameter for attention implementation

# Function to get model parameters count
get_model_params() {
    case "$1" in
        "AICrossSim/clm-60m")
            echo "60000000"
            ;;
        "AICrossSim/clm-200m")
            echo "200000000"
            ;;
        "AICrossSim/clm-400m")
            echo "400000000"
            ;;
        "AICrossSim/clm-600m")
            echo "600000000"
            ;;
        "AICrossSim/clm-1.1b")
            echo "1100000000"
            ;;
        *)
            echo "Unknown model: $1" >&2
            exit 1
            ;;
    esac
}

# Calculate derived parameters
N_PARAMS=$(get_model_params "$MODEL_NAME_OR_PATH")
N_FINE_TUNE_TOKENS=$((22 * N_PARAMS / 100))
N_SAMPLES_PER_STEP=$((NUM_PROCESSES * PER_DEVICE_TRAIN_BATCH_SIZE))
N_TOKENS_PER_STEP=$((N_SAMPLES_PER_STEP * BLOCK_SIZE))

# Calculate max_train_steps using ceiling division: (a + b - 1) / b
MAX_TRAIN_STEPS=$(((N_FINE_TUNE_TOKENS + N_TOKENS_PER_STEP - 1) / N_TOKENS_PER_STEP))

echo "Calculated max_train_steps: ${MAX_TRAIN_STEPS}"


# Generate output directory name
OUTPUT_DIR="./output/$(basename ${MODEL_NAME_OR_PATH})-optical"

# Generate wandb tags
WANDB_TAGS="${MODEL_NAME_OR_PATH},lr${LEARNING_RATE},steps${MAX_TRAIN_STEPS}"

echo "============================================"
echo "Fine-tuning Configuration:"
echo "============================================"
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Model Parameters: ${N_PARAMS}"
echo "Number of Processes: ${NUM_PROCESSES}"
echo "Per Device Train Batch Size: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Weight Decay: ${WEIGHT_DECAY}"
echo "Gradient Accumulation Steps: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Block Size: ${BLOCK_SIZE}"
echo ""
echo "Calculated Parameters:"
echo "Fine-tune Tokens: ${N_FINE_TUNE_TOKENS}"
echo "Samples per Step: ${N_SAMPLES_PER_STEP}"
echo "Tokens per Step: ${N_TOKENS_PER_STEP}"
echo "Max Train Steps: ${MAX_TRAIN_STEPS}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Wandb Tags: ${WANDB_TAGS}"
echo "============================================"

# Run the training
accelerate launch --num_processes=${NUM_PROCESSES} \
    run_clm_no_trainer.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --dataset_name Cheng98/fineweb-edu-1.25B \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --num_train_epochs 1 \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --lr_scheduler_type linear \
    --output_dir ${OUTPUT_DIR} \
    --preprocessing_num_workers 32 \
    --trust_remote_code \
    --with_tracking \
    --report_to wandb \
    --transform_cfg ./transform_cfg.toml \
    --block_size ${BLOCK_SIZE} \
    --log_train_loss_steps 50 \
    --max_train_steps ${MAX_TRAIN_STEPS} \
    --wandb_tags ${WANDB_TAGS}