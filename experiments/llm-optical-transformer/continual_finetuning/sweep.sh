#!/bin/bash

# Hyperparameter sweep for fine-tune-ot-clm.sh
# Sweeps over learning rates while keeping other parameters fixed

# Fixed parameters
NUM_PROCESSES=1
MODEL_NAME_OR_PATH="AICrossSim/clm-600m"
PER_DEVICE_TRAIN_BATCH_SIZE=4
WEIGHT_DECAY=0.01
GRADIENT_ACCUMULATION_STEPS=4
BLOCK_SIZE=1024

# Learning rates to sweep
LEARNING_RATES=("1.00e-07" "5e-07" "7.5e-07")

echo "Starting hyperparameter sweep for fine-tune-ot-clm.sh"
echo "Fixed parameters:"
echo "  NUM_PROCESSES: ${NUM_PROCESSES}"
echo "  MODEL_NAME_OR_PATH: ${MODEL_NAME_OR_PATH}"
echo "  PER_DEVICE_TRAIN_BATCH_SIZE: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "  WEIGHT_DECAY: ${WEIGHT_DECAY}"
echo "  GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}"
echo "  BLOCK_SIZE: ${BLOCK_SIZE}"
echo ""
echo "Sweeping learning rates: ${LEARNING_RATES[@]}"
echo "============================================"
# Loop through learning rates

for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
  echo ""
  echo "Running experiment with learning rate: ${LEARNING_RATE}"
  echo "============================================"

  # Run the fine-tuning script with current learning rate
  ./fine-tune-ot-clm.sh \
    ${NUM_PROCESSES} \
    ${MODEL_NAME_OR_PATH} \
    ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    ${LEARNING_RATE} \
    ${WEIGHT_DECAY} \
    ${GRADIENT_ACCUMULATION_STEPS} \
    ${BLOCK_SIZE}

  # Check if the previous command succeeded
  if [ $? -eq 0 ]; then
    echo "✓ Experiment with learning rate ${LEARNING_RATE} completed successfully"
  else
    echo "✗ Experiment with learning rate ${LEARNING_RATE} failed"
    exit 1
  fi

  echo "============================================"
done

echo ""
echo "Hyperparameter sweep completed!"
