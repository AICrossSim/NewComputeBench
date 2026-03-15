
# task list : [mnli	qnli rte sst mrpc cola qqp stsb]
# Set task parameters
TASK_NAME="mrpc"                                          # GLUE task (mrpc, sst2, cola, etc.)
MODEL_NAME="JeffreyWong/roberta-base-relu-$TASK_NAME"     # Base model
LEARNING_RATE="2e-5"                                      # Learning rate
BATCH_SIZE="64"                                           # Batch size
NUM_EPOCHS="10"                                           # Training epochs
TRANSFORM_CONFIG="./transform_cfg.yaml"                     # SNN transform config

python run_glue.py \
    --model_name_or_path ${MODEL_NAME} \
    --task_name ${TASK_NAME} \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --output_dir ./output/${TASK_NAME}_snn \
    --overwrite_output_dir \
    --transform_config ${TRANSFORM_CONFIG} \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --logging_steps 50 \
    --seed 42