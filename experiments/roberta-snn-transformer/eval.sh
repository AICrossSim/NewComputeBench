# Evaluate the baseline ANN model without any transformation or fine-tuning
MODEL_NAME="JeffreyWong/roberta-base-relu-mrpc"
BATCH_SIZE=32
TASK_NAME="mrpc"
TRANSFORM_CONFIG="/home/thw20/projects/NewComputeBench/experiments/roberta-snn-transformer/transform_cfg.yaml"
python run_glue.py \
    --model_name_or_path ${MODEL_NAME} \
    --task_name ${TASK_NAME} \
    --do_eval \
    --max_seq_length 128 \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --output_dir ./output/${TASK_NAME}_eval \
    --transform_config ${TRANSFORM_CONFIG} \
    --overwrite_output_dir


# Evaluate the converted SNN model in digital form with the fine-tuned weights loaded
MODEL_NAME="JeffreyWong/roberta-base-relu-mrpc"
BATCH_SIZE=32
TASK_NAME="mrpc"
MODEL_WEIGHTS="./output/${TASK_NAME}_snn"
TRANSFORM_CONFIG="/home/thw20/projects/NewComputeBench/experiments/roberta-snn-transformer/transform_cfg.yaml"
python run_glue.py \
    --model_name_or_path ${MODEL_NAME} \
    --task_name ${TASK_NAME} \
    --do_eval \
    --max_seq_length 128 \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --output_dir ./output/${TASK_NAME}_eval \
    --transform_config ${TRANSFORM_CONFIG} \
    --model_weights_path ${MODEL_WEIGHTS} \
    --overwrite_output_dir


# Convert the model to spiking equivalent form and save the converted model
MODEL_NAME="JeffreyWong/roberta-base-relu-mrpc"
BATCH_SIZE=32
TASK_NAME="mrpc"
TRANSFORM_CONFIG="/home/thw20/projects/NewComputeBench/experiments/roberta-snn-transformer/transform_cfg.yaml"
python run_glue.py \
    --model_name_or_path ${MODEL_NAME} \
    --task_name ${TASK_NAME} \
    --do_eval \
    --max_seq_length 128 \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --output_dir ./output/${TASK_NAME}_eval \
    --transform_config ${TRANSFORM_CONFIG} \
    --convert_to_snn \
    --overwrite_output_dir