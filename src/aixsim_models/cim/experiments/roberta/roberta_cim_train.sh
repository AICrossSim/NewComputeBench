#!/bin/bash

DATA_HOME="${CX_DATA_HOME}"
PROJECT_HOME="${CX_PROJECT_HOME}"

DATASETS_PATH="${DATA_HOME}/datasets"
MODELS_PATH="${DATA_HOME}/models" # for models not from huggingface

CHECKPOINT_PATH="${DATA_HOME}/AICrossSim"

export HF_HOME="${DATA_HOME}/hf_home"
export TRANFORMERS_CACHE="${DATA_HOME}/hf_transformers"
export CACHE_DIR="${PROJECT_HOME}/.cache"

# Add custom paths to system PATH
export PYTHONPATH="${PROJECT_HOME}/acxsearch/:$PYTHONPATH"

cd ${PROJECT_HOME}
pwd


eval_list=(sram)
learning_rate_list=(5e-6)
task_list=(cola mnli mrpc qnli qqp rte sst2 stsb)
# task_list=(cola)

for eval_name in ${eval_list[@]}; do
    for task_name in ${task_list[@]}; do
        for learning_rate in ${learning_rate_list[@]}; do
            model_name=JeremiahZ/roberta-base-${task_name}
            CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} \
            python -u -m torch.distributed.launch --master_port=14400 --nproc_per_node=${PROC_NUM} --nnodes=1 --node_rank=0 --use_env \
            acxsearch/roberta_eval/run_glue_training.py \
                --model_name_or_path ${model_name} \
                --cim True \
                --cim_config_path ${PROJECT_HOME}/experiments/${eval_name}.yaml \
                --task_name ${task_name} \
                --data_cache_dir ${DATASETS_PATH} \
                --output_dir ${CHECKPOINT_PATH}/roberta_train/${eval_name}_${learning_rate}/$task_name \
                --do_train \
                --do_eval \
                --per_device_train_batch_size 4 \
                --per_device_eval_batch_size 32 \
                --num_train_epochs 10 \
                --learning_rate ${learning_rate} \
		        --save_strategy "no" \
                --seed 42
        done
    done
done
