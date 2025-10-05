#!/bin/bash

# ViT CIM Evaluation Script
# This script demonstrates how to evaluate a ViT model using the new HfArgumentParser approach

CUDA_VISIBLE_DEVICES=1,2 python experiments/vit-cim/run_vit.py \
    --model_name_or_path google/vit-base-patch16-224 \
    --dataset_name imagenet \
    --cim_config_path ./experiments/llm-cim/configs/sram.yaml \
    --do_eval \
    --output_dir ./log_eval_results \
    --per_device_eval_batch_size 64 \
    --custom_path /data/datasets/imagenet_pytorch/