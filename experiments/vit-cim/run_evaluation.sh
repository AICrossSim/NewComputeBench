#!/bin/bash
CUDA_VISIBLE_DEVICES=1,2 python experiments/vit-finetuning/run_vit.py \
    --model_name_or_path google/vit-base-patch16-224 \
    --dataset_name imagenet \
    --do_eval \
    --output_dir ./log_eval_results \
    --per_device_eval_batch_size 64 \
    --custom_path /data/datasets/imagenet_pytorch/