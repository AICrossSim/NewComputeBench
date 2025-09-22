#!/bin/bash

# ViT Fine-tuning Script
# This script demonstrates how to fine-tune a ViT model using the new HfArgumentParser approach

python experiments/vit-finetuning/run_vit.py \
    --do_train \
    --model_name_or_path google/vit-base-patch16-224 \
    --dataset_name imagenet \
    --custom_path /data/datasets/imagenet_pytorch/ \
    --output_dir ./log_finetune_results \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --logging_steps 50 \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --metric_for_best_model eval_accuracy \
    --greater_is_better \
    --fp16 \
    --seed 42
    # --max_train_samples 1000 \
    # --max_eval_samples 500
