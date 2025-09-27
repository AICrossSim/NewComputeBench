model="AICrossSim/clm-60m"
seed="42"

# batch_size="2"
# max_steps="50"
# gradient_accumulation_steps="48"

batch_size="2"
max_steps="1850"
gradient_accumulation_steps="8"

for learning_rate in 2e-5; do
    mkdir -p ./logs
    OUTPUT_DIR=${CX_DATA_HOME}/clm_finetuning_lora/$model/$learning_rate
    CUDA_VISIBLE_DEVICES=0,1 python ${CX_PROJECT_HOME}/experiments/llm-cim/finetuning/run_clm.py \
        --model_name_or_path ${model} \
        --dataset_name Cheng98/fineweb-edu-1.25B \
        --dataset_config_name "default" \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size ${batch_size} \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --report_to "wandb" \
        --block_size 512 \
        --do_train \
        --learning_rate ${learning_rate} \
        --max_steps ${max_steps} \
        --save_strategy "steps" \
        --save_steps 50 \
        --save_total_limit 2 \
        --bf16 \
        --dataloader_num_workers 16 \
        --preprocessing_num_workers 32 \
        --tokenizer_name HuggingFaceTB/cosmo2-tokenizer \
        --output_dir ${OUTPUT_DIR} \
        --transform_config ${CX_PROJECT_HOME}/src/aixsim_models/cim/experiments/sram.yaml \
        --logging_strategy "steps" \
        --logging_steps 50 \
        --seed ${seed} 2>&1 | tee ./logs/train_${learning_rate}.log

done
