model="AICrossSim/clm-60m"
batch_size="4"
max_steps="50"
gradient_accumulation_steps="24"
seed="42"

OUTPUT_DIR=${CX_DATA_HOME}/clm_peft

python -u -m torch.distributed.launch --master_port=14400 --nproc_per_node=6 --nnodes=1 --node_rank=0 --use_env \
${CX_PROJECT_HOME}/experiments/llm-cim/finetuning/run_clm.py \
    --model_name_or_path ${model} \
    --dataset_name HuggingFaceFW/fineweb-edu \
    --dataset_config_name "sample-10BT" \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --do_train \
    --report_to "wandb" \
    --learning_rate 5e-5 \
    --max_steps ${max_steps} \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --bf16 \
    --dataloader_num_workers 16 \
    --preprocessing_num_workers 32 \
    --tokenizer_name HuggingFaceTB/cosmo2-tokenizer \
    --output_dir ${OUTPUT_DIR} \
    --transform_config ${CX_PROJECT_HOME}/src/aixsim_models/cim/experiments/sram.yaml \
    --seed ${seed}
