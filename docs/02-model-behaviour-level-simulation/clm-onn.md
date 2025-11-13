# Scaling Optical Transformers on Causal Language Models

After the [Roberta-ONN experiments](roberta-onn.md), we further scale the optical transformer experiments to causal language models (CLMs). In this tutorial, we demonstrate how to fine-tune a pre-trained CLM model using our optical transformer implementation with Mase-triton acceleration.

## Starting Point of Training

We tried out three starting points for the ONN-CLM experiments and ended up using full fine-tuning of a pre-trained CLM as our main approach.

| Starting Point | Observations| Codes |
| -------------- | ------- | ----- |
| Training an ONN CLM from scratch | 🙁 The training loss did not decrease| [link](https://github.com/AICrossSim/NewComputeBench/blob/1588586dfb4cce2aaacb218b0353ff98383fab40/experiments/llm-optical-transformer/pretrain) |
| Parameter-efficient fine-tuning a pre-trained CLM using LoRA| 🙁 Only the training loss of 60M model decreased | [link](https://github.com/AICrossSim/NewComputeBench/blob/cz/onn/experiments/llm-optical-transformer/lora_finetuning)
| Full fine-tuning a pre-trained CLM| ✅ The training loss decreases but requires a small learning rate <1e-5 | [link](https://github.com/AICrossSim/NewComputeBench/blob/cz/onn/experiments/llm-optical-transformer/continual_finetuning) |


## Environment Setup

!!! info "Environment Setup?"

    If you have not set up environments, please follow the guidelines in [Environment Setup](../env-setup.md).


## Full Fine-tuning of Pre-trained CLM with Optical Transformer

Based on our experiments, full fine-tuning of pre-trained CLM models with optical transformer shows the most promising results. The optical transformer implementation uses Mase-triton acceleration to simulate optical computing operations with configurable quantization levels and smoothing factors.

The entry point for optical transformer fine-tuning is at [`experiments/llm-optical-transformer/continual_finetuning/run_clm_no_trainer.py`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-optical-transformer/continual_finetuning/run_clm_no_trainer.py).

### Experiment Setup

#### Optical Transformer Simulation Configuration

The optical transformer configuration is controlled through a TOML file that specifies quantization parameters for both attention layers and fully-connected layers.

The configuration file can be found at [experiments/llm-optical-transformer/continual_finetuning/transform_cfg.toml](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-optical-transformer/continual_finetuning/transform_cfg.toml), which contains the following key parameters:

- `use_lora`: Set to `false` for full fine-tuning
- `attention.q_levels`: Number of quantization levels (default: 256)
- `attention.q_lut_min`: Minimum value for lookup table quantization (default: 0.020040)
- `attention.q_smooth_factor`: Smoothing factor for quantization statistics (default: 0.9)
- `attention.q_init_seed`: Random seed for initialization (default: 0)
- `attention.q_bypass`: Whether to bypass quantization in attention layers (default: false)
- `fc` has similar parameters for fully-connected layers

#### Training Setup

| Setting | Description |
| ------- | ----------- |
| Pre-trained Model | `AICrossSim/clm` series |
| Dataset | `Cheng98/fineweb-edu-1.25B`. We created a subset of CLM's pretraining dataset for convenience. |
| Fine-tuning tokens | 22 * N_params / 100 tokens |
| Learning rate | We sweep from 1e-7 to 1e-5 depending on model size. Larger models require smaller learning rates for stability. |
| Effective batch size | 16. Controlled through gradient accumulation steps and number of processes. |

!!! Info
    Detailed experiment configurations can be found in the provided Wandb logs.

### Fine-tuning with Optical Transformer

#### Basic Fine-tuning Command

The main script [`run_clm_no_trainer.py`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-optical-transformer/continual_finetuning/run_clm_no_trainer.py) supports all standard Hugging Face training arguments plus optical transformer-specific configurations:

```bash
accelerate launch --num_processes=1 \
    run_clm_no_trainer.py \
    --model_name_or_path "AICrossSim/clm-60m" \
    --dataset_name "Cheng98/fineweb-edu-1.25B" \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type linear \
    --output_dir "./output/clm-60m-optical" \
    --preprocessing_num_workers 32 \
    --trust_remote_code \
    --with_tracking \
    --report_to wandb \
    --transform_cfg ./transform_cfg.toml \
    --block_size 1024 \
    --log_train_loss_steps 50
```

!!! warning "Learning Rate Selection"
    Based on our experiments, optical transformer fine-tuning requires very small learning rates (< 1e-5) to achieve stable training. Using larger learning rates may cause training instability. The larger the model, the smaller learning rate is recommended. A failed run may produce loss divergence like the following:

    <figure markdown="span">
      ![Loss Divergence](../images/onn/onn-clm-400m-failed.png){ width="600" }
      <figcaption>CLM-400M with learning rate too high</figcaption>
    </figure>

#### Using Bash Script

For convenience, we provide a parameterized shell script [`fine-tune-ot-clm.sh`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-optical-transformer/continual_finetuning/fine-tune-ot-clm.sh) that automatically calculates training steps and sets up appropriate configurations:

```bash
# Basic usage with default parameters
./fine-tune-ot-clm.sh

# Customized parameters
# Usage: ./fine-tune-ot-clm.sh [num_processes] [model_name_or_path] [per_device_train_batch_size] [learning_rate] [weight_decay] [gradient_accumulation_steps] [block_size]

./fine-tune-ot-clm.sh 2 "AICrossSim/clm-200m" 4 "1e-5" 0.01 4 1024
```

The script automatically:
1. Calculates the appropriate number of training steps based on model size
2. Sets up output directories with descriptive names
3. Configures Wandb logging with relevant tags
4. Applies the optical transformer configuration

#### Hyperparameter Sweeping

For learning rate exploration, use the [`sweep.sh`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-optical-transformer/continual_finetuning/sweep.sh) script:

```bash
# Edit sweep.sh to configure your desired learning rate ranges
./sweep.sh
```

## Results

Here we pick the training traces with smallest final training loss for each model size:

<figure markdown="span">
  ![ONN-CLM Results](../images/onn/onn-clm-full-ft.png){ width="800" }
  <figcaption>Optical Transformer Fine-tuning Results on CLM Models</figcaption>
</figure>

The results indicate that full fine-tuning of pretrained optical CLM models does not scale as well as standard CLM fine-tuning. We only observe moderate improvements in training loss for smaller models (60M -> 200M), **while larger models (400M, 600M) even show degradation in performance**.

!!! Info
    The traces above are smoothed for better visualization, and can be found in full detail in the Wandb logs.


    | 60M | 200M | 400M | 600M |
    | ---- | ---- | ---- | ---- |
    | [Wandb Log](https://wandb.ai/cz98/OT-CLM-full-fine-tune/runs/2vs467kj)| [Wandb Log](https://wandb.ai/cz98/OT-CLM-full-fine-tune/runs/3kxuoe4x) |[Wandb Log](https://wandb.ai/cz98/OT-CLM-full-fine-tune/runs/b92g3elq) |[Wandb Log](https://wandb.ai/cz98/OT-CLM-full-fine-tune/runs/2ss0tb5h) |


    More traces with various learning rates can be found at [Wandb Project: OT-CLM-full-ft](https://wandb.ai/cz98/OT-CLM-full-fine-tune).
