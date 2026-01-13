# RoBERTa Optical Transformer

This tutorial demonstrates how to apply optical transformer modifications to RoBERTa models for sequence classification tasks. The optical transformer implementation simulates photonic computing operations with quantization-aware attention mechanisms and linear layers.

## Overview

- **Optical Transform**: Applies optical computing simulation to RoBERTa models by replacing standard attention and linear layers with optical transformer equivalents.
    - The entry point is at [`experiments/roberta-optical-transformer/run_glue.py`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/roberta-optical-transformer/run_glue.py).
- **Optical Attention**: Custom attention mechanism with quantization-aware operations simulating optical matrix operations.
    - Implemented in [`src/aixsim_models/optical_compute/optical_transformer/fine_tune/ot_roberta.py`](https://github.com/AICrossSim/NewComputeBench/blob/master/src/aixsim_models/optical_compute/optical_transformer/fine_tune/ot_roberta.py).
- **GLUE Task Support**: Fine-tune and evaluate optical RoBERTa models on GLUE benchmark tasks.
    - Configuration scripts available at [`experiments/roberta-optical-transformer/finetune_base.sh`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/roberta-optical-transformer/finetune_base.sh).

The optical transformer simulation uses custom triton kernels from [`mase-triton`](https://pypi.org/project/mase-triton/) to accelerate quantization-aware operations:
- **Optical Attention**: Implements quantized matrix operations for Q, K, V projections with configurable quantization levels.
- **Optical Linear**: Replaces standard linear layers with quantization-aware optical equivalents.
- **Quantization Parameters**: Supports configurable quantization levels, smoothing factors, noise injection, and bypass modes.

## Environment Setup

!!! info "Environment Setup?"

    If you have not set up environments, please follow the guidelines in [Environment Setup](../env-setup.md).

## Optical Transform Configuration

The optical transformer behavior is controlled through a YAML configuration file that specifies quantization parameters for both attention (`attn`) and fully connected (`fc`) layers.

### Configuration Parameters

The transform configuration includes the following parameters:

- `q_levels`: Number of quantization levels (default: 256)
- `q_lut_min`: Minimum lookup table value for quantization (default: 0.020040)
- `q_quantiles`: Optional quantile-based range setting (default: null)
- `q_smooth_factor`: Smoothing factor for statistics updates (default: 0.9)
- `q_init_seed`: Random seed for initialization (default: 0)
- `q_bypass`: Whether to bypass optical transform (default: false)

### Default Configuration

```yaml
# experiments/roberta-optical-transformer/transform_cfg.yaml
"attn":
  q_levels: 256
  q_lut_min: 0.020040
  q_quantiles: null
  q_smooth_factor: 0.9
  q_init_seed: 0
  q_bypass: false
"fc":
  q_levels: 256
  q_lut_min: 0.020040
  q_quantiles: null
  q_smooth_factor: 0.9
  q_init_seed: 0
  q_bypass: false
```

## Fine-tuning RoBERTa with Optical Transform

### Single Task Fine-tuning

Fine-tune an optical RoBERTa model on a specific GLUE task:

```bash
cd experiments/roberta-optical-transformer

# Set task parameters
TASK_NAME="mrpc"                                    # GLUE task (mrpc, sst2, cola, etc.)
MODEL_NAME="FacebookAI/roberta-base"                # Base model
LEARNING_RATE="2e-5"                               # Learning rate
BATCH_SIZE="16"                                    # Batch size
NUM_EPOCHS="3"                                     # Training epochs
TRANSFORM_CONFIG="transform_cfg.yaml"              # Optical transform config

python run_glue.py \
    --model_name_or_path ${MODEL_NAME} \
    --task_name ${TASK_NAME} \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --output_dir ./output/${TASK_NAME}_optical \
    --overwrite_output_dir \
    --transform_config ${TRANSFORM_CONFIG} \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --logging_steps 50 \
    --seed 42
```

### Multi-Task Fine-tuning

Fine-tune on multiple GLUE tasks using the provided shell script:

```bash
cd experiments/roberta-optical-transformer

# Configure multi-task parameters in finetune_base.sh
export USE_SINGLE_TASK=false
export TASK_LIST="stsb mrpc cola"
export LR_LIST="1e-3 2e-5 1e-5"
export MODEL_NAME="FacebookAI/roberta-base"
export BATCH_SIZE=16

# Run multi-task fine-tuning
bash finetune_base.sh
```

### Evaluation Only

Evaluate a pre-trained optical RoBERTa model without training:

```bash
python run_glue.py \
    --model_name_or_path ${MODEL_NAME} \
    --task_name ${TASK_NAME} \
    --do_eval \
    --max_seq_length 128 \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --output_dir ./output/${TASK_NAME}_eval \
    --transform_config ${TRANSFORM_CONFIG} \
    --overwrite_output_dir
```

## Baseline Comparison

To compare optical transformer performance with the original model, run without the transform configuration:

```bash
python run_glue.py \
    --model_name_or_path ${MODEL_NAME} \
    --task_name ${TASK_NAME} \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --output_dir ./output/${TASK_NAME}_baseline \
    --overwrite_output_dir \
    --evaluation_strategy epoch \
    --save_strategy epoch
```



## Results

**Post-training transform results** (Applying optical transform to a trained RoBERTa model):

| Model              | MNLI (Acc/mismatch) | QNLI (Acc) | RTE (Acc) | SST (Acc) | MRPC (Acc) | CoLA (Matt) | QQP (Acc) | STSB (P/S corr) | Avg (Avg) |
|---------------------|--------------------:|------------:|-----------:|-----------:|------------:|-------------:|-----------:|----------------:|-----------:|
| Original            | 0.8728             | 0.9244      | 0.7978     | 0.9357     | 0.9019      | 0.6232       | 0.9153     | 0.9089          | 0.8600     |
| Random              | 0.3266             | 0.4946      | 0.5271     | 0.4908     | 0.3162      | 0.0000       | 0.6318     | 0.0332          | 0.3525     |
| Optical Transformer | 0.8000             | 0.7966      | 0.4801     | 0.8704     | 0.7770      | 0.2034       | 0.9075     | 0.8485          | 0.7104     |
| SqueezeLight        | 0.3200             | 0.4961      | 0.4404     | 0.5126     | 0.5025      | 0.0213       | 0.5890     | -0.0543         | 0.3582     |


**Fine-tuning results** (Transform-aware fine-tuning on a trained RoBERTa model):

| Model                  | MNLI (Acc/mismatch) | QNLI (Acc) | RTE (Acc) | SST (Acc) | MRPC (Acc) | CoLA (Matt) | QQP (Acc) | STSB (P/S corr) | Avg (Avg) |
|-------------------------|--------------------:|------------:|-----------:|-----------:|------------:|-------------:|-----------:|----------------:|-----------:|
| Original                | 0.8728             | 0.9244      | 0.7978     | 0.9357     | 0.9019      | 0.6232       | 0.9153     | 0.9089          | 0.8600     |
| Random                  | 0.3266             | 0.4946      | 0.5271     | 0.4908     | 0.3162      | 0.0          | 0.6318     | 0.0332          | 0.3525     |
| Optical Transformer     | 0.8510             | 0.9032      | 0.5813     | 0.9140     | 0.8677      | 0.4441       | 0.9060     | 0.0332          | 0.6876     |
| SqueezeLight            | 0.3212             | 0.4961      | 0.4676     | 0.5131     | 0.5025      | 0.0          | 0.5932     | 0.0514          | 0.3681     |

**Takeaways**:
- Whether post-training transform or transform-aware fine-tuning, the optical transformer significantly outperforms SqueezeLight. This is mainly because SqueezeLight was designed for convolutional networks.
- The continual fine-tuning of the optical transformer usually yields better performance than post-training transform, but sometimes the noisy forward pass and straight-through estimator in the backward pass can break the training stability, leading to suboptimal results like STSB.
- We decide to keep the optical transformer only for future large-scale experiments.